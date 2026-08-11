import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from peft import RoadConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Subset
from peft.tuners.road.layer import _apply_road
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.roadblock_classifier import RoadBlockClassifier
from trainer.wandb_utils import log_wandb_artifact


class RoadBlock(UnlearnTrainer):
    """Routed suppression using a RoAd probe on the language-model head."""

    def __init__(self, strength=42.0, classifier="oracle", model=None, *args, **kwargs):
        config = RoadConfig(
            task_type=TaskType.CAUSAL_LM,
            variant="road_1",
            group_size=64,
            target_modules=["lm_head"],
        )
        model = get_peft_model(model, config, adapter_name="forget")
        super().__init__(*args, model=model, **kwargs)

        if not self.label_names:
            self.label_names = ["labels"]
        self.model_accepts_loss_kwargs = False
        self.strength = float(strength)
        self.classifier = RoadBlockClassifier(
            classifier, model.config.hidden_size, self.model.device
        )
        self._trained = False
        self._is_forget = False
        self._skip_correction = False
        self._classifier_hidden = None
        self._route_cache = None
        self._reference_uploaded = False

        self.model.unlearn_classifier_context = self.classifier_context
        self.model.roadblock_diagnostic_batch = self.diagnostic_batch
        self.model.roadblock_classify_batch = self.classify_batch
        self.model.roadblock_classifier_threshold = lambda: self.classifier.threshold

    @contextmanager
    def classifier_context(self, is_forget):
        previous = self._is_forget
        self._is_forget = bool(is_forget)
        self._route_cache = None
        try:
            yield
        finally:
            self._is_forget = previous
            self._route_cache = None

    @staticmethod
    def _pool_prompt(hidden, attention_mask, labels=None):
        attention_mask = attention_mask[:, -hidden.shape[1] :].bool()
        if labels is not None and labels.shape[1] == hidden.shape[1]:
            prompt_mask = attention_mask & labels.eq(-100)
        else:
            prompt_mask = attention_mask
        empty = ~prompt_mask.any(dim=-1)
        prompt_mask[empty] = attention_mask[empty]
        weights = prompt_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    @torch.no_grad()
    def _embed_batch(self, batch):
        batch = {
            key: value.to(self.model.device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }
        decoder = self.model.get_base_model().get_decoder()
        hidden = decoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return self._pool_prompt(
            hidden, batch["attention_mask"], batch.get("labels")
        ).cpu()

    def _embed_dataset(self, dataset):
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=self.data_collator)
        was_training = self.model.training
        self.model.eval()
        embeddings = torch.cat([self._embed_batch(batch) for batch in dataloader])
        self.model.train(was_training)
        return embeddings

    def _fit_classifier(self):
        if not self.classifier.needs_embeddings:
            return
        forget = self.train_dataset.forget
        retain = self.train_dataset.retain
        retain_indices = torch.linspace(
            0, len(retain) - 1, steps=min(len(forget), len(retain))
        ).long()
        forget_embeddings = self._embed_dataset(forget)
        retain_embeddings = self._embed_dataset(Subset(retain, retain_indices))
        self.classifier.fit(forget_embeddings, retain_embeddings)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = {
            key: inputs["forget"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        outputs = model(**forget_inputs, use_cache=False, return_dict=True)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def train(self, *args, **kwargs):
        self._trained = False
        self._fit_classifier()
        output = super().train(*args, **kwargs)
        self._trained = True
        return output

    def _road_logits(self, base_logits):
        output_layer = self.model.get_base_model().get_output_embeddings()
        theta = output_layer.road_theta["forget"]
        logits = _apply_road(
            output_layer.variant["forget"],
            output_layer.group_size["forget"],
            theta,
            output_layer.road_alpha["forget"],
            base_logits.to(theta.dtype),
        )
        return logits.to(base_logits.dtype)

    def _correct(self, base_logits):
        divergence = self._road_logits(base_logits) - base_logits
        return base_logits - self.strength * F.relu(divergence)

    def _route(self, kwargs):
        attention_mask = kwargs["attention_mask"]
        incremental = attention_mask.shape[1] != self._classifier_hidden.shape[1]
        if incremental and self._route_cache is not None:
            return self._route_cache

        embeddings = self._pool_prompt(
            self._classifier_hidden, attention_mask, kwargs.get("labels")
        )
        _, route = self.classifier.predict(embeddings)
        self._route_cache = route
        return route

    @staticmethod
    def _causal_lm_loss(logits, labels):
        return F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    @contextmanager
    def _apply_forget_correction(self):
        base_model = self.model.get_base_model()
        output_layer = base_model.get_output_embeddings()

        def capture_hidden(module, args, kwargs):
            self._classifier_hidden = args[0]

        def correction_hook(module, args, kwargs, output):
            if self._skip_correction or getattr(output, "_roadblock_corrected", False):
                return output

            if self.classifier.mode == "oracle" and not self._is_forget:
                output._roadblock_corrected = True
                return output

            corrected_logits = self._correct(output.logits)
            if self.classifier.needs_embeddings:
                route = self._route(kwargs)
                output.logits = torch.where(
                    route[:, None, None], corrected_logits, output.logits
                )
            else:
                output.logits = corrected_logits
            labels = kwargs.get("labels")
            if labels is not None:
                output.loss = self._causal_lm_loss(output.logits, labels)
            output._roadblock_corrected = True
            return output

        handles = [
            self.model.register_forward_hook(correction_hook, with_kwargs=True),
            base_model.register_forward_hook(correction_hook, with_kwargs=True),
        ]
        if self.classifier.needs_embeddings:
            handles.append(
                output_layer.register_forward_pre_hook(capture_hidden, with_kwargs=True)
            )
        try:
            with self.model.disable_adapter():
                yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def classify_batch(self, inputs, is_forget):
        batch_size = inputs["input_ids"].shape[0]
        if self.classifier.mode == "oracle":
            scores = torch.full((batch_size,), float(is_forget))
            predictions = scores.bool()
        elif self.classifier.mode == "none":
            scores = torch.ones(batch_size)
            predictions = scores.bool()
        else:
            embeddings = self._embed_batch(inputs)
            scores, predictions = self.classifier.predict(embeddings)
            scores = scores.cpu()
            predictions = predictions.cpu()
        return [
            {"score": float(score), "prediction": bool(prediction)}
            for score, prediction in zip(scores, predictions)
        ]

    @torch.no_grad()
    def diagnostic_batch(self, inputs):
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        labels = inputs["labels"]
        previous_skip = self._skip_correction
        self._skip_correction = True
        try:
            base_logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).logits
        finally:
            self._skip_correction = previous_skip

        statistics = []
        for sample_logits, sample_labels in zip(base_logits[:, :-1], labels[:, 1:]):
            mask = sample_labels != -100
            targets = sample_labels[mask]
            base = sample_logits[mask]
            if targets.numel() == 0:
                statistics.append(
                    {
                        "correction_rms": 0.0,
                        "target_suppression": 0.0,
                        "target_positive_fraction": 0.0,
                        "target_top100_fraction": 0.0,
                    }
                )
                continue

            divergence = self._road_logits(base) - base
            positive = F.relu(divergence)
            target_divergence = divergence.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            target_suppression = positive.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            top_tokens = divergence.topk(min(100, divergence.shape[-1]), dim=-1).indices
            target_in_top100 = (top_tokens == targets.unsqueeze(-1)).any(dim=-1) & (
                target_divergence > 0
            )
            correction = self.strength * positive
            statistics.append(
                {
                    "correction_rms": float(correction.square().mean().sqrt()),
                    "target_suppression": float(
                        self.strength * target_suppression.mean()
                    ),
                    "target_positive_fraction": float(
                        (target_divergence > 0).float().mean()
                    ),
                    "target_top100_fraction": float(target_in_top100.float().mean()),
                }
            )
        return statistics

    def _log_evaluation_artifacts(self, trial):
        if not self.accelerator.is_local_main_process or not self.evaluators:
            return

        run_dir = self._get_output_dir(trial=trial)
        checkpoint = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        eval_dir = os.path.join(run_dir, checkpoint, "evals")
        log_wandb_artifact(eval_dir, "roadblock-evaluation", "evaluation")

        if self._reference_uploaded:
            return
        for evaluator in self.evaluators.values():
            path = evaluator.eval_cfg.get("retain_logs_path")
            if path:
                log_wandb_artifact(
                    path, "roadblock-retain-reference", "evaluation-reference"
                )
                self._reference_uploaded = True
                return

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", trial=None
    ):
        if not self._trained:
            with self.model.disable_adapter():
                return super().evaluate(
                    eval_dataset, ignore_keys, metric_key_prefix, trial
                )

        with self._apply_forget_correction():
            metrics = super().evaluate(
                eval_dataset, ignore_keys, metric_key_prefix, trial
            )
        self._log_evaluation_artifacts(trial)
        return metrics
