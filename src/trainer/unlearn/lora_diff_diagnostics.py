import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from trainer.wandb_utils import log_wandb_artifact


class DiagnosticEpochCallback(TrainerCallback):
    def __init__(self, epochs):
        self.epochs = set(epochs)

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = round(state.epoch) in self.epochs
        return control


class LoraDiffDiagnostics:
    def __init__(self, trainer, retain_model_path=None, retain_strength=4.0):
        self.trainer = trainer
        self.retain_model_path = retain_model_path
        self.retain_strength = retain_strength
        self.retain_complete = False
        self.reference_logs_uploaded = False
        self.batch_cache = {}

    def clear_batch_cache(self):
        self.batch_cache.clear()

    def log_evaluation_artifact(self, label, trial):
        trainer = self.trainer
        run_dir = trainer._get_output_dir(trial=trial)
        checkpoint = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}"
        output_dir = os.path.join(run_dir, checkpoint, "evals", label)
        log_wandb_artifact(output_dir, f"lora-diff-eval-{label}", "evaluation")
        self._log_reference_logs()

    def _log_reference_logs(self):
        if self.reference_logs_uploaded:
            return

        for evaluator in self.trainer.evaluators.values():
            path = evaluator.eval_cfg.get("retain_logs_path")
            if path:
                log_wandb_artifact(
                    path,
                    "lora-diff-retain-reference",
                    "evaluation-reference",
                )
                self.reference_logs_uploaded = True
                return

    @torch.no_grad()
    def batch(self, inputs):
        trainer = self.trainer
        inputs = {key: value.to(trainer.model.device) for key, value in inputs.items()}
        labels = inputs["labels"]
        cache_keys = [
            tuple(input_ids[attention_mask.bool()].tolist())
            for input_ids, attention_mask in zip(
                inputs["input_ids"], inputs["attention_mask"]
            )
        ]
        if trainer.correction_mode == "suppression" and all(
            key in self.batch_cache for key in cache_keys
        ):
            return [self._scale_statistics(self.batch_cache[key]) for key in cache_keys]

        forward_args = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "use_cache": False,
            "return_dict": True,
        }

        trainer._computing_adapter_logits = True
        trainer.model.enable_adapter_layers()
        try:
            with trainer.model.disable_adapter():
                base_logits = self._answer_logits(
                    trainer.model(**forward_args).logits, labels
                )
            with trainer._use_adapter("forget"):
                forget_logits = self._answer_logits(
                    trainer.model(**forward_args).logits, labels
                )
        finally:
            trainer.model.disable_adapter_layers()
            trainer._computing_adapter_logits = False

        reference_logits = base_logits
        if trainer.difference_reference == "retain":
            trainer.model.enable_adapter_layers()
            trainer._computing_adapter_logits = True
            try:
                with trainer._use_adapter("retain"):
                    reference_logits = self._answer_logits(
                        trainer.model(**forward_args).logits, labels
                    )
            finally:
                trainer.model.disable_adapter_layers()
                trainer._computing_adapter_logits = False

        target_tokens = [
            sample_targets[mask].cpu()
            for sample_targets, mask in zip(
                labels[:, 1:], labels[:, 1:] != -100
            )
        ]

        results = []
        for cache_key, sample_base, sample_forget, sample_reference, sample_targets in zip(
            cache_keys,
            base_logits,
            forget_logits,
            reference_logits,
            target_tokens,
        ):
            if sample_targets.numel() == 0:
                results.append(self._empty_statistics())
                continue

            sample_divergence = sample_forget - sample_reference
            positive_divergence = F.relu(sample_divergence)
            target_divergence = sample_divergence.gather(
                -1, sample_targets.unsqueeze(-1)
            ).squeeze(-1)
            target_unit_suppression = positive_divergence.gather(
                -1, sample_targets.unsqueeze(-1)
            ).squeeze(-1)
            top_k = min(100, sample_divergence.shape[-1])
            top_tokens = sample_divergence.topk(top_k, dim=-1).indices
            target_in_top_k = (top_tokens == sample_targets.unsqueeze(-1)).any(dim=-1)

            unit_statistics = {
                "positive_divergence_fraction": float(
                    (sample_divergence > 0).float().mean()
                ),
                "positive_divergence_rms": float(
                    positive_divergence.square().mean().sqrt()
                ),
                "unit_correction_l1": float(positive_divergence.mean()),
                "unit_correction_rms": float(
                    positive_divergence.square().mean().sqrt()
                ),
                "target_positive_fraction": float(
                    (target_divergence > 0).float().mean()
                ),
                "unit_target_suppression": float(target_unit_suppression.mean()),
                "target_top100_fraction": float(target_in_top_k.float().mean()),
            }
            if trainer.correction_mode == "suppression":
                self.batch_cache[cache_key] = unit_statistics
                results.append(self._scale_statistics(unit_statistics))
            else:
                sample_correction = trainer._apply_correction(
                    sample_base, sample_forget, sample_reference
                ) - sample_base
                results.append(
                    {
                        **self._constant_statistics(unit_statistics),
                        "correction_l1": float(sample_correction.abs().mean()),
                        "correction_rms": float(
                            sample_correction.square().mean().sqrt()
                        ),
                        "target_suppression": float(
                            -sample_correction.gather(
                                -1, sample_targets.unsqueeze(-1)
                            )
                            .squeeze(-1)
                            .mean()
                        ),
                    }
                )
        return results

    @staticmethod
    def _constant_statistics(statistics):
        return {
            key: value
            for key, value in statistics.items()
            if not key.startswith("unit_")
        }

    def _scale_statistics(self, statistics):
        strength = self.trainer.strength
        return {
            **self._constant_statistics(statistics),
            "correction_l1": strength * statistics["unit_correction_l1"],
            "correction_rms": strength * statistics["unit_correction_rms"],
            "target_suppression": strength
            * statistics["unit_target_suppression"],
        }

    @staticmethod
    def _empty_statistics():
        return {
            "positive_divergence_fraction": 0.0,
            "positive_divergence_rms": 0.0,
            "correction_l1": 0.0,
            "correction_rms": 0.0,
            "target_positive_fraction": 0.0,
            "target_suppression": 0.0,
            "target_top100_fraction": 0.0,
        }

    @staticmethod
    def _answer_logits(logits, labels):
        valid_tokens = labels[:, 1:] != -100
        return [
            sample_logits[mask].float().cpu()
            for sample_logits, mask in zip(logits[:, :-1], valid_tokens)
        ]

    @staticmethod
    def _direction_statistics(base_logits, forget_logits, retain_logits, strengths):
        candidate_delta = forget_logits - base_logits
        ideal_direction = retain_logits - base_logits
        centered_candidate = candidate_delta - candidate_delta.mean(dim=-1, keepdim=True)
        ideal_direction = ideal_direction - ideal_direction.mean(dim=-1, keepdim=True)
        candidate_suppression = F.relu(centered_candidate)
        ideal_suppression = F.relu(-ideal_direction)

        candidate_direction = -candidate_suppression
        cosine = F.cosine_similarity(
            candidate_direction.reshape(1, -1),
            ideal_direction.reshape(1, -1),
        ).item()
        norm_ratio = (
            candidate_direction.norm() / ideal_direction.norm().clamp_min(1e-12)
        ).item()

        top_k = min(100, base_logits.shape[-1])
        candidate_tokens = candidate_suppression.topk(top_k, dim=-1).indices
        ideal_tokens = ideal_suppression.topk(top_k, dim=-1).indices
        top100_overlap = (
            (candidate_tokens.unsqueeze(-1) == ideal_tokens.unsqueeze(-2))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )

        retain_log_probs = F.log_softmax(retain_logits, dim=-1)
        retain_probs = retain_log_probs.exp()

        def retain_kl(candidate_logits):
            candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
            return (
                retain_probs * (retain_log_probs - candidate_log_probs)
            ).sum(dim=-1).mean().item()

        statistics = {
            "direction_cosine": cosine,
            "direction_norm_ratio": norm_ratio,
            "top100_suppression_overlap": top100_overlap,
            "retain_kl_base": retain_kl(base_logits),
        }
        actual_suppression = F.relu(forget_logits - base_logits)
        for strength in strengths:
            slug = f"{strength:g}".replace(".", "p")
            corrected_logits = base_logits - strength * actual_suppression
            statistics[f"retain_kl_strength_{slug}"] = retain_kl(corrected_logits)
        return statistics

    @torch.no_grad()
    def run_retain_direction(self):
        trainer = self.trainer
        if self.retain_complete or not self.retain_model_path or trainer._training:
            return

        forget_data = getattr(trainer.train_dataset, "forget", None)
        if forget_data is None:
            raise ValueError("Retain-direction diagnostics require a forget dataset")

        retain_model = AutoModelForCausalLM.from_pretrained(
            self.retain_model_path,
            torch_dtype=trainer.model.dtype,
            device_map={"": trainer.model.device},
        )
        retain_model.eval()
        dataloader = DataLoader(
            forget_data,
            batch_size=1,
            collate_fn=trainer.data_collator,
        )
        strengths = trainer.diagnostic_strengths or [self.retain_strength]
        value_by_index = {}

        for index, inputs in enumerate(dataloader):
            inputs = {
                key: value.to(trainer.model.device) for key, value in inputs.items()
            }
            labels = inputs["labels"]
            forward_args = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "use_cache": False,
                "return_dict": True,
            }
            with trainer.model.disable_adapter():
                base_logits = self._answer_logits(
                    trainer.model(**forward_args).logits, labels
                )[0]
            with trainer._use_adapter("forget"):
                forget_logits = self._answer_logits(
                    trainer.model(**forward_args).logits, labels
                )[0]
            retain_logits = self._answer_logits(
                retain_model(**forward_args).logits, labels
            )[0]
            value_by_index[str(index)] = self._direction_statistics(
                base_logits, forget_logits, retain_logits, strengths
            )

        names = next(iter(value_by_index.values())).keys()
        summary = {
            name: sum(values[name] for values in value_by_index.values())
            / len(value_by_index)
            for name in names
        }
        output_dir = os.path.join(trainer.args.output_dir, "diagnostics")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "retain_direction.json")
        with open(output_path, "w") as file:
            json.dump(
                {"summary": summary, "value_by_index": value_by_index},
                file,
                indent=2,
            )
        trainer.log(
            {f"eval_retain_direction_{key}": value for key, value in summary.items()}
        )
        log_wandb_artifact(
            output_path,
            "lora-diff-retain-direction",
            "evaluation-diagnostics",
        )
        self.retain_complete = True
        del retain_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
