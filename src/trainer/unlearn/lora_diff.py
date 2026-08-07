from contextlib import contextmanager

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.lora_diff_diagnostics import (
    DiagnosticEpochCallback,
    LoraDiffDiagnostics,
)


class LoraDiff(UnlearnTrainer):
    """Oracle-gated difference between a forget LoRA and a reference.

    The reference can be either a retain LoRA or the unchanged base model.
    After training, forget evaluation uses one of three corrections:

        linear:      base - strength * (forget - reference)
        suppression: base - strength * relu(forget - reference)
        rank:         replace the top-k positively divergent token logits

    while retain and holdout evaluation use the unchanged base model.
    """

    def __init__(
        self,
        rank=16,
        lora_alpha=16,
        lora_dropout=0.0,
        strength=1.0,
        correction_mode="linear",
        difference_reference="retain",
        rank_k=20,
        diagnostic_strengths=None,
        diagnostic_epochs=None,
        diagnostic_retain_model_path=None,
        diagnostic_retain_strength=4.0,
        model=None,
        *args,
        **kwargs,
    ):
        def make_lora_config():
            return LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules="all-linear",
            )

        difference_reference = difference_reference.lower()
        if difference_reference not in {"retain", "base"}:
            raise ValueError("difference_reference must be 'retain' or 'base'")

        model = get_peft_model(model, make_lora_config(), adapter_name="forget")
        if difference_reference == "retain":
            model.add_adapter("retain", make_lora_config())
            model.base_model.set_adapter(["forget", "retain"])

        super().__init__(*args, model=model, **kwargs)
        if not self.label_names:
            self.label_names = ["labels"]
        self.model_accepts_loss_kwargs = False
        self.strength = strength
        self.correction_mode = correction_mode.lower()
        self.difference_reference = difference_reference
        self.rank_k = rank_k
        self.diagnostic_strengths = (
            [float(value) for value in diagnostic_strengths]
            if diagnostic_strengths
            else []
        )
        self.diagnostic_epochs = (
            [int(value) for value in diagnostic_epochs] if diagnostic_epochs else []
        )
        self.diagnostic_retain_model_path = diagnostic_retain_model_path
        self.diagnostic_retain_strength = float(diagnostic_retain_strength)
        if self.correction_mode not in {"linear", "suppression", "rank"}:
            raise ValueError(
                "correction_mode must be 'linear', 'suppression', or 'rank'"
            )
        if self.rank_k < 1:
            raise ValueError("rank_k must be positive")
        self._trained = False
        self._training = False
        self._is_forget = False
        self._computing_adapter_logits = False
        self.diagnostics = LoraDiffDiagnostics(
            self,
            retain_model_path=diagnostic_retain_model_path,
            retain_strength=self.diagnostic_retain_strength,
        )

        # Evaluation datasets set this context before calling the model.
        self.model.unlearn_classifier_context = self.classifier_context
        self.model.lora_diff_diagnostic_batch = self.diagnostics.batch
        if self.diagnostic_epochs:
            self.add_callback(DiagnosticEpochCallback(self.diagnostic_epochs))

    @contextmanager
    def classifier_context(self, is_forget):
        previous = self._is_forget
        self._is_forget = bool(is_forget)
        try:
            yield
        finally:
            self._is_forget = previous

    @contextmanager
    def _use_adapter(self, name):
        previous = list(self.model.active_adapters)
        self.model.set_adapter(name)
        try:
            yield
        finally:
            self.model.base_model.set_adapter(previous)

    def _adapter_forward(self, model, inputs, adapter_name):
        with self._use_adapter(adapter_name):
            return model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                use_cache=False,
                return_dict=True,
            )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_outputs = self._adapter_forward(model, inputs["forget"], "forget")
        loss = forget_outputs.loss
        if self.difference_reference == "retain":
            retain_outputs = self._adapter_forward(model, inputs["retain"], "retain")
            loss = loss + retain_outputs.loss
        return (loss, forget_outputs) if return_outputs else loss

    def train(self, *args, **kwargs):
        self._trained = False
        self._training = True
        try:
            output = super().train(*args, **kwargs)
        finally:
            self._training = False
        self._trained = True
        return output

    @staticmethod
    def _causal_lm_loss(logits, labels):
        return F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    def _apply_correction(self, base_logits, forget_logits, reference_logits):
        divergence = forget_logits - reference_logits

        if self.correction_mode == "linear":
            return base_logits - self.strength * divergence

        if self.correction_mode == "suppression":
            return base_logits - self.strength * F.relu(divergence)

        k = min(self.rank_k, base_logits.shape[-1])
        top_divergence, token_ids = divergence.topk(k, dim=-1)
        selected_logits = base_logits.gather(-1, token_ids)
        kth_logit = base_logits.topk(k, dim=-1).values[..., -1:]
        replacements = kth_logit.expand_as(selected_logits)
        replacements = torch.where(
            top_divergence > 0, replacements, selected_logits
        )
        return base_logits.scatter(-1, token_ids, replacements)

    @contextmanager
    def _correct_forget_logits(self):
        """Install the small inference-only correction used by evaluators."""
        base_model = self.model.get_base_model()
        previous_adapters = list(self.model.active_adapters)
        original_generate = self.model.generate

        def generate_without_cache(*args, **kwargs):
            kwargs["use_cache"] = False
            return original_generate(*args, **kwargs)

        def correction_hook(module, args, kwargs, output):
            if (
                self._computing_adapter_logits
                or not self._is_forget
                or getattr(output, "_lora_diff_corrected", False)
            ):
                return output

            branch_kwargs = dict(kwargs)
            labels = branch_kwargs.pop("labels", None)
            branch_kwargs["use_cache"] = False
            branch_kwargs["return_dict"] = True

            self._computing_adapter_logits = True
            self.model.enable_adapter_layers()
            try:
                with self._use_adapter("forget"):
                    forget_logits = module(*args, **branch_kwargs).logits
                if self.difference_reference == "retain":
                    with self._use_adapter("retain"):
                        reference_logits = module(*args, **branch_kwargs).logits
                else:
                    reference_logits = output.logits
            finally:
                self.model.disable_adapter_layers()
                self._computing_adapter_logits = False

            output.logits = self._apply_correction(
                output.logits, forget_logits, reference_logits
            )
            if labels is not None:
                output.loss = self._causal_lm_loss(output.logits, labels)
            output._lora_diff_corrected = True
            return output

        # PEFT forward() bypasses hooks on the wrapped model, while generate()
        # bypasses hooks on the PEFT wrapper. Cover both paths; the marker above
        # prevents a correction from being applied twice.
        handles = [
            self.model.register_forward_hook(correction_hook, with_kwargs=True),
            base_model.register_forward_hook(correction_hook, with_kwargs=True),
        ]

        # Recomputing the complete prefix keeps all-linear LoRA generation
        # correct without maintaining three separate KV caches.
        self.model.generate = generate_without_cache
        try:
            with self.model.disable_adapter():
                yield
        finally:
            for handle in handles:
                handle.remove()
            self.model.generate = original_generate
            self.model.base_model.set_adapter(previous_adapters)

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", trial=None
    ):
        correction_ready = self._trained or (
            self._training and self.state.global_step > 0
        )
        if not correction_ready:
            with self.model.disable_adapter():
                return super().evaluate(
                    eval_dataset, ignore_keys, metric_key_prefix, trial
                )

        if self.diagnostic_strengths:
            self.diagnostics.clear_batch_cache()
            epoch_prefix = ""
            if self._training:
                epoch_prefix = f"epoch_{round(self.state.epoch)}_"

            original_strength = self.strength
            metrics = {}
            try:
                for strength in self.diagnostic_strengths:
                    self.strength = strength
                    strength_slug = f"{strength:g}".replace(".", "p")
                    label = f"{epoch_prefix}strength_{strength_slug}"
                    with self._correct_forget_logits():
                        strength_metrics = super().evaluate(
                            eval_dataset,
                            ignore_keys,
                            f"{metric_key_prefix}_{label}",
                            trial,
                            output_subdir=label,
                        )
                    metrics.update(strength_metrics)
                    self.diagnostics.log_evaluation_artifact(label, trial)
            finally:
                self.strength = original_strength
            self.diagnostics.run_retain_direction()
            return metrics

        with self._correct_forget_logits():
            metrics = super().evaluate(
                eval_dataset, ignore_keys, metric_key_prefix, trial
            )
        self.diagnostics.run_retain_direction()
        return metrics
