from contextlib import contextmanager

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model

from trainer.unlearn.base import UnlearnTrainer


class LoraDiff(UnlearnTrainer):
    """Oracle-gated difference between forget and retain LoRA adapters.

    Both adapters are trained with ordinary language-modeling loss. After
    training, forget evaluation uses one of three corrections:

        linear:      base - strength * (forget - retain)
        suppression: base - strength * relu(forget - retain)
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
        rank_k=20,
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

        model = get_peft_model(model, make_lora_config(), adapter_name="forget")
        model.add_adapter("retain", make_lora_config())
        model.base_model.set_adapter(["forget", "retain"])

        super().__init__(*args, model=model, **kwargs)
        if not self.label_names:
            self.label_names = ["labels"]
        self.model_accepts_loss_kwargs = False
        self.strength = strength
        self.correction_mode = correction_mode.lower()
        self.rank_k = rank_k
        if self.correction_mode not in {"linear", "suppression", "rank"}:
            raise ValueError(
                "correction_mode must be 'linear', 'suppression', or 'rank'"
            )
        if self.rank_k < 1:
            raise ValueError("rank_k must be positive")
        self._trained = False
        self._is_forget = False
        self._computing_adapter_logits = False

        # Evaluation datasets set this context before calling the model.
        self.model.unlearn_classifier_context = self.classifier_context

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
        retain_outputs = self._adapter_forward(model, inputs["retain"], "retain")
        loss = forget_outputs.loss + retain_outputs.loss
        return (loss, forget_outputs) if return_outputs else loss

    def train(self, *args, **kwargs):
        self._trained = False
        output = super().train(*args, **kwargs)
        self._trained = True
        return output

    @staticmethod
    def _causal_lm_loss(logits, labels):
        return F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    def _apply_correction(self, base_logits, forget_logits, retain_logits):
        divergence = forget_logits - retain_logits

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
            if self._computing_adapter_logits or not self._is_forget:
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
                with self._use_adapter("retain"):
                    retain_logits = module(*args, **branch_kwargs).logits
            finally:
                self.model.disable_adapter_layers()
                self._computing_adapter_logits = False

            output.logits = self._apply_correction(
                output.logits, forget_logits, retain_logits
            )
            if labels is not None:
                output.loss = self._causal_lm_loss(output.logits, labels)
            return output

        handle = base_model.register_forward_hook(
            correction_hook, with_kwargs=True
        )

        # Recomputing the complete prefix keeps all-linear LoRA generation
        # correct without maintaining three separate KV caches.
        self.model.generate = generate_without_cache
        try:
            with self.model.disable_adapter():
                yield
        finally:
            handle.remove()
            self.model.generate = original_generate
            self.model.base_model.set_adapter(previous_adapters)

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", trial=None
    ):
        if not self._trained:
            with self.model.disable_adapter():
                return super().evaluate(
                    eval_dataset, ignore_keys, metric_key_prefix, trial
                )

        with self._correct_forget_logits():
            return super().evaluate(
                eval_dataset, ignore_keys, metric_key_prefix, trial
            )
