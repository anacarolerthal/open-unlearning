import statistics
from contextlib import contextmanager
from copy import deepcopy
from math import isnan
from typing import ClassVar

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model

from trainer.unlearn.base import UnlearnTrainer


def _safe_mean(values):
    values = [v for v in values if not isnan(v)]
    return statistics.fmean(values) if values else float("nan")


def _fraction_above(values, threshold):
    values = [v for v in values if not isnan(v)]
    return (
        sum(1 for v in values if v > threshold) / len(values)
        if values
        else float("nan")
    )


class _HookGroup:
    def __init__(self, handles, cleanup=None):
        self.handles = handles
        self.cleanup = cleanup

    def remove(self):
        for handle in self.handles:
            handle.remove()
        if self.cleanup is not None:
            self.cleanup()


class UnlearnHead(UnlearnTrainer):
    """Two Hugging Face PEFT all-linear LoRA adapters on a frozen model.

    The historical handler name is retained for config compatibility, but the
    method no longer trains vocabulary heads. A forget adapter and a retain
    adapter each specialize the complete frozen model. At inference their
    logit contrast is subtracted from the base logits.

    ``classifier="likelihood_ratio"`` uses the original prompt likelihood
    router. ``classifier="oracle"`` uses evaluator-provided dataset provenance
    and therefore activates exactly for forget examples.
    """

    CLASSIFIERS: ClassVar[frozenset[str]] = frozenset(
        {"likelihood_ratio", "oracle"}
    )

    def __init__(
        self,
        rank=16,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lam=1.0,
        classifier="likelihood_ratio",
        epsilon=0.05,
        calibrate=False,
        prompt_loss_weight=1.0,
        diagnostics_max_samples=128,
        model=None,
        *args,
        **kwargs,
    ):
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank!r}")
        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")
        if not 0 <= lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {lora_dropout}")
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if classifier not in self.CLASSIFIERS:
            raise ValueError(
                f"classifier must be one of {sorted(self.CLASSIFIERS)}, "
                f"got {classifier!r}"
            )
        if classifier == "oracle" and calibrate:
            raise ValueError("calibrate is not applicable to the oracle classifier")
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if prompt_loss_weight < 0:
            raise ValueError("prompt_loss_weight must be non-negative")
        if not isinstance(diagnostics_max_samples, int) or diagnostics_max_samples <= 0:
            raise ValueError("diagnostics_max_samples must be a positive integer")

        if model is None:
            raise ValueError("model must be provided")
        if hasattr(model, "peft_config"):
            raise ValueError(
                "UnlearnHead expects an unwrapped base model, not an existing "
                "PEFT model."
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
        )
        model = get_peft_model(
            model, lora_config, adapter_name="forget"
        )
        model.add_adapter("retain", deepcopy(lora_config))
        # PEFT requires all trainable adapters to be active before Trainer
        # creates its optimizer. Individual forwards still select one adapter.
        model.base_model.set_adapter(["forget", "retain"])

        super().__init__(*args, model=model, **kwargs)
        if not self.label_names:
            self.label_names = ["labels"]
        # The custom loss is already token-normalized and does not consume
        # model loss kwargs. This restores Trainer's gradient-accumulation
        # scaling for models such as Llama whose forward accepts **kwargs.
        self.model_accepts_loss_kwargs = False
        self.lam = lam
        self.classifier = classifier
        self.epsilon = epsilon
        self.calibrate = calibrate
        self.prompt_loss_weight = prompt_loss_weight
        self.diagnostics_max_samples = diagnostics_max_samples
        if classifier == "oracle":
            self.tau = 0.5
        else:
            self.tau = None if calibrate else 0.0
        self._answer_loss_sum = 0.0
        self._answer_loss_count = 0
        self._oracle_mask = None
        self._inside_correction_branch = False
        self._adapters_trained = False

        # Metrics execute directly against the model. Exposing this context
        # manager lets the evaluator supply the provenance required by the
        # oracle classifier without adding unsupported kwargs to model.forward.
        self.model.unlearn_classifier_context = self.classifier_context

        self.retain_cal = getattr(self.train_dataset, "retain", None)
        self.forget_cal = getattr(self.train_dataset, "forget", None)

    @contextmanager
    def _using_adapter(self, adapter_name):
        if adapter_name not in (None, "forget", "retain"):
            raise ValueError(f"Unknown adapter {adapter_name!r}")
        if adapter_name is None:
            with self.model.disable_adapter():
                yield
            return

        previous = list(self.model.active_adapters)
        self.model.set_adapter(adapter_name)
        try:
            yield
        finally:
            if len(previous) == 1:
                self.model.set_adapter(previous[0])
            else:
                self.model.base_model.set_adapter(previous)

    @contextmanager
    def classifier_context(self, is_forget):
        """Temporarily provide dataset provenance to the oracle classifier."""
        previous = self._oracle_mask
        if is_forget is None:
            self._oracle_mask = None
        elif torch.is_tensor(is_forget):
            self._oracle_mask = is_forget.detach().to(dtype=torch.bool)
        elif isinstance(is_forget, (list, tuple)):
            self._oracle_mask = torch.tensor(is_forget, dtype=torch.bool)
        else:
            self._oracle_mask = torch.tensor([bool(is_forget)], dtype=torch.bool)
        try:
            yield
        finally:
            self._oracle_mask = previous

    def log(self, logs, start_time=None):
        if self._answer_loss_count:
            logs["unlearn_head/answer_loss"] = (
                self._answer_loss_sum / self._answer_loss_count
            )
            self._answer_loss_sum = 0.0
            self._answer_loss_count = 0
        super().log(logs, start_time)

    def _adapter_outputs(self, model, inputs, adapter_name):
        with self._using_adapter(adapter_name):
            return model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            )

    @staticmethod
    def _ce_loss(logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss = F.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100, reduction="sum"
        )
        valid_count = (flat_labels != -100).sum()
        return loss / valid_count.clamp(min=1)

    @staticmethod
    def _prompt_labels(input_ids, labels, attention_mask):
        prompt_mask = (labels == -100) & attention_mask.bool()
        return input_ids.masked_fill(~prompt_mask, -100)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_outputs = self._adapter_outputs(model, forget_inputs, "forget")
        forget_answer_loss = self._ce_loss(
            forget_outputs.logits, forget_inputs["labels"]
        )
        forget_prompt_labels = self._prompt_labels(
            forget_inputs["input_ids"],
            forget_inputs["labels"],
            forget_inputs["attention_mask"],
        )
        forget_prompt_loss = self._ce_loss(forget_outputs.logits, forget_prompt_labels)

        retain_inputs = inputs["retain"]
        retain_outputs = self._adapter_outputs(model, retain_inputs, "retain")
        retain_answer_loss = self._ce_loss(
            retain_outputs.logits, retain_inputs["labels"]
        )
        retain_prompt_labels = self._prompt_labels(
            retain_inputs["input_ids"],
            retain_inputs["labels"],
            retain_inputs["attention_mask"],
        )
        retain_prompt_loss = self._ce_loss(retain_outputs.logits, retain_prompt_labels)

        answer_loss = forget_answer_loss + retain_answer_loss
        prompt_loss = forget_prompt_loss + retain_prompt_loss
        loss = (answer_loss + self.prompt_loss_weight * prompt_loss) / (
            1.0 + self.prompt_loss_weight
        )

        self._answer_loss_sum += answer_loss.detach().item()
        self._answer_loss_count += 1

        return (loss, forget_outputs) if return_outputs else loss

    def _all_logits(self, input_ids, attention_mask):
        """Run the frozen base, forget adapter, and retain adapter."""
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        with torch.no_grad():
            z0 = self._adapter_outputs(self.model, inputs, None).logits
            zf = self._adapter_outputs(self.model, inputs, "forget").logits
            zr = self._adapter_outputs(self.model, inputs, "retain").logits
        return z0, zf, zr

    @staticmethod
    def _masked_ratio(tok_forget_logp, tok_retain_logp, mask):
        """Per-row mean of log p_forget - log p_retain over ``mask``."""
        diff = tok_forget_logp - tok_retain_logp
        counts = mask.sum(dim=-1).clamp(min=1)
        return (diff * mask).sum(dim=-1) / counts

    @staticmethod
    def _token_logp(logits, next_tokens):
        logp = F.log_softmax(logits, dim=-1)
        return logp[..., :-1, :].gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

    def _likelihood_ratio_score(self, input_ids, attention_mask, labels, zf, zr):
        next_tokens = input_ids[..., 1:]
        tokf = self._token_logp(zf, next_tokens)
        tokr = self._token_logp(zr, next_tokens)
        prompt_mask = (labels[..., 1:] == -100) & (attention_mask[..., 1:] == 1)
        return self._masked_ratio(tokf, tokr, prompt_mask)

    def _prompt_score(self, input_ids, attention_mask, labels):
        _, zf, zr = self._all_logits(input_ids, attention_mask)
        return self._likelihood_ratio_score(input_ids, attention_mask, labels, zf, zr)

    def _oracle_scores(self, batch_size, device):
        if self._oracle_mask is None:
            raise RuntimeError(
                "The oracle classifier needs example provenance. Evaluator datasets "
                "provide it automatically; pass is_forget to "
                "generate_with_correction() for manual generation."
            )
        scores = self._oracle_mask.to(device=device, dtype=torch.float32).flatten()
        if scores.numel() == 1 and batch_size != 1:
            scores = scores.expand(batch_size)
        if scores.numel() != batch_size:
            raise RuntimeError(
                "Oracle labels do not match the model batch size. Beam search is "
                "not supported; use num_beams=1."
            )
        return scores

    def _classifier_scores(self, input_ids, attention_mask, labels, zf, zr):
        if self.classifier == "oracle":
            return self._oracle_scores(input_ids.shape[0], zf.device)
        return self._likelihood_ratio_score(input_ids, attention_mask, labels, zf, zr)

    def _classifier_alpha(self, scores):
        return torch.where(
            scores > self.tau,
            torch.full_like(scores, self.lam),
            torch.zeros_like(scores),
        )

    def _example_metrics(self, item, adapter_name, is_forget):
        """Classifier score and answer NLLs for one sampled example."""
        device = next(self.model.parameters()).device
        input_ids = item["input_ids"].to(device).unsqueeze(0)
        attention_mask = item["attention_mask"].to(device).unsqueeze(0)
        labels = item["labels"].to(device).unsqueeze(0)

        z0, zf, zr = self._all_logits(input_ids, attention_mask)
        next_tokens = input_ids[..., 1:]
        tok0 = self._token_logp(z0, next_tokens)
        tokf = self._token_logp(zf, next_tokens)
        tokr = self._token_logp(zr, next_tokens)

        valid = attention_mask[..., 1:] == 1
        answer_mask = (labels[..., 1:] != -100) & valid

        def masked_mean(values, mask):
            selected = values[mask]
            return selected.mean().item() if selected.numel() else float("nan")

        with self.classifier_context(is_forget):
            score = self._classifier_scores(
                input_ids, attention_mask, labels, zf, zr
            ).item()
        alpha = self.lam if score > self.tau else 0.0
        zu = z0 - alpha * (zf - zr)

        adapter_tokens = tokf if adapter_name == "forget" else tokr
        return {
            "score": score,
            "nll_z0": -masked_mean(tok0, answer_mask),
            "nll_adapter": -masked_mean(adapter_tokens, answer_mask),
            "nll_gated": -masked_mean(self._token_logp(zu, next_tokens), answer_mask),
        }

    def _calibrate(self):
        if self.classifier != "likelihood_ratio":
            raise RuntimeError("Only the likelihood-ratio classifier is calibrated")
        self.model.eval()
        device = next(self.model.parameters()).device
        scores = []
        with torch.no_grad():
            for index in range(len(self.retain_cal)):
                item = self.retain_cal[index]
                input_ids = item["input_ids"].to(device).unsqueeze(0)
                attention_mask = item["attention_mask"].to(device).unsqueeze(0)
                labels = item["labels"].to(device).unsqueeze(0)
                scores.append(
                    self._prompt_score(input_ids, attention_mask, labels).item()
                )
        scores = torch.tensor(scores)
        self.tau = torch.quantile(scores, 1 - self.epsilon).item()
        self.log({"unlearn_head/tau": self.tau})

    def _diagnostic_indices(self, dataset):
        count = len(dataset)
        if count <= self.diagnostics_max_samples:
            return list(range(count))
        generator = torch.Generator().manual_seed(self.args.seed)
        permutation = torch.randperm(count, generator=generator)
        return permutation[: self.diagnostics_max_samples].tolist()

    def _log_diagnostics(self):
        """Classifier, adapter, and corrected-model diagnostics."""
        self.model.eval()
        forget_metrics = [
            self._example_metrics(self.forget_cal[index], "forget", True)
            for index in self._diagnostic_indices(self.forget_cal)
        ]
        retain_metrics = [
            self._example_metrics(self.retain_cal[index], "retain", False)
            for index in self._diagnostic_indices(self.retain_cal)
        ]

        forget_scores = [metric["score"] for metric in forget_metrics]
        retain_scores = [metric["score"] for metric in retain_metrics]
        forget_nll_z0 = _safe_mean([metric["nll_z0"] for metric in forget_metrics])
        forget_nll_zf = _safe_mean([metric["nll_adapter"] for metric in forget_metrics])
        retain_nll_z0 = _safe_mean([metric["nll_z0"] for metric in retain_metrics])
        retain_nll_zr = _safe_mean([metric["nll_adapter"] for metric in retain_metrics])

        metrics = {
            "unlearn_head/classifier_score_mean_forget": _safe_mean(forget_scores),
            "unlearn_head/classifier_score_mean_retain": _safe_mean(retain_scores),
            "unlearn_head/classifier_tpr_at_tau": _fraction_above(
                forget_scores, self.tau
            ),
            "unlearn_head/classifier_fpr_at_tau": _fraction_above(
                retain_scores, self.tau
            ),
            "unlearn_head/adapter_forget_nll_gap": forget_nll_zf - forget_nll_z0,
            "unlearn_head/adapter_retain_nll_gap": retain_nll_zr - retain_nll_z0,
            "unlearn_head/e2e_forget_nll_gated": _safe_mean(
                [metric["nll_gated"] for metric in forget_metrics]
            ),
            "unlearn_head/e2e_retain_nll_gated": _safe_mean(
                [metric["nll_gated"] for metric in retain_metrics]
            ),
        }
        self.log(metrics)
        return metrics

    def train(self, *args, **kwargs):
        self._adapters_trained = False
        output = super().train(*args, **kwargs)
        self._adapters_trained = True
        if self.accelerator.is_local_main_process:
            if self.calibrate and self.retain_cal is not None:
                self._calibrate()
            if self.retain_cal is not None and self.forget_cal is not None:
                self._log_diagnostics()
        return output

    @staticmethod
    def _is_prefill(kwargs):
        cache_position = kwargs.get("cache_position")
        if cache_position is None:
            return kwargs.get("past_key_values") is None
        return cache_position.numel() == 0 or cache_position[0].item() == 0

    def _install_correction_hook(self):
        """Install all-linear adapter contrast during forward/generation.

        Adapter-specific KV caches are kept separately during generation, so
        the forget and retain logits reflect complete adapted transformer
        passes rather than only an adapted output projection.
        """
        cache = {
            "alpha": None,
            "scores": None,
            "forget_past": None,
            "retain_past": None,
        }
        previous_active_adapters = list(self.model.active_adapters)
        self.model.disable_adapter_layers()
        # PeftModel.generate delegates to the wrapped Transformers model, so
        # install correction hooks on that model rather than on PeftModel.
        correction_model = self.model.get_base_model()

        def correction_hook(module, args, kwargs, output):
            if self._inside_correction_branch:
                return output
            if not hasattr(output, "logits"):
                raise RuntimeError(
                    "UnlearnHead requires return_dict=True outputs with logits."
                )

            is_prefill = self._is_prefill(kwargs)

            # Oracle routing is known before either adapter is evaluated. If
            # the whole batch is inactive, return the base output immediately.
            if is_prefill and self.classifier == "oracle":
                cache["scores"] = self._oracle_scores(
                    output.logits.shape[0], output.logits.device
                )
                cache["alpha"] = self._classifier_alpha(cache["scores"])
                if not torch.any(cache["alpha"] != 0).item():
                    return output
            elif not is_prefill:
                if (
                    cache["alpha"] is None
                    or cache["alpha"].shape[0] != output.logits.shape[0]
                ):
                    raise RuntimeError(
                        "Batch size changed between decoding steps. Beam search is "
                        "not supported; use greedy or sampling decoding."
                    )
                # Once routing rejects a prompt, its gate remains closed for
                # the complete generation, so no adapter KV cache is needed.
                if not torch.any(cache["alpha"] != 0).item():
                    return output

            branch_outputs = {}
            self._inside_correction_branch = True
            self.model.enable_adapter_layers()
            try:
                for adapter_name in ("forget", "retain"):
                    branch_kwargs = dict(kwargs)
                    branch_kwargs.pop("labels", None)
                    branch_kwargs["return_dict"] = True
                    if is_prefill:
                        branch_kwargs["past_key_values"] = None
                    else:
                        branch_past = cache[f"{adapter_name}_past"]
                        if branch_past is None:
                            raise RuntimeError(
                                "Missing adapter KV cache during decoding. Start a "
                                "new generate() call with a full prompt."
                            )
                        branch_kwargs["past_key_values"] = branch_past

                    with self._using_adapter(adapter_name):
                        branch_output = module(*args, **branch_kwargs)
                    branch_outputs[adapter_name] = branch_output
                    cache[f"{adapter_name}_past"] = getattr(
                        branch_output, "past_key_values", None
                    )
            finally:
                self.model.disable_adapter_layers()
                self._inside_correction_branch = False

            zf = branch_outputs["forget"].logits
            zr = branch_outputs["retain"].logits

            if is_prefill and self.classifier != "oracle":
                input_ids = kwargs.get("input_ids")
                if input_ids is None and args:
                    input_ids = args[0]
                if input_ids is None:
                    raise RuntimeError(
                        "The correction hook needs input_ids at the prefill step."
                    )
                attention_mask = kwargs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                labels = kwargs.get("labels")
                if labels is None:
                    labels = torch.full_like(input_ids, -100)
                scores = self._classifier_scores(
                    input_ids, attention_mask, labels, zf, zr
                )
                cache["scores"] = scores
                cache["alpha"] = self._classifier_alpha(scores)

                # The adapter prefill was needed to classify the prompt, but
                # no correction or adapter decoding is needed when it rejects.
                if not torch.any(cache["alpha"] != 0).item():
                    return output

            alpha = cache["alpha"].view(-1, *([1] * (zf.dim() - 1)))
            output.logits = output.logits - alpha.to(zf.device) * (zf - zr)
            labels = kwargs.get("labels")
            if labels is not None:
                # Correction hooks are installed only for post-training
                # evaluation. Keep loss consistent with the corrected logits
                # for callers that consume the model-provided loss.
                output.loss = self._ce_loss(output.logits, labels)
            return output

        hook_targets = [self.model, correction_model]
        handles = []
        for hook_target in hook_targets:
            handles.append(
                hook_target.register_forward_hook(
                    correction_hook, with_kwargs=True
                )
            )

        def restore_adapter_state():
            self.model.enable_adapter_layers()
            if len(previous_active_adapters) == 1:
                self.model.set_adapter(previous_active_adapters[0])
            else:
                self.model.base_model.set_adapter(previous_active_adapters)

        return _HookGroup(
            handles, cleanup=restore_adapter_state
        ), cache

    def generate_with_correction(
        self, prompts, max_new_tokens=64, is_forget=None, **generate_kwargs
    ):
        """Generate with adapter correction for one prompt or a prompt list.

        ``is_forget`` is required only for the oracle classifier. It can be a
        bool for a single prompt or one bool per prompt.
        """
        if not self._adapters_trained:
            raise RuntimeError("Call train() before generating with correction.")
        if self.tau is None:
            raise RuntimeError("Classifier isn't calibrated yet; call train() first.")
        if self.classifier == "oracle" and is_forget is None:
            raise RuntimeError(
                "Oracle generation requires is_forget=True/False for each prompt."
            )

        single = isinstance(prompts, str)
        prompt_list = [prompts] if single else list(prompts)
        tokenizer = self.processing_class
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            encoding = tokenizer(prompt_list, return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = original_padding_side

        device = next(self.model.parameters()).device
        encoding = encoding.to(device)
        self.model.eval()
        handle, cache = self._install_correction_hook()
        try:
            with self.classifier_context(is_forget), torch.no_grad():
                output_ids = self.model.generate(
                    **encoding,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )
        finally:
            handle.remove()

        texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results = [
            {
                "text": text,
                "score": score.item(),
                "alpha": alpha.item(),
                "tau": self.tau,
                "classifier": self.classifier,
            }
            for text, score, alpha in zip(texts, cache["scores"], cache["alpha"])
        ]
        return results[0] if single else results

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", trial=None
    ):
        """Wrap evaluation so every metric receives corrected logits."""
        if not self._adapters_trained or self.tau is None:
            with self.model.disable_adapter():
                return super().evaluate(
                    eval_dataset, ignore_keys, metric_key_prefix, trial
                )

        handle, _ = self._install_correction_hook()
        try:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, trial)
        finally:
            handle.remove()
