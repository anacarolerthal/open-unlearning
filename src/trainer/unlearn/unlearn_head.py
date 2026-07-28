import os
import statistics

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import TrainerCallback

from trainer.unlearn.base import UnlearnTrainer


def _safe_mean(values):
    values = [v for v in values if v == v]  # drop NaNs
    return statistics.fmean(values) if values else float("nan")


def _safe_std(values):
    values = [v for v in values if v == v]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _fraction_above(values, threshold):
    values = [v for v in values if v == v]
    return sum(1 for v in values if v > threshold) / len(values) if values else float("nan")


def _auc(forget_scores, retain_scores):
    forget_scores = [v for v in forget_scores if v == v]
    retain_scores = [v for v in retain_scores if v == v]
    if not forget_scores or not retain_scores:
        return float("nan")
    y_true = [1] * len(forget_scores) + [0] * len(retain_scores)
    y_score = forget_scores + retain_scores
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")


class LowRankHead(nn.Module):
    """B @ (A @ h); B zero-initialized so the head starts as a no-op."""

    def __init__(self, hidden_size, vocab_size, rank):
        super().__init__()
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, vocab_size, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, h):
        return self.B(self.A(h))


class _HeadGradClipCallback(TrainerCallback):
    """HF's built-in clipping only covers self.model.parameters() (frozen);
    this clips the actual trainable heads, since max_grad_norm otherwise
    silently does nothing for them."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if args.max_grad_norm and args.max_grad_norm > 0:
            head_params = list(self.trainer.forget_head.parameters()) + list(
                self.trainer.retain_head.parameters()
            )
            nn.utils.clip_grad_norm_(head_params, args.max_grad_norm)


class UnlearnHead(UnlearnTrainer):
    """Single-request UnlearnHead: two low-rank heads on a frozen backbone,
    gated at inference by a forget/retain likelihood-ratio router.

    By default (calibrate=False) the router uses a fixed tau=0: the gate
    opens whenever the forget head finds a prompt more likely than the
    retain head does, no calibration pass needed. Set calibrate=True to
    instead pick tau from a target false-positive rate (epsilon) on the
    retain set, via _calibrate() (unused by default, kept for later use).
    """

    def __init__(
        self, rank=16, lam=1.0, epsilon=0.05, calibrate=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lam = lam
        self.epsilon = epsilon
        self.calibrate = calibrate
        self.tau = None if calibrate else 0.0

        for p in self.model.parameters():
            p.requires_grad = False

        # Trainer attributes, not submodules of self.model, so self.model's
        # save_pretrained stays a plain reloadable base-model checkpoint.
        base_param = next(self.model.parameters())
        self.forget_head = LowRankHead(
            self.model.config.hidden_size, self.model.config.vocab_size, rank
        ).to(device=base_param.device, dtype=base_param.dtype)
        self.retain_head = LowRankHead(
            self.model.config.hidden_size, self.model.config.vocab_size, rank
        ).to(device=base_param.device, dtype=base_param.dtype)
        self.add_callback(_HeadGradClipCallback(self))

        # Diagnostics/calibration run on the training data itself (as TOFU
        # eval does), not a held-out slice.
        self.retain_cal = getattr(self.train_dataset, "retain", None)
        self.forget_cal = getattr(self.train_dataset, "forget", None)

    def create_optimizer(self):
        if self.optimizer is None:
            head_params = list(self.forget_head.parameters()) + list(
                self.retain_head.parameters()
            )
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(head_params, **optimizer_kwargs)
        return self.optimizer

    def _head_outputs(self, model, input_ids, attention_mask, head):
        with torch.no_grad():
            base_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        h = base_out.hidden_states[-1]
        base_out.logits = base_out.logits + head(h)
        return base_out

    @staticmethod
    def _ce_loss(logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_outputs = self._head_outputs(
            model,
            forget_inputs["input_ids"],
            forget_inputs["attention_mask"],
            self.forget_head,
        )
        forget_loss = self._ce_loss(forget_outputs.logits, forget_inputs["labels"])

        retain_inputs = inputs["retain"]
        retain_outputs = self._head_outputs(
            model,
            retain_inputs["input_ids"],
            retain_inputs["attention_mask"],
            self.retain_head,
        )
        retain_loss = self._ce_loss(retain_outputs.logits, retain_inputs["labels"])

        loss = forget_loss + retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    def _frozen_logits(self, input_ids, attention_mask):
        """One frozen backbone forward + both heads; no gradients."""
        with torch.no_grad():
            base_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            h = base_out.hidden_states[-1]
            z0 = base_out.logits
            zF = z0 + self.forget_head(h)
            zR = z0 + self.retain_head(h)
        return z0, zF, zR

    @staticmethod
    def _masked_ratio(tok_forget_logp, tok_retain_logp, mask):
        """Per-row mean of (tok_forget_logp - tok_retain_logp) over `mask`."""
        diff = tok_forget_logp - tok_retain_logp
        counts = mask.sum(dim=-1).clamp(min=1)
        return (diff * mask).sum(dim=-1) / counts  # 0 where a row has no masked tokens

    def _token_logp(self, logits, next_tokens):
        logp = F.log_softmax(logits, dim=-1)
        return logp[..., :-1, :].gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

    def _prompt_score(self, input_ids, attention_mask, labels):
        """s(q) per example: avg log p_F - log p_R over prompt tokens. Shape [batch]."""
        _, zF, zR = self._frozen_logits(input_ids, attention_mask)
        next_tokens = input_ids[..., 1:]
        tokF, tokR = self._token_logp(zF, next_tokens), self._token_logp(zR, next_tokens)
        prompt_mask = (labels[..., 1:] == -100) & (attention_mask[..., 1:] == 1)
        return self._masked_ratio(tokF, tokR, prompt_mask)

    def _example_metrics(self, item, head_name):
        """s(q), NLL under z0/z_head/z_U-gated for one held-out example."""
        device = next(self.model.parameters()).device
        input_ids = item["input_ids"].to(device).unsqueeze(0)
        attention_mask = item["attention_mask"].to(device).unsqueeze(0)
        labels = item["labels"].to(device).unsqueeze(0)

        z0, zF, zR = self._frozen_logits(input_ids, attention_mask)
        next_tokens = input_ids[..., 1:]
        tok0 = self._token_logp(z0, next_tokens)
        tokF = self._token_logp(zF, next_tokens)
        tokR = self._token_logp(zR, next_tokens)

        valid = attention_mask[..., 1:] == 1
        prompt_mask = (labels[..., 1:] == -100) & valid
        answer_mask = (labels[..., 1:] != -100) & valid

        def masked_mean(t, mask):
            vals = t[mask]
            return vals.mean().item() if vals.numel() > 0 else float("nan")

        score = self._masked_ratio(tokF, tokR, prompt_mask).item()

        tok_head = tokF if head_name == "forget" else tokR
        nll_z0 = -masked_mean(tok0, answer_mask)
        nll_head = -masked_mean(tok_head, answer_mask)

        alpha = self.lam if score > self.tau else 0.0
        zU = z0 - alpha * (zF - zR)
        nll_gated = -masked_mean(self._token_logp(zU, next_tokens), answer_mask)

        return {
            "score": score,
            "nll_z0": nll_z0,
            "nll_head": nll_head,
            "nll_gated": nll_gated,
        }

    def _calibrate(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        scores = []
        with torch.no_grad():
            for i in range(len(self.retain_cal)):
                item = self.retain_cal[i]
                input_ids = item["input_ids"].to(device).unsqueeze(0)
                attention_mask = item["attention_mask"].to(device).unsqueeze(0)
                labels = item["labels"].to(device).unsqueeze(0)
                scores.append(self._prompt_score(input_ids, attention_mask, labels).item())
        scores = torch.tensor(scores)
        self.tau = torch.quantile(scores, 1 - self.epsilon).item()
        self.log({"unlearn_head/tau": self.tau})

    def _log_diagnostics(self):
        """Router/head/end-to-end diagnostics on held-out data, via self.log."""
        self.model.eval()
        forget_metrics = [
            self._example_metrics(self.forget_cal[i], "forget")
            for i in range(len(self.forget_cal))
        ]
        retain_metrics = [
            self._example_metrics(self.retain_cal[i], "retain")
            for i in range(len(self.retain_cal))
        ]

        forget_scores = [m["score"] for m in forget_metrics]
        retain_scores = [m["score"] for m in retain_metrics]

        forget_nll_z0 = _safe_mean([m["nll_z0"] for m in forget_metrics])
        forget_nll_zF = _safe_mean([m["nll_head"] for m in forget_metrics])
        retain_nll_z0 = _safe_mean([m["nll_z0"] for m in retain_metrics])
        retain_nll_zR = _safe_mean([m["nll_head"] for m in retain_metrics])

        metrics = {
            # Sanity checks
            "unlearn_head/router_score_mean_forget": _safe_mean(forget_scores),
            "unlearn_head/router_score_std_forget": _safe_std(forget_scores),
            "unlearn_head/router_score_mean_retain": _safe_mean(retain_scores),
            "unlearn_head/router_score_std_retain": _safe_std(retain_scores),
            # Router metrics: is the classifier good, independent of the heads?
            "unlearn_head/router_auc": _auc(forget_scores, retain_scores),
            "unlearn_head/router_tpr_at_tau": _fraction_above(forget_scores, self.tau),
            "unlearn_head/router_fpr_at_tau": _fraction_above(retain_scores, self.tau),
            # Head metrics: do the heads edit logits enough, gate forced open?
            "unlearn_head/head_forget_nll_z0": forget_nll_z0,
            "unlearn_head/head_forget_nll_zF": forget_nll_zF,
            "unlearn_head/head_forget_nll_gap": forget_nll_zF - forget_nll_z0,
            "unlearn_head/head_retain_nll_z0": retain_nll_z0,
            "unlearn_head/head_retain_nll_zR": retain_nll_zR,
            "unlearn_head/head_retain_nll_gap": retain_nll_zR - retain_nll_z0,
            # End-to-end metrics: router and heads together, as a user sees it
            "unlearn_head/e2e_forget_nll_gated": _safe_mean(
                [m["nll_gated"] for m in forget_metrics]
            ),
            "unlearn_head/e2e_retain_nll_gated": _safe_mean(
                [m["nll_gated"] for m in retain_metrics]
            ),
        }
        self.log(metrics)
        return metrics

    def _save_heads(self):
        path = os.path.join(self.args.output_dir, "unlearn_head.pt")
        torch.save(
            {
                "forget_head": self.forget_head.state_dict(),
                "retain_head": self.retain_head.state_dict(),
                "tau": self.tau,
                "lam": self.lam,
                "epsilon": self.epsilon,
            },
            path,
        )

    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)
        if self.accelerator.is_local_main_process:
            if self.calibrate and self.retain_cal is not None:
                self._calibrate()
            if self.retain_cal is not None and self.forget_cal is not None:
                self._log_diagnostics()
            self._save_heads()
        return output

    def _install_correction_hook(self):
        """Shared hook for generate_with_correction/evaluate. Computes alpha(q)
        once at prefill (cache_position==0) from that call's own logits/hidden
        state, caches it for later cached steps. Uses real labels when present
        (teacher-forced metrics), else treats the whole input as the prompt
        (generation). Assumes greedy/sampling decoding, not beam search.
        Returns (handle, cache); cache fills in "scores"/"alpha" after prefill."""
        forget_head, retain_head = self.forget_head, self.retain_head
        cache = {"alpha": None, "scores": None}

        def hook(module, args, kwargs, output):
            cache_position = kwargs.get("cache_position")
            is_prefill = cache_position is None or cache_position[0].item() == 0

            h = output.hidden_states[-1]
            delta_f, delta_r = forget_head(h), retain_head(h)

            if is_prefill:
                input_ids = kwargs.get("input_ids")
                if input_ids is None:
                    raise RuntimeError(
                        "UnlearnHead correction hook needs input_ids at the "
                        "prefill step to compute the router score; none were "
                        "found in this forward call."
                    )
                attention_mask = kwargs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                labels = kwargs.get("labels")
                if labels is None:
                    labels = torch.full_like(input_ids, -100)

                next_tokens = input_ids[..., 1:]
                tok_f = self._token_logp(output.logits + delta_f, next_tokens)
                tok_r = self._token_logp(output.logits + delta_r, next_tokens)
                prompt_mask = (labels[..., 1:] == -100) & (attention_mask[..., 1:] == 1)
                scores = self._masked_ratio(tok_f, tok_r, prompt_mask)
                cache["scores"] = scores
                cache["alpha"] = torch.where(
                    scores > self.tau,
                    torch.full_like(scores, self.lam),
                    torch.zeros_like(scores),
                )
            elif cache["alpha"].shape[0] != output.logits.shape[0]:
                raise RuntimeError(
                    "Batch size changed between decoding steps (likely beam "
                    "search). UnlearnHead's correction hook only supports "
                    "greedy/sampling decoding."
                )

            alpha = cache["alpha"].view(-1, *([1] * (delta_f.dim() - 1)))
            output.logits = output.logits - alpha.to(delta_f.device) * (delta_f - delta_r)
            return output

        handle = self.model.register_forward_hook(hook, with_kwargs=True)
        return handle, cache

    def generate_with_correction(self, prompts, max_new_tokens=64, **generate_kwargs):
        """Corrected decoding for one prompt or a list of prompts, each with
        its own independently gated alpha(q) (Eq. 13, single-request case)."""
        if self.tau is None:
            raise RuntimeError("Router isn't calibrated yet — call train() first.")

        single = isinstance(prompts, str)
        prompt_list = [prompts] if single else list(prompts)

        tokenizer = self.processing_class
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # required for batched causal-LM generation
        try:
            enc = tokenizer(prompt_list, return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = original_padding_side

        device = next(self.model.parameters()).device
        enc = enc.to(device)

        self.model.eval()
        handle, cache = self._install_correction_hook()
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    output_hidden_states=True,
                    **generate_kwargs,
                )
        finally:
            handle.remove()

        texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results = [
            {"text": t, "score": s.item(), "alpha": a.item(), "tau": self.tau}
            for t, s, a in zip(texts, cache["scores"], cache["alpha"])
        ]
        return results[0] if single else results

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", trial=None
    ):
        """Wraps FinetuneTrainer.evaluate so every metric sees corrected z_U."""
        if self.tau is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, trial)

        original_output_hidden_states = self.model.config.output_hidden_states
        self.model.config.output_hidden_states = True
        handle, _ = self._install_correction_hook()
        try:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, trial)
        finally:
            handle.remove()
            self.model.config.output_hidden_states = original_output_hidden_states