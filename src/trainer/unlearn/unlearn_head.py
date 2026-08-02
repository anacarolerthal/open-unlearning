import math
import os
import statistics

import torch
import torch.nn.functional as F
from torch import nn
from transformers import TrainerCallback
from transformers.modeling_outputs import CausalLMOutput

from trainer.unlearn.base import UnlearnTrainer


def _safe_mean(values):
    values = [v for v in values if not math.isnan(v)]
    return statistics.fmean(values) if values else float("nan")


def _fraction_above(values, threshold):
    values = [v for v in values if not math.isnan(v)]
    return (
        sum(1 for v in values if v > threshold) / len(values)
        if values
        else float("nan")
    )


def _pairwise_auc(positive_scores, negative_scores):
    """Empirical AUROC without adding a metrics-library dependency."""
    positive_scores = [v for v in positive_scores if not math.isnan(v)]
    negative_scores = [v for v in negative_scores if not math.isnan(v)]
    if not positive_scores or not negative_scores:
        return float("nan")
    wins = sum(
        positive > negative
        for positive in positive_scores
        for negative in negative_scores
    )
    ties = sum(
        positive == negative
        for positive in positive_scores
        for negative in negative_scores
    )
    return (wins + 0.5 * ties) / (len(positive_scores) * len(negative_scores))


class LowRankHead(nn.Module):
    """B @ (A @ h); B is zero-initialized so the head starts as a no-op."""

    def __init__(self, hidden_size, vocab_size, rank):
        super().__init__()
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, vocab_size, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, h):
        return self.B(self.A(h.to(dtype=self.A.weight.dtype)))


class _HeadOptimizerCallback(TrainerCallback):
    """Clip and clear the heads, which live outside the frozen base model."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if args.max_grad_norm and args.max_grad_norm > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.trainer._trainable_head_parameters(), args.max_grad_norm
            )
            self.trainer._last_head_grad_norm = grad_norm.detach().item()

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer.forget_head.zero_grad(set_to_none=True)
        self.trainer.retain_head.zero_grad(set_to_none=True)


class UnlearnHead(UnlearnTrainer):
    """Frozen-backbone unlearning through a normalized low-rank logit edit.

    ``independent_ce`` is the original two-specialist objective.
    ``answer_contrast`` adds a signed answer log-likelihood-ratio margin.
    ``direct_correction`` trains one correction head with an NPO-style forget
    loss and retain NLL, leaving the retain head at zero.

    Routing is either calibrated, always-on for head-only diagnosis, or oracle
    during evaluation (on for forget metrics and off for utility metrics).
    """

    OBJECTIVES = ("independent_ce", "answer_contrast", "direct_correction")
    ROUTER_MODES = ("calibrated", "always_on", "oracle")
    ORACLE_FORGET_METRICS = ("extraction_strength", "exact_memorization")

    def __init__(
        self,
        rank=32,
        lam=1.0,
        epsilon=0.05,
        router_mode="calibrated",
        objective="independent_ce",
        answer_contrast_weight=1.0,
        answer_contrast_margin=0.1,
        direct_beta=0.1,
        scale_regularization_weight=1e-3,
        normalize_correction=True,
        head_checkpoint=None,
        prompt_loss_weight=0.1,
        diagnostics_max_samples=128,
        save_base_model=False,
        *args,
        **kwargs,
    ):
        checkpoint_state = None
        if head_checkpoint:
            checkpoint_state = self._read_head_checkpoint(head_checkpoint)
            rank = checkpoint_state.get("rank", rank)
            objective = checkpoint_state.get("objective", objective)

        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank!r}")
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if router_mode not in self.ROUTER_MODES:
            raise ValueError(
                f"router_mode must be one of {sorted(self.ROUTER_MODES)}, "
                f"got {router_mode!r}"
            )
        if objective not in self.OBJECTIVES:
            raise ValueError(
                f"objective must be one of {sorted(self.OBJECTIVES)}, got {objective!r}"
            )
        if prompt_loss_weight < 0:
            raise ValueError("prompt_loss_weight must be non-negative")
        if answer_contrast_weight < 0 or answer_contrast_margin < 0:
            raise ValueError("answer contrast weight and margin must be non-negative")
        if direct_beta <= 0:
            raise ValueError("direct_beta must be positive")
        if scale_regularization_weight < 0:
            raise ValueError("scale_regularization_weight must be non-negative")
        if not isinstance(diagnostics_max_samples, int) or diagnostics_max_samples <= 0:
            raise ValueError("diagnostics_max_samples must be a positive integer")

        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.rank = rank
        self.lam = lam
        self.epsilon = epsilon
        self.router_mode = router_mode
        self.objective = objective
        self.answer_contrast_weight = answer_contrast_weight
        self.answer_contrast_margin = answer_contrast_margin
        self.direct_beta = direct_beta
        self.scale_regularization_weight = scale_regularization_weight
        self.normalize_correction = normalize_correction
        self.head_checkpoint = head_checkpoint
        self.prompt_loss_weight = prompt_loss_weight
        self.diagnostics_max_samples = diagnostics_max_samples
        self.save_base_model = save_base_model
        self.tau = None
        self.correction_scale = None if normalize_correction else 1.0
        self._eval_metric_name = None
        self._last_head_grad_norm = None
        self._loss_totals = {}
        self._loss_counts = {}

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

        base_parameter = next(self.model.parameters())
        self.forget_head = LowRankHead(
            self.model.config.hidden_size, self.model.config.vocab_size, rank
        ).to(device=base_parameter.device)
        self.retain_head = LowRankHead(
            self.model.config.hidden_size, self.model.config.vocab_size, rank
        ).to(device=base_parameter.device)
        if objective == "direct_correction":
            self.retain_head.requires_grad_(False)

        if checkpoint_state is not None:
            self.forget_head.load_state_dict(checkpoint_state["forget_head"])
            self.retain_head.load_state_dict(checkpoint_state["retain_head"])
            self.tau = checkpoint_state.get("tau")
            self.correction_scale = checkpoint_state.get("correction_scale")
            if not normalize_correction:
                self.correction_scale = 1.0

        self.add_callback(_HeadOptimizerCallback(self))

        self.retain_cal = getattr(self.train_dataset, "retain", None)
        self.forget_cal = getattr(self.train_dataset, "forget", None)
        self.router_cal = getattr(self.train_dataset, "calibration", None)
        if self.router_cal is None:
            raise ValueError(
                "UnlearnHead requires data.calibration for routing and correction "
                "scale estimation."
            )
        if not self.args.do_train and checkpoint_state is None:
            raise ValueError("Evaluation-only UnlearnHead requires head_checkpoint")

    @staticmethod
    def _read_head_checkpoint(path):
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            path = os.path.join(path, "unlearn_head.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"UnlearnHead checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not {"forget_head", "retain_head", "rank"}.issubset(state):
            raise ValueError(f"Invalid UnlearnHead checkpoint: {path}")
        return state

    def _trainable_head_parameters(self):
        return [
            parameter
            for head in (self.forget_head, self.retain_head)
            for parameter in head.parameters()
            if parameter.requires_grad
        ]

    def _record_loss(self, name, value):
        self._loss_totals[name] = (
            self._loss_totals.get(name, 0.0) + value.detach().item()
        )
        self._loss_counts[name] = self._loss_counts.get(name, 0) + 1

    def log(self, logs, start_time=None):
        if self._last_head_grad_norm is not None:
            logs["unlearn_head/grad_norm"] = self._last_head_grad_norm
            self._last_head_grad_norm = None
        for name, total in self._loss_totals.items():
            logs[f"unlearn_head/{name}"] = total / self._loss_counts[name]
        self._loss_totals.clear()
        self._loss_counts.clear()
        super().log(logs, start_time)

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                self._trainable_head_parameters(), **optimizer_kwargs
            )
        return self.optimizer

    @staticmethod
    def _base_logits_and_hidden(model, input_ids, attention_mask):
        """Run the frozen transformer and ordinary vocabulary projection once."""
        model.eval()
        base_model = getattr(model, model.base_model_prefix)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("UnlearnHead requires a model with output embeddings")
        with torch.no_grad():
            base_output = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = base_output.last_hidden_state
            base_logits = output_embeddings(hidden)
        return base_logits, hidden

    @staticmethod
    def _ce_loss(logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss = F.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100, reduction="sum"
        )
        return loss / (flat_labels != -100).sum().clamp(min=1)

    @staticmethod
    def _sequence_nll(logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.transpose(-1, -2),
            shift_labels,
            ignore_index=-100,
            reduction="none",
        )
        mask = shift_labels != -100
        return (losses * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    @staticmethod
    def _prompt_labels(input_ids, labels, attention_mask):
        prompt_mask = (labels == -100) & attention_mask.bool()
        return input_ids.masked_fill(~prompt_mask, -100)

    @staticmethod
    def _masked_ratio(tok_forget_logp, tok_retain_logp, mask):
        difference = tok_forget_logp - tok_retain_logp
        return (difference * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    @staticmethod
    def _token_logp(logits, next_tokens):
        logp = F.log_softmax(logits, dim=-1)
        return logp[..., :-1, :].gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

    def _specialist_logits(self, model, batch):
        z0, hidden = self._base_logits_and_hidden(
            model, batch["input_ids"], batch["attention_mask"]
        )
        delta_forget = self.forget_head(hidden)
        delta_retain = self.retain_head(hidden)
        return z0, z0 + delta_forget, z0 + delta_retain, delta_forget - delta_retain

    def _answer_ratio(self, z_forget, z_retain, batch):
        next_tokens = batch["input_ids"][..., 1:]
        answer_mask = (batch["labels"][..., 1:] != -100) & (
            batch["attention_mask"][..., 1:] == 1
        )
        return self._masked_ratio(
            self._token_logp(z_forget, next_tokens),
            self._token_logp(z_retain, next_tokens),
            answer_mask,
        )

    def _two_head_loss(self, model, inputs):
        forget = inputs["forget"]
        retain = inputs["retain"]
        _, z_forget_f, z_retain_f, contrast_f = self._specialist_logits(model, forget)
        _, z_forget_r, z_retain_r, contrast_r = self._specialist_logits(model, retain)

        forget_answer = self._ce_loss(z_forget_f, forget["labels"])
        retain_answer = self._ce_loss(z_retain_r, retain["labels"])
        answer_loss = forget_answer + retain_answer

        forget_prompt = self._ce_loss(
            z_forget_f,
            self._prompt_labels(
                forget["input_ids"], forget["labels"], forget["attention_mask"]
            ),
        )
        retain_prompt = self._ce_loss(
            z_retain_r,
            self._prompt_labels(
                retain["input_ids"], retain["labels"], retain["attention_mask"]
            ),
        )
        prompt_loss = forget_prompt + retain_prompt
        loss = (answer_loss + self.prompt_loss_weight * prompt_loss) / (
            1.0 + self.prompt_loss_weight
        )

        if self.objective == "answer_contrast":
            forget_ratio = self._answer_ratio(z_forget_f, z_retain_f, forget)
            retain_ratio = self._answer_ratio(z_forget_r, z_retain_r, retain)
            contrast_loss = (
                F.softplus(self.answer_contrast_margin - forget_ratio).mean()
                + F.softplus(self.answer_contrast_margin + retain_ratio).mean()
            )
            loss = loss + self.answer_contrast_weight * contrast_loss
            self._record_loss("answer_contrast_loss", contrast_loss)

        scale_penalty = (
            contrast_f.float().square().mean() + contrast_r.float().square().mean()
        )
        loss = loss + self.scale_regularization_weight * scale_penalty
        self._record_loss("answer_loss", answer_loss)
        self._record_loss("scale_penalty", scale_penalty)
        return loss, CausalLMOutput(logits=z_forget_f)

    def _direct_correction_loss(self, model, inputs):
        forget = inputs["forget"]
        retain = inputs["retain"]

        z0_forget, hidden_forget = self._base_logits_and_hidden(
            model, forget["input_ids"], forget["attention_mask"]
        )
        delta_forget = self.forget_head(hidden_forget)
        z_unlearn_forget = z0_forget - delta_forget
        z_specialist_forget = z0_forget + delta_forget

        z0_retain, hidden_retain = self._base_logits_and_hidden(
            model, retain["input_ids"], retain["attention_mask"]
        )
        delta_retain = self.forget_head(hidden_retain)
        z_unlearn_retain = z0_retain - delta_retain

        forget_nll = self._sequence_nll(z_unlearn_forget, forget["labels"])
        base_forget_nll = self._sequence_nll(z0_forget, forget["labels"])
        forget_loss = (
            -2.0
            / self.direct_beta
            * F.logsigmoid(self.direct_beta * (forget_nll - base_forget_nll)).mean()
        )
        retain_loss = self._ce_loss(z_unlearn_retain, retain["labels"])
        prompt_loss = self._ce_loss(
            z_specialist_forget,
            self._prompt_labels(
                forget["input_ids"], forget["labels"], forget["attention_mask"]
            ),
        )
        scale_penalty = (
            delta_forget.float().square().mean() + delta_retain.float().square().mean()
        )
        loss = (
            forget_loss
            + retain_loss
            + self.prompt_loss_weight * prompt_loss
            + self.scale_regularization_weight * scale_penalty
        )
        self._record_loss("direct_forget_loss", forget_loss)
        self._record_loss("direct_retain_loss", retain_loss)
        self._record_loss("scale_penalty", scale_penalty)
        return loss, CausalLMOutput(logits=z_unlearn_forget)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.objective == "direct_correction":
            loss, outputs = self._direct_correction_loss(model, inputs)
        else:
            loss, outputs = self._two_head_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss

    def _frozen_logits(self, input_ids, attention_mask):
        with torch.no_grad():
            z0, hidden = self._base_logits_and_hidden(
                self.model, input_ids, attention_mask
            )
            delta_forget = self.forget_head(hidden)
            delta_retain = self.retain_head(hidden)
            z_forget = z0 + delta_forget
            z_retain = z0 + delta_retain
        return z0, z_forget, z_retain, delta_forget - delta_retain

    def _prompt_score(self, input_ids, attention_mask, labels):
        _, z_forget, z_retain, _ = self._frozen_logits(input_ids, attention_mask)
        next_tokens = input_ids[..., 1:]
        prompt_mask = (labels[..., 1:] == -100) & (attention_mask[..., 1:] == 1)
        return self._masked_ratio(
            self._token_logp(z_forget, next_tokens),
            self._token_logp(z_retain, next_tokens),
            prompt_mask,
        )

    def _oracle_active(self):
        if self._eval_metric_name is None:
            raise RuntimeError("Oracle routing requires an evaluation metric context")
        return self._eval_metric_name.startswith("forget_") or (
            self._eval_metric_name in self.ORACLE_FORGET_METRICS
        )

    def _routing_alpha(self, scores, oracle_active=None):
        if self.router_mode == "always_on":
            return torch.full_like(scores, self.lam)
        if self.router_mode == "oracle":
            active = self._oracle_active() if oracle_active is None else oracle_active
            return (
                torch.full_like(scores, self.lam)
                if active
                else torch.zeros_like(scores)
            )
        if self.tau is None:
            raise RuntimeError("Router is not calibrated")
        return torch.where(
            scores > self.tau,
            torch.full_like(scores, self.lam),
            torch.zeros_like(scores),
        )

    def _correction_weight(self, alpha):
        if self.correction_scale is None:
            raise RuntimeError("Correction scale has not been estimated")
        return alpha / self.correction_scale

    def _calibrate(self):
        device = next(self.model.parameters()).device
        scores = []
        with torch.no_grad():
            for index in range(len(self.router_cal)):
                item = self.router_cal[index]
                scores.append(
                    self._prompt_score(
                        item["input_ids"].to(device).unsqueeze(0),
                        item["attention_mask"].to(device).unsqueeze(0),
                        item["labels"].to(device).unsqueeze(0),
                    ).item()
                )
        if not scores:
            raise ValueError("UnlearnHead calibration dataset is empty")
        scores_tensor = torch.tensor(scores)
        self.tau = torch.quantile(scores_tensor, 1 - self.epsilon).item()
        if self.accelerator.is_local_main_process:
            self.log(
                {
                    "unlearn_head/tau": self.tau,
                    "unlearn_head/calibration_size": len(scores),
                    "unlearn_head/calibration_activation_rate": _fraction_above(
                        scores, self.tau
                    ),
                }
            )

    def _estimate_correction_scale(self):
        device = next(self.model.parameters()).device
        token_rms_values = []
        with torch.no_grad():
            for index in range(len(self.router_cal)):
                item = self.router_cal[index]
                input_ids = item["input_ids"].to(device).unsqueeze(0)
                attention_mask = item["attention_mask"].to(device).unsqueeze(0)
                labels = item["labels"].to(device).unsqueeze(0)
                _, _, _, delta = self._frozen_logits(input_ids, attention_mask)
                token_rms = torch.linalg.vector_norm(delta, dim=-1) / math.sqrt(
                    delta.shape[-1]
                )
                prompt_mask = (labels == -100) & attention_mask.bool()
                token_rms_values.extend(token_rms[prompt_mask].float().cpu().tolist())
        if not token_rms_values:
            raise ValueError("No prompt tokens available for correction normalization")
        self.correction_scale = max(
            torch.tensor(token_rms_values).median().item(), 1e-6
        )
        if self.accelerator.is_local_main_process:
            self.log({"unlearn_head/correction_scale": self.correction_scale})

    def _prepare_inference_state(self):
        if self.tau is None:
            self._calibrate()
        if self.normalize_correction and self.correction_scale is None:
            self._estimate_correction_scale()
        elif not self.normalize_correction:
            self.correction_scale = 1.0

    def _example_metrics(self, item, head_name):
        device = next(self.model.parameters()).device
        input_ids = item["input_ids"].to(device).unsqueeze(0)
        attention_mask = item["attention_mask"].to(device).unsqueeze(0)
        labels = item["labels"].to(device).unsqueeze(0)
        z0, z_forget, z_retain, contrast = self._frozen_logits(
            input_ids, attention_mask
        )
        next_tokens = input_ids[..., 1:]
        tok0 = self._token_logp(z0, next_tokens)
        tok_forget = self._token_logp(z_forget, next_tokens)
        tok_retain = self._token_logp(z_retain, next_tokens)
        valid = attention_mask[..., 1:] == 1
        prompt_mask = (labels[..., 1:] == -100) & valid
        answer_mask = (labels[..., 1:] != -100) & valid

        def masked_mean(values, mask):
            selected = values[mask]
            return selected.mean().item() if selected.numel() else float("nan")

        scores = self._masked_ratio(tok_forget, tok_retain, prompt_mask)
        tok_head = tok_forget if head_name == "forget" else tok_retain
        alpha = self._routing_alpha(scores, oracle_active=head_name == "forget")
        z_unlearn = z0 - self._correction_weight(alpha).item() * contrast
        return {
            "score": scores.item(),
            "nll_z0": -masked_mean(tok0, answer_mask),
            "nll_head": -masked_mean(tok_head, answer_mask),
            "nll_gated": -masked_mean(
                self._token_logp(z_unlearn, next_tokens), answer_mask
            ),
        }

    def _diagnostic_indices(self, dataset):
        if len(dataset) <= self.diagnostics_max_samples:
            return list(range(len(dataset)))
        generator = torch.Generator().manual_seed(self.args.seed)
        return torch.randperm(len(dataset), generator=generator)[
            : self.diagnostics_max_samples
        ].tolist()

    def _log_diagnostics(self):
        forget_metrics = [
            self._example_metrics(self.forget_cal[index], "forget")
            for index in self._diagnostic_indices(self.forget_cal)
        ]
        retain_metrics = [
            self._example_metrics(self.retain_cal[index], "retain")
            for index in self._diagnostic_indices(self.retain_cal)
        ]
        forget_scores = [metric["score"] for metric in forget_metrics]
        retain_scores = [metric["score"] for metric in retain_metrics]
        metrics = {
            "unlearn_head/router_score_mean_forget": _safe_mean(forget_scores),
            "unlearn_head/router_score_mean_retain": _safe_mean(retain_scores),
            "unlearn_head/router_auc": _pairwise_auc(forget_scores, retain_scores),
            "unlearn_head/router_tpr_at_tau": _fraction_above(forget_scores, self.tau),
            "unlearn_head/router_fpr_at_tau": _fraction_above(retain_scores, self.tau),
            "unlearn_head/head_forget_nll_gap": _safe_mean(
                [metric["nll_head"] - metric["nll_z0"] for metric in forget_metrics]
            ),
            "unlearn_head/head_retain_nll_gap": _safe_mean(
                [metric["nll_head"] - metric["nll_z0"] for metric in retain_metrics]
            ),
            "unlearn_head/e2e_forget_nll_gated": _safe_mean(
                [metric["nll_gated"] for metric in forget_metrics]
            ),
            "unlearn_head/e2e_retain_nll_gated": _safe_mean(
                [metric["nll_gated"] for metric in retain_metrics]
            ),
            "unlearn_head/correction_scale": self.correction_scale,
        }
        self.log(metrics)
        return metrics

    def _save_heads(self, output_dir=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "forget_head": self.forget_head.state_dict(),
                "retain_head": self.retain_head.state_dict(),
                "tau": self.tau,
                "correction_scale": self.correction_scale,
                "rank": self.rank,
                "objective": self.objective,
                "epsilon": self.epsilon,
                "prompt_loss_weight": self.prompt_loss_weight,
                "answer_contrast_weight": self.answer_contrast_weight,
                "answer_contrast_margin": self.answer_contrast_margin,
                "direct_beta": self.direct_beta,
                "scale_regularization_weight": self.scale_regularization_weight,
                "normalize_correction": self.normalize_correction,
                "calibration_size": len(self.router_cal),
                "backbone": self.model.config._name_or_path,
                "model_type": self.model.config.model_type,
                "format_version": 2,
            },
            os.path.join(output_dir, "unlearn_head.pt"),
        )

    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)
        self._prepare_inference_state()
        if self.accelerator.is_local_main_process:
            if self.retain_cal is not None and self.forget_cal is not None:
                self._log_diagnostics()
            self._save_heads()
        return output

    def save_model(self, output_dir=None, _internal_call=False):
        if self.save_base_model:
            return super().save_model(output_dir, _internal_call)
        if self.accelerator.is_local_main_process:
            self._save_heads(output_dir)

    def _install_correction_hook(self):
        forget_head = self.forget_head
        retain_head = self.retain_head
        cache = {"alpha": None, "scores": None, "weight": None}

        def hook(module, args, kwargs, output):
            cache_position = kwargs.get("cache_position")
            is_prefill = cache_position is None or cache_position[0].item() == 0
            hidden = output.hidden_states[-1]
            delta_forget = forget_head(hidden)
            delta_retain = retain_head(hidden)

            if is_prefill:
                input_ids = kwargs.get("input_ids")
                if input_ids is None:
                    raise RuntimeError("UnlearnHead correction hook requires input_ids")
                attention_mask = kwargs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                labels = kwargs.get("labels")
                if labels is None:
                    labels = torch.full_like(input_ids, -100)
                next_tokens = input_ids[..., 1:]
                prompt_mask = (labels[..., 1:] == -100) & (attention_mask[..., 1:] == 1)
                scores = self._masked_ratio(
                    self._token_logp(output.logits + delta_forget, next_tokens),
                    self._token_logp(output.logits + delta_retain, next_tokens),
                    prompt_mask,
                )
                cache["scores"] = scores
                cache["alpha"] = self._routing_alpha(scores)
                cache["weight"] = self._correction_weight(cache["alpha"])
            elif cache["weight"].shape[0] != output.logits.shape[0]:
                raise RuntimeError(
                    "Batch size changed during decoding; beam search is unsupported"
                )

            weight = cache["weight"].view(-1, *([1] * (delta_forget.dim() - 1)))
            output.logits = output.logits - weight.to(delta_forget.device) * (
                delta_forget - delta_retain
            )
            return output

        return self.model.register_forward_hook(hook, with_kwargs=True), cache

    def generate_with_correction(self, prompts, max_new_tokens=64, **generate_kwargs):
        if self.router_mode == "oracle":
            raise ValueError("Oracle routing is evaluation-only")
        self._prepare_inference_state()
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
        encoding = encoding.to(next(self.model.parameters()).device)

        original_output_hidden_states = self.model.config.output_hidden_states
        self.model.config.output_hidden_states = True
        handle, cache = self._install_correction_hook()
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoding,
                    max_new_tokens=max_new_tokens,
                    output_hidden_states=True,
                    **generate_kwargs,
                )
        finally:
            handle.remove()
            self.model.config.output_hidden_states = original_output_hidden_states

        texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results = [
            {
                "text": text,
                "score": score.item(),
                "alpha": alpha.item(),
                "correction_weight": weight.item(),
                "correction_scale": self.correction_scale,
                "tau": self.tau,
                "router_mode": self.router_mode,
            }
            for text, score, alpha, weight in zip(
                texts, cache["scores"], cache["alpha"], cache["weight"]
            )
        ]
        return results[0] if single else results

    def _set_eval_context(self, metric_name):
        self._eval_metric_name = metric_name

    def _router_diagnostics(self, input_ids, attention_mask, labels):
        scores = self._prompt_score(input_ids, attention_mask, labels)
        activations = scores > self.tau
        return scores, activations

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", trial=None
    ):
        self._prepare_inference_state()
        original_output_hidden_states = self.model.config.output_hidden_states
        self.model.config.output_hidden_states = True
        self.model._unlearn_head_set_eval_context = self._set_eval_context
        self.model._unlearn_head_router_diagnostics = self._router_diagnostics
        handle, _ = self._install_correction_hook()
        try:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, trial)
        finally:
            handle.remove()
            del self.model._unlearn_head_set_eval_context
            del self.model._unlearn_head_router_diagnostics
            self._eval_metric_name = None
            self.model.config.output_hidden_states = original_output_hidden_states
