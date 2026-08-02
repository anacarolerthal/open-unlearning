from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

from data.unlearn import ForgetRetainDataset
from evals.metrics.utils import evaluate_probability
from evals.metrics.utility import constrained_selection
from trainer.unlearn.unlearn_head import UnlearnHead


class _TinyQA(torch.utils.data.Dataset):
    def __init__(self, offset):
        self.offset = offset

    def __len__(self):
        return 4

    def __getitem__(self, index):
        input_ids = (
            torch.tensor([1, 2 + self.offset, 3 + index % 2, 4 + index % 3, 5]) % 31
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.tensor(
                [-100, -100, -100, int(input_ids[3]), int(input_ids[4])]
            ),
        }


def _collate(items):
    if "forget" in items[0]:
        return {key: _collate([item[key] for item in items]) for key in items[0]}
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _model():
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
    )


@pytest.mark.parametrize("objective", UnlearnHead.OBJECTIVES)
def test_objective_has_finite_head_gradients(tmp_path, objective):
    data = ForgetRetainDataset(_TinyQA(0), _TinyQA(5), _TinyQA(9))
    args = TrainingArguments(
        output_dir=tmp_path / objective,
        do_train=True,
        do_eval=False,
        report_to="none",
        per_device_train_batch_size=2,
        max_steps=1,
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    trainer = UnlearnHead(
        model=_model(),
        args=args,
        train_dataset=data,
        data_collator=_collate,
        objective=objective,
        rank=4,
    )
    batch = trainer._prepare_inputs(_collate([data[0], data[1]]))

    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()

    trainable = trainer._trainable_head_parameters()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in trainable
    )
    if objective == "direct_correction":
        assert all(
            parameter.grad is None for parameter in trainer.retain_head.parameters()
        )


def _selection(forget_quality, forget_truth_ratio, model_utility):
    pre_compute = {
        "forget_quality": {"agg_value": forget_quality},
        "forget_truth_ratio": {"agg_value": forget_truth_ratio},
        "model_utility": {"agg_value": model_utility},
    }
    return constrained_selection._metric_fn(
        None,
        pre_compute=pre_compute,
        utility_floor=0.587,
        truth_ratio_floor=0.475,
        forget_quality_threshold=0.05,
        log_gap_scale=20.0,
    )


def test_constrained_selection_prioritizes_preservation():
    safe_but_incomplete = _selection(4.2e-17, 0.494, 0.594)
    destructive = _selection(4.3e-9, 0.428, 0.426)
    feasible = _selection(0.1, 0.6, 0.59)

    assert -1 < safe_but_incomplete["agg_value"] < 0
    assert destructive["agg_value"] < -2
    assert feasible["agg_value"] > 1
    assert feasible["feasible"] == 1


def test_oracle_routing_uses_metric_context():
    trainer = object.__new__(UnlearnHead)
    trainer.router_mode = "oracle"
    trainer.lam = 1.5
    scores = torch.tensor([-10.0, 10.0])

    trainer._eval_metric_name = "forget_Q_A_Prob"
    assert torch.equal(trainer._routing_alpha(scores), torch.full_like(scores, 1.5))

    trainer._eval_metric_name = "retain_Q_A_Prob"
    assert torch.equal(trainer._routing_alpha(scores), torch.zeros_like(scores))


def test_evaluation_grid_reuses_each_trained_head():
    trainer = SimpleNamespace(
        lam=1.0,
        router_mode="calibrated",
        eval_lambdas=(0.0, 0.5),
        eval_router_modes=("calibrated", "oracle"),
        has_evaluation_grid=True,
        _lambda_label=UnlearnHead._lambda_label,
    )
    calls = []
    logged = []

    def fake_evaluate(**kwargs):
        prefix = kwargs["metric_key_prefix"]
        calls.append((trainer.lam, trainer.router_mode, prefix))
        score = trainer.lam + (trainer.router_mode == "oracle")
        return {
            f"{prefix}_constrained_selection": score,
            f"{prefix}_constrained_selection/feasible": float(score > 1),
            f"{prefix}_model_utility": 0.6,
            f"{prefix}_forget_truth_ratio": 0.5,
            f"{prefix}_forget_quality": 0.1,
            f"{prefix}_forget_Q_A_Prob": 0.4,
            f"{prefix}_forget_Q_A_ROUGE": 0.3,
        }

    trainer.evaluate = fake_evaluate
    trainer.log = logged.append

    metrics = UnlearnHead.evaluate_grid(trainer)

    assert len(calls) == 4
    assert trainer.lam == 1.0
    assert trainer.router_mode == "calibrated"
    assert metrics["eval_grid/num_points"] == 4
    assert metrics["eval_grid/feasible_points"] == 1
    assert metrics["eval_grid/best_constrained_selection"] == 1.5
    assert metrics["eval_grid/best_lambda"] == 0.5
    assert metrics["eval_grid/best_router_index"] == 1
    assert metrics["eval_grid/lambda_0_spread/model_utility"] == 0
    assert logged[-1]["eval_grid/oracle/best_constrained_selection"] == 1.5


def test_probability_evaluation_converts_bfloat16_outputs():
    class _BFloat16Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))

        @property
        def device(self):
            return self.anchor.device

        def forward(self, input_ids, **kwargs):
            batch_size, sequence_length = input_ids.shape
            logits = torch.zeros(
                batch_size,
                sequence_length,
                8,
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
            return SimpleNamespace(logits=logits)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3]]),
    }

    results = evaluate_probability(_BFloat16Model(), batch)

    assert len(results) == 1
    assert isinstance(results[0]["avg_loss"], float)
    assert isinstance(results[0]["prob"], float)
