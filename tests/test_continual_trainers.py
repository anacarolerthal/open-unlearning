from pathlib import Path

import torch
import pytest
from hydra import compose, initialize_config_dir
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments

from data.unlearn import ForgetRetainDataset
from continual_unlearn import _release_stage_trainer, _stage_trainer_config
from trainer.unlearn.npo import NPO
from trainer.unlearn.roadblock import RoadBlock


class _TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        input_ids = torch.tensor([1, 2 + index, 3, 4])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.tensor([-100, -100, 3, 4]),
            "index": index,
        }


def _collate(instances):
    if "input_ids" not in instances[0]:
        return {key: _collate([instance[key] for instance in instances]) for key in instances[0]}
    return {
        "input_ids": torch.stack([instance["input_ids"] for instance in instances]),
        "attention_mask": torch.stack([instance["attention_mask"] for instance in instances]),
        "labels": torch.stack([instance["labels"] for instance in instances]),
    }


def _model():
    config = GPT2Config(
        vocab_size=128,
        n_embd=16,
        n_inner=32,
        n_layer=1,
        n_head=2,
        n_positions=32,
    )
    return GPT2LMHeadModel(config)


def _args(path):
    return TrainingArguments(
        output_dir=str(path),
        do_train=True,
        do_eval=False,
        report_to=[],
        per_device_train_batch_size=1,
        max_steps=1,
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        remove_unused_columns=False,
        use_cpu=True,
    )


def _dataset():
    data = _TinyDataset()
    return ForgetRetainDataset(data, data)


def test_continual_stage_keeps_nested_batches_and_accepts_npo_warmup(tmp_path):
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="continual_unlearn.yaml",
            overrides=[
                "trainer=NPO",
                "+trainer.args.warmup_epochs=1.0",
                "+trainer.args.lr_scheduler_type=linear",
            ],
        )

    stage = _stage_trainer_config(config, tmp_path, "request_01")

    assert stage.args.remove_unused_columns is False
    assert stage.args.warmup_epochs == 1.0
    assert stage.args.lr_scheduler_type == "linear"


def test_npo_next_stage_uses_live_model_and_current_reference(tmp_path):
    live_model = _model()
    first = NPO(
        model=live_model,
        args=_args(tmp_path / "first"),
        train_dataset=_dataset(),
        data_collator=_collate,
        retain_loss_type="NLL",
    )
    assert first.model is live_model
    first.train()
    first.ref_model = None

    with torch.no_grad():
        next(live_model.parameters()).add_(1)
    current = next(live_model.parameters()).detach().clone()

    second = NPO(
        model=live_model,
        args=_args(tmp_path / "second"),
        train_dataset=_dataset(),
        data_collator=_collate,
        retain_loss_type="NLL",
    )

    assert second.model is live_model
    reference = next(second.ref_model.parameters()).detach()
    assert reference.data_ptr() != next(live_model.parameters()).data_ptr()
    assert torch.equal(reference, current)
    second.train()


def test_roadblock_accumulates_and_routes_named_adapters(tmp_path):
    first = RoadBlock(
        model=_model(),
        args=_args(tmp_path / "first"),
        train_dataset=_dataset(),
        data_collator=_collate,
        request_name="request_01",
        classifier="oracle",
    )
    live_model = first.model
    first.train()
    second = RoadBlock(
        model=live_model,
        args=_args(tmp_path / "second"),
        train_dataset=_dataset(),
        data_collator=_collate,
        request_name="request_02",
        classifier="oracle",
    )
    second.train()

    assert second.model is live_model
    assert set(live_model.peft_config) == {"request_01", "request_02"}
    output_layer = live_model.get_base_model().get_output_embeddings()
    assert not output_layer.road_theta["request_01"].requires_grad
    assert output_layer.road_theta["request_02"].requires_grad

    with second.request_evaluation_context("request_01"):
        assert second._active_request == "request_01"
        with second.classifier_context(True):
            assert second._active_request == "request_01"
        with second.classifier_context(False):
            assert second._active_request is None

    live_model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad(), live_model.disable_adapter():
        base_logits = live_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
    with torch.no_grad(), second.request_evaluation_context(None):
        utility_logits = live_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
    assert torch.equal(utility_logits, base_logits)

    with torch.no_grad():
        output_layer.road_theta["request_01"].fill_(0.5)
    with torch.no_grad(), second.request_evaluation_context("request_01"):
        corrected_logits = live_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
    assert not torch.equal(corrected_logits, base_logits)

    _release_stage_trainer(second)
    assert not hasattr(live_model, "unlearn_classifier_context")
    assert not hasattr(live_model, "roadblock_classifier_threshold")
    assert not hasattr(live_model, "roadblock_router_diagnostics")


@pytest.mark.parametrize("classifier", ["guard_multiclass", "guard_prototype"])
def test_learned_continual_router_replays_activations(tmp_path, classifier):
    first = RoadBlock(
        model=_model(),
        args=_args(tmp_path / f"{classifier}_first"),
        train_dataset=_dataset(),
        data_collator=_collate,
        request_name="request_01",
        classifier=classifier,
        continual=True,
        num_centroids=1,
    )
    live_model = first.model
    first.train()

    cache = live_model._roadblock_activation_cache
    assert cache["retain"].device.type == "cpu"
    assert set(cache["forget"]) == {"request_01"}
    assert not any("roadblock_activation_cache" in key for key in live_model.state_dict())

    second = RoadBlock(
        model=live_model,
        args=_args(tmp_path / f"{classifier}_second"),
        train_dataset=_dataset(),
        data_collator=_collate,
        request_name="request_02",
        classifier=classifier,
        continual=True,
        num_centroids=1,
    )
    second.train()
    assert second.model is live_model
    assert set(cache["forget"]) == {"request_01", "request_02"}

    _release_stage_trainer(second)
    assert hasattr(live_model, "_roadblock_activation_cache")
