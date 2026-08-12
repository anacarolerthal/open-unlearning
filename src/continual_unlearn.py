import gc
import json
import os
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict

from data import get_collators, get_data
from evals import get_evaluator
from evals.continual import (
    REQUEST_METRICS,
    aggregate_request_logs,
    request_partitions,
    summarize_logs,
)
from model import get_model
from trainer import load_trainer
from trainer.unlearn.roadblock import RoadBlock
from trainer.utils import seed_everything


UTILITY_METRICS = ("model_utility",)


def _resolved_copy(config):
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def _patch_forget_dataset_split(node, forget_names, split):
    if isinstance(node, dict):
        hf_args = node.get("hf_args")
        if isinstance(hf_args, dict) and hf_args.get("name") in forget_names:
            hf_args["split"] = split
        for value in node.values():
            _patch_forget_dataset_split(value, forget_names, split)
    elif isinstance(node, list):
        for value in node:
            _patch_forget_dataset_split(value, forget_names, split)


def _evaluation_config(base_config, metric_names, forget_names=None, split=None):
    config = OmegaConf.to_container(base_config, resolve=True)
    config["metrics"] = {
        name: value for name, value in config["metrics"].items() if name in metric_names
    }
    config["overwrite"] = True
    if split is not None:
        _patch_forget_dataset_split(config, set(forget_names), split)
    return OmegaConf.create(config)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)


def _read_json(path):
    with Path(path).open() as file:
        return json.load(file)


def _validate_runtime(cfg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError("Continual unlearning supports exactly one process")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Continual unlearning requires exactly one visible CUDA device; "
            "set CUDA_VISIBLE_DEVICES to a single GPU"
        )
    if cfg.trainer.handler not in {"NPO", "RoadBlock"}:
        raise ValueError("trainer must be NPO or RoadBlock")

    reference_path = Path(to_absolute_path(cfg.retain_logs_path))
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing retain95 reference log: {reference_path}. "
            "Run `python setup_data.py --eval_logs`."
        )
    return reference_path


def _stage_data_config(cfg, split):
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    data_cfg["forget"]["TOFU_QA_forget"]["args"]["hf_args"]["split"] = split
    return OmegaConf.create(data_cfg)


def _stage_trainer_config(cfg, stage_dir, request_name):
    trainer_cfg = _resolved_copy(cfg.trainer)
    with open_dict(trainer_cfg):
        trainer_cfg.args.output_dir = str(stage_dir / "train")
        trainer_cfg.args.logging_dir = str(stage_dir / "train" / "logs")
        trainer_cfg.args.do_eval = False
        trainer_cfg.args.eval_strategy = "no"
        trainer_cfg.args.eval_on_start = False
        trainer_cfg.args.save_strategy = "no"
        if trainer_cfg.handler == "RoadBlock":
            trainer_cfg.method_args.request_name = request_name
    return trainer_cfg


def _run_evaluator(eval_cfg, model, tokenizer, template_args, output_dir):
    evaluator = get_evaluator("tofu", eval_cfg)
    return evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        template_args=template_args,
        output_dir=str(output_dir),
        overwrite=True,
    )


def _evaluation_context(trainer, request_name):
    if isinstance(trainer, RoadBlock):
        return trainer.request_evaluation_context(request_name)
    return nullcontext()


def _release_stage_trainer(trainer):
    if hasattr(trainer, "ref_model"):
        trainer.ref_model = None
    if isinstance(trainer, RoadBlock):
        for name in (
            "unlearn_classifier_context",
            "roadblock_diagnostic_batch",
            "roadblock_classify_batch",
            "roadblock_classifier_threshold",
        ):
            delattr(trainer.model, name)
    trainer.optimizer = None
    trainer.lr_scheduler = None
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(
    version_base=None, config_path="../configs", config_name="continual_unlearn.yaml"
)
def main(cfg: DictConfig):
    reference_path = _validate_runtime(cfg)
    with open_dict(cfg.eval.tofu):
        cfg.eval.tofu.retain_logs_path = str(reference_path)
    seed_everything(cfg.seed)

    partitions = request_partitions(**cfg.continual)
    output_root = Path(to_absolute_path(cfg.paths.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    reference_logs = _read_json(reference_path)
    target_model = cfg.model.model_args.pretrained_model_name_or_path

    model, tokenizer = get_model(cfg.model)
    template_args = cfg.model.template_args
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    forget_names = {cfg.forget_split, f"{cfg.forget_split}_perturbed"}

    continual_summary = {
        "method": cfg.trainer.handler,
        "model": target_model,
        "forget_split": cfg.forget_split,
        "retain_split": cfg.retain_split,
        "holdout_split": cfg.holdout_split,
        "requests": partitions,
        "stages": {},
    }

    trainer = None
    for stage_index, current_request in enumerate(partitions, start=1):
        stage_name = f"stage_{stage_index:02d}"
        stage_dir = output_root / stage_name
        split = f"train[{current_request['start']}:{current_request['stop']}]"
        data = get_data(
            _stage_data_config(cfg, split),
            mode="unlearn",
            tokenizer=tokenizer,
            template_args=template_args,
        )
        if len(data["train"].forget) != cfg.continual.examples_per_request:
            raise RuntimeError(f"{current_request['name']} did not load exactly 40 examples")

        trainer_cfg = _stage_trainer_config(
            cfg, stage_dir, current_request["name"]
        )
        trainer, _ = load_trainer(
            trainer_cfg=trainer_cfg,
            model=model,
            train_dataset=data["train"],
            processing_class=tokenizer,
            data_collator=collator,
            evaluators=None,
            template_args=template_args,
        )
        trainer.train()
        model = trainer.model
        if hasattr(trainer, "ref_model"):
            trainer.ref_model = None
        trainer.optimizer = None
        trainer.lr_scheduler = None
        gc.collect()
        torch.cuda.empty_cache()

        stage_request_logs = []
        stage_summary = {"requests": {}}
        for request in partitions[:stage_index]:
            request_split = f"train[{request['start']}:{request['stop']}]"
            request_eval_cfg = _evaluation_config(
                cfg.eval.tofu,
                REQUEST_METRICS,
                forget_names=forget_names,
                split=request_split,
            )
            request_dir = stage_dir / "requests" / request["name"]
            with _evaluation_context(trainer, request["name"]):
                _run_evaluator(
                    request_eval_cfg,
                    model,
                    tokenizer,
                    template_args,
                    request_dir,
                )
            logs = _read_json(request_dir / "TOFU_EVAL.json")
            evaluated = len(logs["forget_Q_A_Prob"]["value_by_index"])
            if evaluated != cfg.continual.examples_per_request:
                raise RuntimeError(
                    f"{request['name']} evaluation loaded {evaluated} examples, "
                    f"expected {cfg.continual.examples_per_request}"
                )
            stage_request_logs.append((request["start"], logs))
            stage_summary["requests"][request["name"]] = summarize_logs(logs)

        utility_cfg = _evaluation_config(cfg.eval.tofu, UTILITY_METRICS)
        utility_dir = stage_dir / "utility"
        with _evaluation_context(trainer, None):
            _run_evaluator(
                utility_cfg, model, tokenizer, template_args, utility_dir
            )
        utility_logs = _read_json(utility_dir / "TOFU_EVAL.json")
        stage_summary["utility"] = summarize_logs(utility_logs, UTILITY_METRICS)

        cumulative_logs = aggregate_request_logs(stage_request_logs, reference_logs)
        cumulative_dir = stage_dir / "cumulative"
        _write_json(cumulative_dir / "TOFU_EVAL.json", cumulative_logs)
        cumulative_summary = summarize_logs(cumulative_logs)
        _write_json(cumulative_dir / "TOFU_SUMMARY.json", cumulative_summary)
        stage_summary["cumulative"] = cumulative_summary

        trainer.log(
            {
                "eval_continual_stage": stage_index,
                **{
                    f"eval_continual_{name}": value
                    for name, value in cumulative_summary.items()
                },
                **{
                    f"eval_continual_{name}": value
                    for name, value in stage_summary["utility"].items()
                },
            }
        )

        continual_summary["stages"][stage_name] = stage_summary
        _write_json(output_root / "CONTINUAL_SUMMARY.json", continual_summary)
        _release_stage_trainer(trainer)
        trainer = None


if __name__ == "__main__":
    main()
