import numpy as np
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals


@unlearning_metric(name="roadblock_classifier_quality")
def roadblock_classifier_quality(model, **kwargs):
    classify_batch = getattr(model, "roadblock_classify_batch", None)
    if not callable(classify_batch):
        raise TypeError("roadblock_classifier_quality requires the RoadBlock trainer")

    values = {}
    rates = {}
    for split, dataset in kwargs["data"].items():
        is_forget = split.startswith("forget")
        dataloader = DataLoader(
            dataset,
            batch_size=kwargs["batch_size"],
            collate_fn=kwargs["collators"],
        )
        split_values = {}
        for batch in dataloader:
            indices = batch.pop("index").tolist()
            predictions = classify_batch(batch, is_forget)
            split_values.update(dict(zip(indices, predictions)))
        values[split] = split_values
        rates[split] = float(
            np.mean([item["prediction"] for item in split_values.values()])
        )

    forget_rates = [value for key, value in rates.items() if key.startswith("forget")]
    retain_rates = [
        value for key, value in rates.items() if not key.startswith("forget")
    ]
    recall = float(np.mean(forget_rates))
    false_positive_rate = float(np.mean(retain_rates))
    summary = {
        **{f"{split}_positive_rate": rate for split, rate in rates.items()},
        "threshold": float(model.roadblock_classifier_threshold()),
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "balanced_accuracy": 0.5 * (recall + 1 - false_positive_rate),
    }
    return {
        "agg_value": summary["balanced_accuracy"],
        "summary": summary,
        "value_by_split": values,
    }


@unlearning_metric(name="roadblock_diagnostics")
def roadblock_diagnostics(model, **kwargs):
    diagnose_batch = getattr(model, "roadblock_diagnostic_batch", None)
    if not callable(diagnose_batch):
        raise TypeError("roadblock_diagnostics requires the RoadBlock trainer")

    dataloader = DataLoader(
        kwargs["data"],
        batch_size=kwargs["batch_size"],
        collate_fn=kwargs["collators"],
    )
    value_by_index = run_batchwise_evals(
        model,
        dataloader,
        lambda model, batch: diagnose_batch(batch),
        {},
        "Calculating RoAdBlock diagnostics",
    )
    names = next(iter(value_by_index.values())).keys()
    summary = {
        name: float(np.mean([values[name] for values in value_by_index.values()]))
        for name in names
    }
    return {
        "agg_value": summary["correction_rms"],
        "summary": summary,
        "value_by_index": value_by_index,
    }
