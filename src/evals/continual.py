from copy import deepcopy

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


REQUEST_METRICS = (
    "forget_truth_ratio",
    "forget_quality",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "privleak",
    "extraction_strength",
)


def request_partitions(num_requests=5, examples_per_request=40, dataset_size=200):
    if num_requests * examples_per_request != dataset_size:
        raise ValueError("Continual requests must exactly partition the forget dataset")
    return [
        {
            "name": f"request_{request + 1:02d}",
            "start": request * examples_per_request,
            "stop": (request + 1) * examples_per_request,
        }
        for request in range(num_requests)
    ]


def _merge_value_by_index(request_logs, metric_name):
    merged = {}
    for offset, logs in request_logs:
        values = logs[metric_name]["value_by_index"]
        for index, value in values.items():
            merged[str(offset + int(index))] = deepcopy(value)
    return merged


def _mean(values, key):
    scores = [np.asarray(value[key], dtype=float).mean() for value in values.values()]
    return float(np.mean(scores))


def _truth_ratio_mean(values):
    scores = np.asarray([value["score"] for value in values.values()], dtype=float)
    return float(np.mean(np.minimum(scores, 1 / (scores + 1e-10))))


def _reference_values(reference_logs, metric_name, branch=None):
    metric = reference_logs[metric_name]
    if branch is not None:
        metric = metric[branch]
    return metric["value_by_index"]


def aggregate_request_logs(request_logs, reference_logs):
    """Combine disjoint request evaluations and recompute cumulative metrics."""
    if not request_logs:
        raise ValueError("At least one request log is required")

    probability = _merge_value_by_index(request_logs, "forget_Q_A_Prob")
    rouge = _merge_value_by_index(request_logs, "forget_Q_A_ROUGE")
    truth_ratio = _merge_value_by_index(request_logs, "forget_truth_ratio")
    extraction = _merge_value_by_index(request_logs, "extraction_strength")

    truth_scores = np.asarray(
        [value["score"] for value in truth_ratio.values()], dtype=float
    )
    reference_truth = np.asarray(
        [
            value["score"]
            for value in _reference_values(
                reference_logs, "forget_truth_ratio"
            ).values()
        ],
        dtype=float,
    )
    forget_quality = ks_2samp(truth_scores, reference_truth)

    cumulative = {
        "forget_Q_A_Prob": {
            "agg_value": _mean(probability, "prob"),
            "value_by_index": probability,
        },
        "forget_Q_A_ROUGE": {
            "agg_value": _mean(rouge, "rougeL_recall"),
            "value_by_index": rouge,
        },
        "forget_truth_ratio": {
            "agg_value": _truth_ratio_mean(truth_ratio),
            "value_by_index": truth_ratio,
        },
        "forget_quality": {
            "agg_value": float(forget_quality.pvalue),
            "summary": {"ks_statistic": float(forget_quality.statistic)},
        },
        "extraction_strength": {
            "agg_value": _mean(extraction, "score"),
            "value_by_index": extraction,
        },
    }

    if all("mia_min_k" in logs for _, logs in request_logs):
        forget_mia = {}
        for offset, logs in request_logs:
            values = logs["mia_min_k"]["forget"]["value_by_index"]
            for index, value in values.items():
                forget_mia[str(offset + int(index))] = deepcopy(value)

        first_mia = request_logs[0][1]["mia_min_k"]
        holdout = deepcopy(first_mia["holdout"])
        holdout_values = holdout["value_by_index"]
        forget_scores = [value["score"] for value in forget_mia.values()]
        holdout_scores = [value["score"] for value in holdout_values.values()]
        labels = np.asarray(
            [0] * len(forget_scores) + [1] * len(holdout_scores), dtype=int
        )
        mia_auc = float(roc_auc_score(labels, forget_scores + holdout_scores))
        cumulative["mia_min_k"] = {
            "forget": {"value_by_index": forget_mia},
            "holdout": holdout,
            "auc": mia_auc,
            "agg_value": mia_auc,
        }

        reference_auc = float(reference_logs.get("mia_min_k", {}).get("agg_value", 0.5))
        score = 1 - mia_auc
        reference_score = 1 - reference_auc
        cumulative["privleak"] = {
            "agg_value": (score - reference_score) / (reference_score + 1e-10) * 100
        }

    return cumulative


def summarize_logs(logs, metric_names=REQUEST_METRICS):
    summary = {}
    for metric_name in metric_names:
        result = logs.get(metric_name)
        if not result:
            continue
        if result.get("agg_value") is not None:
            summary[metric_name] = result["agg_value"]
        for name, value in result.get("summary", {}).items():
            summary[f"{metric_name}_{name}"] = value
    return summary
