import math

import torch
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric


def _pairwise_auc(positive_scores, negative_scores):
    if not positive_scores or not negative_scores:
        return float("nan")
    wins = sum(p > n for p in positive_scores for n in negative_scores)
    ties = sum(p == n for p in positive_scores for n in negative_scores)
    return (wins + 0.5 * ties) / (len(positive_scores) * len(negative_scores))


@unlearning_metric(name="router_activations")
def router_activations(model, **kwargs):
    """Evaluate the calibrated router on each configured dataset.

    The result is independent of the effective evaluation mode: an always-on
    or oracle run still reports what the calibrated likelihood-ratio router
    would have done on every subset.
    """

    diagnose = getattr(model, "_unlearn_head_router_diagnostics", None)
    if diagnose is None:
        raise RuntimeError("router_activations requires evaluation through UnlearnHead")

    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    results = {}
    scores_by_split = {}

    for split_name, dataset in data.items():
        scores = []
        activations = []
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
        for batch in dataloader:
            batch = {
                key: value.to(model.device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }
            with torch.no_grad():
                batch_scores, batch_activations = diagnose(**batch)
            scores.extend(batch_scores.float().cpu().tolist())
            activations.extend(batch_activations.float().cpu().tolist())

        scores_by_split[split_name] = scores
        results[f"{split_name}_activation_rate"] = (
            sum(activations) / len(activations) if activations else float("nan")
        )
        results[f"{split_name}_score_mean"] = (
            sum(scores) / len(scores) if scores else float("nan")
        )

    forget_scores = scores_by_split.get("forget_exact", [])
    retain_scores = scores_by_split.get("retain", [])
    auc = _pairwise_auc(forget_scores, retain_scores)
    results["auc_forget_vs_retain"] = auc
    results["agg_value"] = auc if not math.isnan(auc) else 0.0
    return results
