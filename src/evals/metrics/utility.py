import math

import numpy as np
import scipy as sc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import aggregate_to_1D


@unlearning_metric(name="hm_aggregate")
def hm_aggregate(model, **kwargs):
    values = [result["agg_value"] for _, result in kwargs["pre_compute"].items()]
    return {"agg_value": sc.stats.hmean(values)}


@unlearning_metric(name="constrained_selection")
def constrained_selection(model, **kwargs):
    """Rank unlearning runs without rewarding destructive metric collapse.

    Preservation constraints are lexicographically stronger than forget
    quality: runs that regress utility or forget truth ratio always score below
    runs that preserve both. A positive score additionally requires the TOFU
    forget-quality hypothesis test to pass its configured threshold.
    """

    metrics = kwargs["pre_compute"]
    forget_quality = metrics["forget_quality"]["agg_value"]
    forget_truth_ratio = metrics["forget_truth_ratio"]["agg_value"]
    model_utility = metrics["model_utility"]["agg_value"]
    utility_floor = kwargs["utility_floor"]
    truth_ratio_floor = kwargs["truth_ratio_floor"]
    forget_quality_threshold = kwargs["forget_quality_threshold"]
    log_gap_scale = kwargs.get("log_gap_scale", 20.0)

    utility_ok = model_utility >= utility_floor
    truth_ratio_ok = forget_truth_ratio >= truth_ratio_floor
    forgetting_ok = forget_quality >= forget_quality_threshold

    if not utility_ok or not truth_ratio_ok:
        utility_violation = max(0.0, utility_floor - model_utility) / utility_floor
        truth_violation = (
            max(0.0, truth_ratio_floor - forget_truth_ratio) / truth_ratio_floor
        )
        score = -2.0 - utility_violation - truth_violation
    elif not forgetting_ok:
        log_gap = math.log10(forget_quality_threshold) - math.log10(
            max(forget_quality, 1e-300)
        )
        score = -min(0.999999, log_gap / log_gap_scale)
    else:
        score = 1.0 + min(float(forget_quality), 1.0)

    return {
        "agg_value": score,
        "feasible": float(utility_ok and truth_ratio_ok and forgetting_ok),
        "utility_preserved": float(utility_ok),
        "truth_ratio_preserved": float(truth_ratio_ok),
        "forget_quality_passed": float(forgetting_ok),
    }


@unlearning_metric(name="classifier_prob")
def classifier_prob(model, **kwargs):
    batch_size = kwargs.get("batch_size", 32)
    max_length = kwargs.get("max_length", 512)
    class_id = kwargs.get("class_id", 0)
    text_key = kwargs.get("text_key", "generation")
    classifier_model_args = kwargs["classifier_model_args"]
    classifier_tokenization_args = kwargs["classifier_tokenization_args"]
    device = kwargs.get("device", "cuda")

    tokenizer = AutoTokenizer.from_pretrained(**classifier_tokenization_args)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        **classifier_model_args
    ).to(device)

    data = kwargs["pre_compute"]["text"]["value_by_index"]
    data_list = [
        {"text": entry[text_key], "index": int(key)} for key, entry in data.items()
    ]

    # Create DataLoader
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    scores_by_index = {}
    for batch in tqdm(dataloader):
        batch_texts = batch["text"]
        batch_indices = batch["index"].tolist()

        # Tokenize the batch of texts
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run the classifier
        with torch.no_grad():
            outputs = classifier(**inputs)
        # Convert logits to probabilities
        scores = F.softmax(outputs.logits, dim=-1)[:, class_id].cpu().numpy().tolist()

        # Map predictions to labels
        for idx, prob, text in zip(batch_indices, scores, batch_texts):
            # Add the prediction to the original data
            scores_by_index[idx] = {"score": prob, text_key: text}
    class_scores = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    class_scores = aggregate_to_1D(class_scores)
    return {"agg_value": np.mean(class_scores), "value_by_index": scores_by_index}
