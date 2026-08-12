import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from evals.continual import aggregate_request_logs, request_partitions, summarize_logs


def _request_log(probabilities, truth_ratios, rouge, extraction, mia_scores):
    values = range(len(probabilities))
    return {
        "forget_Q_A_Prob": {
            "value_by_index": {
                str(index): {"prob": probability}
                for index, probability in zip(values, probabilities)
            }
        },
        "forget_Q_A_ROUGE": {
            "value_by_index": {
                str(index): {"rougeL_recall": score}
                for index, score in enumerate(rouge)
            }
        },
        "forget_truth_ratio": {
            "value_by_index": {
                str(index): {"score": score}
                for index, score in enumerate(truth_ratios)
            }
        },
        "extraction_strength": {
            "value_by_index": {
                str(index): {"score": score}
                for index, score in enumerate(extraction)
            }
        },
        "mia_min_k": {
            "forget": {
                "value_by_index": {
                    str(index): {"score": score}
                    for index, score in enumerate(mia_scores)
                }
            },
            "holdout": {
                "value_by_index": {
                    "0": {"score": 0.7},
                    "1": {"score": 0.9},
                }
            },
        },
    }


def test_request_partitions_cover_forget05_once():
    partitions = request_partitions()

    assert len(partitions) == 5
    assert all(partition["stop"] - partition["start"] == 40 for partition in partitions)
    assert [index for partition in partitions for index in range(partition["start"], partition["stop"])] == list(range(200))


def test_request_partitions_reject_incomplete_partition():
    with pytest.raises(ValueError, match="exactly partition"):
        request_partitions(num_requests=4)


def test_cumulative_metrics_are_recomputed_from_examples():
    first = _request_log(
        probabilities=[0.1, 0.3],
        truth_ratios=[0.5, 2.0],
        rouge=[0.2, 0.4],
        extraction=[0.0, 0.5],
        mia_scores=[0.1, 0.4],
    )
    second = _request_log(
        probabilities=[0.5, 0.7],
        truth_ratios=[1.0, 4.0],
        rouge=[0.6, 0.8],
        extraction=[0.5, 1.0],
        mia_scores=[0.2, 0.6],
    )
    reference_truth = [0.8, 0.9, 1.1, 1.2]
    reference = {
        "forget_truth_ratio": {
            "value_by_index": {
                str(index): {"score": score}
                for index, score in enumerate(reference_truth)
            }
        },
        "mia_min_k": {"agg_value": 0.6},
    }

    cumulative = aggregate_request_logs([(0, first), (2, second)], reference)

    assert list(cumulative["forget_Q_A_Prob"]["value_by_index"]) == ["0", "1", "2", "3"]
    assert cumulative["forget_Q_A_Prob"]["agg_value"] == pytest.approx(0.4)
    assert cumulative["forget_Q_A_ROUGE"]["agg_value"] == pytest.approx(0.5)
    assert cumulative["forget_truth_ratio"]["agg_value"] == pytest.approx(0.5625)
    assert cumulative["extraction_strength"]["agg_value"] == pytest.approx(0.5)

    expected_ks = ks_2samp([0.5, 2.0, 1.0, 4.0], reference_truth)
    assert cumulative["forget_quality"]["agg_value"] == pytest.approx(expected_ks.pvalue)
    assert cumulative["forget_quality"]["summary"]["ks_statistic"] == pytest.approx(expected_ks.statistic)

    expected_auc = roc_auc_score(
        [0, 0, 0, 0, 1, 1], [0.1, 0.4, 0.2, 0.6, 0.7, 0.9]
    )
    assert cumulative["mia_min_k"]["agg_value"] == pytest.approx(expected_auc)
    expected_privleak = ((1 - expected_auc) - 0.4) / 0.4 * 100
    assert cumulative["privleak"]["agg_value"] == pytest.approx(expected_privleak)

    summary = summarize_logs(cumulative)
    assert summary["forget_quality_ks_statistic"] == pytest.approx(expected_ks.statistic)
    assert set(summary) == {
        "forget_truth_ratio",
        "forget_quality",
        "forget_quality_ks_statistic",
        "forget_Q_A_Prob",
        "forget_Q_A_ROUGE",
        "privleak",
        "extraction_strength",
    }
