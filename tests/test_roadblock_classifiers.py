import torch
import pytest
from torch import nn

from trainer.unlearn.roadblock_classifier import RoadBlockClassifier


def _cache():
    return {
        "retain": torch.tensor(
            [[-1.0, 0.0, 0.0, 0.0], [-1.0, 0.1, 0.0, 0.0]]
        ),
        "forget": {
            "request_01": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.1, 0.0, 0.0]]
            ),
            "request_02": torch.tensor(
                [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.1, 0.0]]
            ),
        },
    }


@pytest.mark.parametrize("mode", ["guard_multiclass", "guard_prototype"])
def test_continual_classifiers_route_retain_and_requests(mode):
    torch.manual_seed(0)
    classifier = RoadBlockClassifier(
        mode,
        hidden_size=4,
        device=torch.device("cpu"),
        continual=True,
        num_centroids=2,
        seed=7,
    )
    classifier.fit_replay(_cache())

    embeddings = torch.tensor(
        [[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    _, routes = classifier.route(embeddings)

    assert routes[0] is None
    assert routes[1] == "request_01"
    assert routes[2] == "request_02"


def test_prototype_count_is_validated():
    classifier = RoadBlockClassifier(
        "guard_prototype",
        hidden_size=4,
        device=torch.device("cpu"),
        continual=True,
        num_centroids=3,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        classifier.fit_replay(_cache())


def test_multiclass_margin_threshold_can_reject_request():
    classifier = RoadBlockClassifier(
        "guard_multiclass",
        hidden_size=4,
        device=torch.device("cpu"),
        continual=True,
        router_threshold=0.5,
    )
    classifier.request_names = ["request_01"]

    class FixedHead(nn.Module):
        def forward(self, features):
            return torch.tensor([[0.0, 0.25], [0.0, 1.0]])

    classifier.head = FixedHead()
    scores, routes = classifier.route(torch.zeros(2, 4))

    assert torch.equal(scores, torch.tensor([0.25, 1.0]))
    assert routes == [None, "request_01"]


def test_multiclass_diagnostics_report_oracle_gap_components():
    classifier = RoadBlockClassifier(
        "guard_multiclass",
        hidden_size=4,
        device=torch.device("cpu"),
        continual=True,
        router_threshold=0.0,
    )

    class SeparableHead(nn.Module):
        def forward(self, features):
            return torch.stack(
                (-features[:, 0] - features[:, 1], 2 * features[:, 0], 2 * features[:, 1]),
                dim=-1,
            )

    classifier.head = SeparableHead()
    classifier.request_names = ["request_01", "request_02"]
    diagnostics = classifier.diagnostics(_cache())

    assert diagnostics["available"] is True
    assert diagnostics["router_oracle_gap"] == 0.0
    assert diagnostics["forget_oracle_gap"] == 0.0
    assert diagnostics["retain_false_positive_rate"] == 0.0


def test_continual_modes_are_rejected_outside_runner():
    with pytest.raises(ValueError, match="requires the continual"):
        RoadBlockClassifier(
            "guard_multiclass", 4, torch.device("cpu"), continual=False
        )
    with pytest.raises(ValueError, match="ordinary single-request"):
        RoadBlockClassifier("guard", 4, torch.device("cpu"), continual=True)

    for mode in ("none", "curate"):
        with pytest.raises(ValueError, match="classifier must be one of"):
            RoadBlockClassifier(mode, 4, torch.device("cpu"))
