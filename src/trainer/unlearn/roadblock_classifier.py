import torch
import torch.nn.functional as F
from torch import nn


class RoadBlockClassifier:
    """Small prompt routers used by RoadBlock.

    The continual modes are deliberately activation-only.  The owning
    ``PeftModel`` keeps the replay tensors between trainer instances; this
    object only owns the freshly fitted head for the current stage.
    """

    MODES = {"oracle", "guard", "guard_multiclass", "guard_prototype"}
    CONTINUAL_MODES = {"guard_multiclass", "guard_prototype"}

    def __init__(
        self,
        mode,
        hidden_size,
        device,
        request_name=None,
        request_names=None,
        num_centroids=2,
        seed=0,
        continual=False,
        router_threshold=0.0,
    ):
        if mode not in self.MODES:
            raise ValueError(
                f"classifier must be one of {sorted(self.MODES)}, got {mode}"
            )
        if continual and mode == "guard":
            raise ValueError(
                "classifier=guard is for ordinary single-request runs; "
                "use guard_multiclass or guard_prototype for continual unlearning"
            )
        if not continual and mode in self.CONTINUAL_MODES:
            raise ValueError(
                f"classifier={mode} requires the continual unlearning runner"
            )
        if int(num_centroids) < 1:
            raise ValueError("num_centroids must be at least 1")

        self.mode = mode
        self.device = device
        self.hidden_size = hidden_size
        self.request_name = request_name
        self.request_names = list(request_names or [])
        self.num_centroids = int(num_centroids)
        self.seed = int(seed)
        self.continual = bool(continual)
        self.router_threshold = float(router_threshold)
        self.head = None
        self.prototypes = {}
        self.threshold = (
            self.router_threshold if mode == "guard_multiclass" else 0.5
        )

        if mode == "guard":
            self.head = self._make_head(1)

    def _make_head(self, output_size):
        return nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, output_size),
        ).to(self.device)

    @property
    def needs_embeddings(self):
        return self.mode == "guard" or self.mode in self.CONTINUAL_MODES

    @staticmethod
    def _normalize(embeddings):
        return F.normalize(embeddings.float(), dim=-1)

    @staticmethod
    def _inverse_frequency_weights(labels, num_classes):
        counts = torch.bincount(labels, minlength=num_classes).float()
        if torch.any(counts == 0):
            raise ValueError("Every continual classifier class needs an activation")
        return labels.new_tensor(len(labels), dtype=torch.float32) / (
            num_classes * counts
        )

    def _fit_binary_head(self, forget_embeddings, retain_embeddings):
        features = torch.cat([forget_embeddings, retain_embeddings]).to(self.device)
        labels = torch.cat(
            [
                torch.ones(len(forget_embeddings), device=self.device),
                torch.zeros(len(retain_embeddings), device=self.device),
            ]
        )
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=1e-2)
        positive_weight = torch.tensor(
            len(retain_embeddings) / len(forget_embeddings), device=self.device
        )
        self.head.train()
        for _ in range(100):
            logits = self.head(features).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=positive_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.head.eval()

    def fit(self, forget_embeddings, retain_embeddings):
        """Fit the ordinary single-request GUARD classifier."""
        if self.mode != "guard":
            raise ValueError(f"fit is only valid for classifier=guard, got {self.mode}")
        forget_embeddings = self._normalize(forget_embeddings).to(self.device)
        retain_embeddings = self._normalize(retain_embeddings).to(self.device)
        self._fit_binary_head(forget_embeddings, retain_embeddings)

    def fit_replay(self, activation_cache):
        """Fit a continual classifier from cached, frozen activations."""
        if self.mode not in self.CONTINUAL_MODES:
            raise ValueError(
                f"fit_replay is only valid for continual classifiers, got {self.mode}"
            )
        retain = self._normalize(activation_cache["retain"])
        names = list(activation_cache["forget"])
        if not names:
            raise ValueError("Continual classifier needs at least one forget request")
        forget = [self._normalize(activation_cache["forget"][name]) for name in names]
        self.request_names = names

        if self.mode == "guard_multiclass":
            features = torch.cat([retain, *forget]).to(self.device)
            labels = torch.cat(
                [
                    torch.zeros(len(retain), dtype=torch.long),
                    *[
                        torch.full((len(values),), index, dtype=torch.long)
                        for index, values in enumerate(forget, start=1)
                    ],
                ]
            ).to(self.device)
            self.head = self._make_head(len(names) + 1)
            class_weights = self._inverse_frequency_weights(labels, len(names) + 1).to(
                self.device
            )
            optimizer = torch.optim.AdamW(self.head.parameters(), lr=1e-2)
            self.head.train()
            for _ in range(100):
                logits = self.head(features)
                loss = F.cross_entropy(logits, labels, reduction="none")
                loss = (loss * class_weights[labels]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.head.eval()
            return

        # The prototype selector uses the same successful binary GUARD gate,
        # then compares all request centroids in one shared cosine space.
        if self.num_centroids > min(len(values) for values in forget):
            raise ValueError(
                "num_centroids cannot exceed the examples in any forget request"
            )
        self.head = self._make_head(1)
        self._fit_binary_head(torch.cat(forget), retain)

        from sklearn.cluster import KMeans

        self.prototypes = {}
        for name, values in zip(names, forget):
            values_cpu = values.cpu().numpy()
            kmeans = KMeans(
                n_clusters=self.num_centroids,
                n_init=10,
                random_state=self.seed,
            ).fit(values_cpu)
            centers = torch.as_tensor(
                kmeans.cluster_centers_, dtype=torch.float32, device=self.device
            )
            self.prototypes[name] = self._normalize(centers)

    @torch.no_grad()
    def _binary_scores(self, embeddings):
        return (
            self.head(self._normalize(embeddings).to(self.device))
            .squeeze(-1)
            .sigmoid()
        )

    @torch.no_grad()
    def route(self, embeddings):
        """Return a score and one adapter name (or ``None``) per example."""
        embeddings = self._normalize(embeddings).to(self.device)
        if self.mode == "guard":
            scores = self._binary_scores(embeddings)
            routes = [
                self.request_name if value else None
                for value in scores >= self.threshold
            ]
            return scores, routes

        if self.mode == "guard_multiclass":
            logits = self.head(embeddings)
            choices = logits.argmax(dim=-1)
            request_logits = logits[:, 1:]
            scores = request_logits.max(dim=-1).values - logits[:, 0]
            routes = [
                (
                    self.request_names[index - 1]
                    if index and score >= self.threshold
                    else None
                )
                for index, score in zip(choices.tolist(), scores.tolist())
            ]
            return scores, routes

        if self.mode == "guard_prototype":
            gate_scores = self._binary_scores(embeddings)
            prototype_scores = torch.stack(
                [
                    (embeddings @ centers.T).max(dim=-1).values
                    for centers in self.prototypes.values()
                ],
                dim=-1,
            )
            choices = prototype_scores.argmax(dim=-1)
            routes = [
                name if gate else None
                for name, gate in zip(
                    [list(self.prototypes)[index] for index in choices.tolist()],
                    (gate_scores >= self.threshold).tolist(),
                )
            ]
            return gate_scores, routes

        raise RuntimeError(f"{self.mode} does not route embeddings")

    @torch.no_grad()
    def score(self, embeddings):
        scores, _ = self.route(embeddings)
        return scores

    @torch.no_grad()
    def predict(self, embeddings):
        scores, routes = self.route(embeddings)
        return scores, torch.tensor(
            [route is not None for route in routes], device=scores.device
        )

    @torch.no_grad()
    def diagnostics(self, activation_cache):
        """Compare learned routes with the oracle labels in the replay cache.

        The cache contains the only labels needed for this diagnostic: retain
        examples should route to ``None`` and each request's examples should
        route to that request.  This reports the routing error separately from
        the downstream unlearning metrics, so a gap to oracle can be attributed
        to the router rather than to the adapter itself.
        """
        if self.mode not in self.CONTINUAL_MODES:
            return {"available": False, "reason": "oracle_or_single_request"}

        retain = activation_cache.get("retain")
        forget = activation_cache.get("forget", {})
        if retain is None or not forget:
            return {"available": False, "reason": "activation_cache_incomplete"}

        names = list(forget)
        embeddings = torch.cat([retain, *[forget[name] for name in names]])
        scores, routes = self.route(embeddings)
        routes = list(routes)
        expected = [None] * len(retain)
        for name in names:
            expected.extend([name] * len(forget[name]))

        def fraction(values, predicate):
            if not values:
                return 0.0
            return float(sum(predicate(value) for value in values) / len(values))

        exact = [route == target for route, target in zip(routes, expected)]
        retain_routes = routes[: len(retain)]
        forget_offset = len(retain)
        forget_routes = routes[forget_offset:]
        forget_expected = expected[forget_offset:]
        forget_correct = [
            route == target for route, target in zip(forget_routes, forget_expected)
        ]
        forget_routed = [route is not None for route in forget_routes]
        wrong_adapter = [
            route is not None and route != target
            for route, target in zip(forget_routes, forget_expected)
        ]

        per_request = {}
        cursor = 0
        for name in names:
            count = len(forget[name])
            request_routes = forget_routes[cursor : cursor + count]
            cursor += count
            per_request[name] = {
                "count": count,
                "route_accuracy": fraction(
                    request_routes,
                    lambda route, expected_name=name: route == expected_name,
                ),
                "recall": fraction(request_routes, lambda route: route is not None),
                "rejection_rate": fraction(request_routes, lambda route: route is None),
                "wrong_adapter_rate": fraction(
                    request_routes,
                    lambda route, expected_name=name: route is not None
                    and route != expected_name,
                ),
            }

        score_values = torch.as_tensor(scores).detach().float().cpu()
        return {
            "available": True,
            "threshold": float(self.threshold),
            "num_examples": len(expected),
            "num_retain_examples": len(retain),
            "num_forget_examples": len(forget_expected),
            "router_oracle_accuracy": float(sum(exact) / len(exact)),
            "router_oracle_gap": float(1.0 - sum(exact) / len(exact)),
            "forget_route_accuracy": float(sum(forget_correct) / len(forget_correct)),
            "forget_oracle_gap": float(
                1.0 - sum(forget_correct) / len(forget_correct)
            ),
            "forget_recall": float(sum(forget_routed) / len(forget_routed)),
            "forget_rejection_rate": float(
                1.0 - sum(forget_routed) / len(forget_routed)
            ),
            "wrong_adapter_rate": float(sum(wrong_adapter) / len(wrong_adapter)),
            "retain_false_positive_rate": fraction(
                retain_routes, lambda route: route is not None
            ),
            "retain_rejection_rate": fraction(
                retain_routes, lambda route: route is None
            ),
            "score_mean": float(score_values.mean()),
            "score_std": float(score_values.std(unbiased=False)),
            "per_request": per_request,
        }
