import torch
import torch.nn.functional as F
from torch import nn


class RoadBlockClassifier:
    """Small prompt routers used by RoadBlock."""

    MODES = {"oracle", "none", "guard", "curate"}

    def __init__(self, mode, hidden_size, device):
        if mode not in self.MODES:
            raise ValueError(
                f"classifier must be one of {sorted(self.MODES)}, got {mode}"
            )
        self.mode = mode
        self.device = device
        self.head = (
            nn.Linear(hidden_size, 1, device=device) if mode == "guard" else None
        )
        self.forget_embeddings = None
        self.threshold = 0.5

    @property
    def needs_embeddings(self):
        return self.mode in {"guard", "curate"}

    def fit(self, forget_embeddings, retain_embeddings):
        forget_embeddings = F.normalize(forget_embeddings.float(), dim=-1).to(
            self.device
        )
        retain_embeddings = F.normalize(retain_embeddings.float(), dim=-1).to(
            self.device
        )

        if self.mode == "guard":
            features = torch.cat([forget_embeddings, retain_embeddings])
            labels = torch.cat(
                [
                    torch.ones(len(forget_embeddings), device=self.device),
                    torch.zeros(len(retain_embeddings), device=self.device),
                ]
            )
            optimizer = torch.optim.AdamW(self.head.parameters(), lr=1e-2)
            self.head.train()
            for _ in range(100):
                loss = F.binary_cross_entropy_with_logits(
                    self.head(features).squeeze(-1), labels
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.head.eval()
            return

        if self.mode == "curate":
            self.forget_embeddings = forget_embeddings
            forget_similarity = forget_embeddings @ forget_embeddings.T
            forget_similarity.fill_diagonal_(-1)
            positive_scores = forget_similarity.max(dim=-1).values
            negative_scores = (
                (retain_embeddings @ forget_embeddings.T).max(dim=-1).values
            )
            scores = torch.cat([positive_scores, negative_scores])
            labels = torch.cat(
                [torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]
            )
            thresholds = scores.unique()
            predictions = scores[:, None] >= thresholds[None, :]
            recall = predictions[labels.bool()].float().mean(dim=0)
            specificity = (~predictions[~labels.bool()]).float().mean(dim=0)
            self.threshold = float(thresholds[(recall + specificity).argmax()])

    @torch.no_grad()
    def score(self, embeddings):
        embeddings = F.normalize(embeddings.float(), dim=-1).to(self.device)
        if self.mode == "guard":
            return self.head(embeddings).squeeze(-1).sigmoid()
        if self.mode == "curate":
            return (embeddings @ self.forget_embeddings.T).max(dim=-1).values
        raise RuntimeError(f"{self.mode} does not score embeddings")

    @torch.no_grad()
    def predict(self, embeddings):
        scores = self.score(embeddings)
        return scores, scores >= self.threshold
