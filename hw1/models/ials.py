import torch
from torch import nn

from .base import BaseRecommender


class ImplicitALS(BaseRecommender):
    def __init__(self, data, n_factors=32, alpha=20.0, regularization=1.0, n_iterations=10):
        super().__init__(data)
        self.n_factors = n_factors
        self.alpha = alpha
        self.regularization = regularization
        self.n_iterations = n_iterations

        self.user_factors = nn.Parameter(torch.randn(data.n_users, self.n_factors) * .01)
        self.item_factors = nn.Parameter(torch.randn(data.n_items, self.n_factors) * .01)

        user_counts = data.seen.sum(dim=1).float()
        item_counts = data.seen.sum(dim=0).float()
        self.user_reg_weights = user_counts / user_counts.mean()
        self.item_reg_weights = item_counts / item_counts.mean()
        self.user_penalty = regularization * self.user_reg_weights[:, None, None] * torch.eye(n_factors)
        self.item_penalty = regularization * self.item_reg_weights[:, None, None] * torch.eye(n_factors)

        self.seen = data.seen.float().to_sparse_coo().coalesce()
        self.positives = data.positives.to_sparse_coo().coalesce()
        self.seen_t = self.seen.T.coalesce()
        self.positives_t = self.positives.T.coalesce()
        self.targets = data.positive_mask.float()

    def solve(self, seen, positives, fixed, penalty):
        f = self.n_factors
        outer = (fixed[:, :, None] * fixed[:, None, :]).reshape(len(fixed), -1)
        systems = self.alpha * torch.sparse.mm(seen, outer).reshape(-1, f, f)
        systems += fixed.T @ fixed
        systems += penalty

        rhs = (1 + self.alpha) * torch.sparse.mm(positives, fixed)
        return torch.linalg.solve(systems, rhs.unsqueeze(-1)).squeeze(-1)

    def objective(self):
        a, b = self.user_factors, self.item_factors
        predictions = (a[self.data.users] * b[self.data.items]).sum(dim=1)
        unweighted = ((a.T @ a) * (b.T @ b)).sum()
        correction = ((1 + self.alpha) * (self.targets - predictions).square() - predictions.square()).sum()

        penalty = (
            (a.square().sum(dim=1) * self.user_reg_weights).sum()
            + (b.square().sum(dim=1) * self.item_reg_weights).sum()
        )
        return unweighted + correction + self.regularization * penalty

    @torch.no_grad()
    def fit(self):
        self.train()
        self.loss_history.append(float(self.objective()))

        for _ in range(self.n_iterations):
            self.user_factors.copy_(self.solve(self.seen, self.positives, self.item_factors, self.user_penalty))
            self.item_factors.copy_(self.solve(self.seen_t, self.positives_t, self.user_factors, self.item_penalty))
            self.loss_history.append(float(self.objective()))

        return self

    def score(self, users, timestamps=None):
        return self.user_factors[users] @ self.item_factors.T
