import torch
from torch import nn

from .base import BaseRecommender


class SLIM(BaseRecommender):
    def __init__(self, data, l1=1.0, l2=10.0, max_iter=100, tol=1e-4, patience=3):
        super().__init__(data)
        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience

        self.coefficients = nn.Parameter(torch.zeros(data.n_items, data.n_items))

        self.converged = False
        self.n_iter = 0
        self.loss_steps = []

        self.gram = data.gram
        self.denominator = self.gram.diag() + l2
        self.constant = .5 * data.positives.sum()

    def objective(self):
        weight = self.coefficients
        return (
            self.constant + .5 * (weight * (self.gram @ weight)).sum()
            - (self.gram * weight).sum()
            + .5 * self.l2 * weight.square().sum() + self.l1 * weight.sum()
        )

    @torch.no_grad()
    def fit(self):
        self.train()
        weight = self.coefficients

        self.loss_history.append(float(self.objective()))
        self.loss_steps.append(0)
        stable_passes = 0

        for self.n_iter in range(1, self.max_iter + 1):
            for j in range(self.data.n_items):
                old = weight[j].clone()
                numerator = self.gram[j] - self.gram[j] @ weight + self.gram[j, j] * old - self.l1
                weight[j] = (numerator / self.denominator[j]).clamp_min(0)
                weight[j, j] = 0

            previous = self.loss_history[-1]
            current = float(self.objective())
            improvement = (previous - current) / max(abs(previous), 1.0)
            stable_passes = stable_passes + 1 if improvement <= self.tol else 0

            self.loss_history.append(current)
            self.loss_steps.append(self.n_iter)
            self.converged = stable_passes >= self.patience

            if self.converged:
                break

        return self

    def score(self, users, timestamps=None):
        return self.data.positives[users] @ self.coefficients
