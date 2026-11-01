import torch
from torch import nn

from .base import BaseRecommender


class ItemKNN(BaseRecommender):
    def __init__(self, data, n_neighbors=100):
        super().__init__(data)
        self.n_neighbors = n_neighbors

        self.seen = data.seen.float()
        self.user_mean = data.rating_matrix.sum(dim=1) / self.seen.sum(dim=1)
        self.centered = (data.rating_matrix - self.user_mean[:, None]) * self.seen

        common_squares = self.centered.square().T @ self.seen
        self.denominator = (common_squares * common_squares.T).sqrt().clamp_min(1e-12)

        self.similarity = nn.Parameter(torch.zeros(data.n_items, data.n_items))

    @torch.no_grad()
    def fit(self):
        self.train()

        similarity = (self.centered.T @ self.centered) / self.denominator
        similarity.fill_diagonal_(0)

        neighbors = torch.argsort(similarity, dim=0, descending=True, stable=True)[:self.n_neighbors]
        self.similarity.zero_().scatter_(0, neighbors, similarity.gather(0, neighbors))

        return self

    def score(self, users, timestamps=None):
        numerator = self.centered[users] @ self.similarity
        denominator = self.seen[users] @ self.similarity.abs()
        return self.user_mean[users, None] + numerator / denominator.clamp_min(1e-12)
