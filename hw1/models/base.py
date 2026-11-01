from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TrainData:
    n_users: int
    n_items: int
    users: torch.Tensor
    items: torch.Tensor
    ratings: torch.Tensor
    timestamps: torch.Tensor
    item_feature_indices: torch.Tensor
    item_feature_values: torch.Tensor
    n_item_features: int
    relevance_threshold: float = 4.0

    def __post_init__(self):
        for name in ['users', 'items', 'timestamps', 'item_feature_indices']:
            setattr(self, name, torch.tensor(np.array(getattr(self, name)), dtype=torch.long))
        for name in ['ratings', 'item_feature_values']:
            setattr(self, name, torch.tensor(np.array(getattr(self, name)), dtype=torch.float32))

        self.n_interactions = len(self.users)
        self.item_ids = torch.arange(self.n_items)
        self.mean_rating = self.ratings.mean()
        self.positive_mask = self.ratings >= self.relevance_threshold

        self.positives = torch.zeros(self.n_users, self.n_items)
        self.positives[self.users[self.positive_mask], self.items[self.positive_mask]] = 1

        self.seen = torch.zeros(self.n_users, self.n_items, dtype=torch.bool)
        self.seen[self.users, self.items] = True
        self.rating_matrix = torch.zeros_like(self.positives)
        self.rating_matrix[self.users, self.items] = self.ratings
        self.gram = self.positives.T @ self.positives

        mask = self.positive_mask
        users = self.users[mask].numpy()
        items = self.items[mask].numpy()
        times = self.timestamps[mask].numpy()
        order = np.lexsort((items, times, users))
        offsets = np.r_[0, np.cumsum(np.bincount(users, minlength=self.n_users))]
        self.positive_histories = offsets, items[order], times[order]


class BaseRecommender(nn.Module, ABC):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.loss_history = []

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def score(self, users, timestamps=None):
        pass

    @torch.no_grad()
    def recommend(self, user_ids, timestamps, seen_mask, k=10):
        users = torch.as_tensor(user_ids, dtype=torch.long)
        recommendations = []
        self.eval()

        for batch in users.split(128):
            scores = self.score(batch, timestamps)
            scores[seen_mask[batch]] = -torch.inf

            ids = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
            ids[torch.isneginf(scores.gather(1, ids))] = -1
            recommendations.append(ids)

        return torch.cat(recommendations).numpy()
