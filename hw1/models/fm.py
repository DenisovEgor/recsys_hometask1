import numpy as np
import torch
from torch import nn

from .base import BaseRecommender


class FactorizationMachine(BaseRecommender):
    def __init__(self, data, n_factors=16, n_epochs=6, batch_size=2048, use_metadata=True):
        super().__init__(data)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.step_loss_history = []

        self.item_indices = (data.n_users + data.item_ids)[:, None]
        self.item_values = torch.ones(data.n_items, 1)
        if use_metadata:
            metadata = data.item_feature_indices + data.n_users + data.n_items
            self.item_indices = torch.cat([self.item_indices, metadata], dim=1)
            self.item_values = torch.cat([self.item_values, data.item_feature_values], dim=1)

        n_features = data.n_users + data.n_items + (data.n_item_features if use_metadata else 0)
        self.factors = nn.Parameter(torch.randn(n_features, n_factors) * .01)
        self.linear = nn.Parameter(torch.zeros(n_features))
        self.bias = nn.Parameter(data.mean_rating.clone())

    def forward(self, users, items):
        indices = torch.cat([users[:, None], self.item_indices[items]], dim=1)
        values = torch.cat([torch.ones(len(users), 1), self.item_values[items]], dim=1)

        vectors = self.factors[indices] * values[:, :, None]
        interactions = .5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1)).sum(dim=1)
        return self.bias + (self.linear[indices] * values).sum(dim=1) + interactions

    def fit(self, loss, optimizer):
        self.train()
        data = self.data

        for _ in range(self.n_epochs):
            order = np.random.permutation(data.n_interactions)
            total = 0.0

            for start in range(0, data.n_interactions, self.batch_size):
                ids = order[start:start + self.batch_size]
                optimizer.zero_grad()
                predictions = self(data.users[ids], data.items[ids])
                value = loss(predictions, data.ratings[ids])

                value.backward()
                optimizer.step()

                batch_loss = value.item()
                self.step_loss_history.append(batch_loss)
                total += batch_loss * len(ids)

            self.loss_history.append(total / data.n_interactions)

        return self

    def score(self, users, timestamps=None):
        items = self.data.item_ids
        flat_users = users.repeat_interleave(len(items))
        flat_items = items.repeat(len(users))

        scores = [self(u, i) for u, i in zip(flat_users.split(16384), flat_items.split(16384))]
        return torch.cat(scores).reshape(len(users), len(items))
