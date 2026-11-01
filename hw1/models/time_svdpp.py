import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseRecommender


class TimeSVDPlusPlus(BaseRecommender):
    def __init__(self, data, n_factors=16, n_bins=10, n_epochs=6, batch_size=2048):
        super().__init__(data)
        self.n_bins = n_bins
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.step_loss_history = []

        day = data.timestamps // 86400
        self.mean_rating = data.mean_rating
        self.min_day = int(day.min())
        self.n_days = int(day.max()) - self.min_day + 1
        self.last_timestamp = int(data.timestamps.max())
        self.day_keys = (data.users * self.n_days + day - self.min_day).unique(sorted=True)
        totals = torch.zeros(data.n_users).index_add_(0, data.users, day.float())
        self.mean_day = totals / torch.bincount(data.users, minlength=data.n_users)

        self.history_offsets, self.history_items, self.history_times = data.positive_histories
        self.history_norm = data.positives.sum(dim=1).sqrt().clamp_min(1)[:, None]

        self.p = nn.Parameter(torch.randn(data.n_users, n_factors) * .01)
        self.q = nn.Parameter(torch.randn(data.n_items, n_factors) * .01)
        self.y = nn.Parameter(torch.randn(data.n_items, n_factors) * .01)
        self.drift_p = nn.Parameter(torch.zeros(data.n_users, n_factors))
        self.day_p = nn.Parameter(torch.zeros(len(self.day_keys) + 1, n_factors))

        self.user_bias = nn.Parameter(torch.zeros(data.n_users))
        self.item_bias = nn.Parameter(torch.zeros(data.n_items))
        self.drift_bias = nn.Parameter(torch.zeros(data.n_users))
        self.day_bias = nn.Parameter(torch.zeros(len(self.day_keys) + 1))
        self.item_bin_bias = nn.Parameter(torch.zeros(data.n_items, self.n_bins))

    def context(self, users, timestamps):
        day = timestamps // 86400
        delta = day.float() - self.mean_day[users]
        dev = delta.sign() * (delta.abs() / 365).pow(.4)

        relative_day = day - self.min_day
        bins = (relative_day * self.n_bins // self.n_days).clamp(0, self.n_bins - 1)

        keys = users * self.n_days + relative_day.clamp(0, self.n_days - 1)
        positions = torch.searchsorted(self.day_keys, keys).clamp(max=len(self.day_keys) - 1)
        known = ((self.day_keys[positions] == keys) & (relative_day >= 0) & (relative_day < self.n_days))
        day_ids = torch.where(known, positions + 1, 0)
        return dev, bins, day_ids

    def implicit(self, users, timestamps):
        if (timestamps > self.last_timestamp).all():
            return self.data.positives[users] @ self.y / self.history_norm[users]

        chunks, sizes = [], []
        for user, timestamp in zip(users.numpy(), timestamps.numpy()):
            start, stop = self.history_offsets[user:user + 2]
            size = np.searchsorted(self.history_times[start:stop], timestamp, side='left')
            chunks.append(self.history_items[start:start + size])
            sizes.append(size)

        indices = torch.tensor(np.concatenate(chunks))
        counts = torch.tensor(sizes)
        bag_offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        summed = F.embedding_bag(indices, self.y, bag_offsets, mode='sum', include_last_offset=True)
        return summed / counts.sqrt().clamp_min(1)[:, None]

    def user_state(self, users, timestamps):
        dev, bins, day_ids = self.context(users, timestamps)
        factors = (self.p[users] + self.drift_p[users] * dev[:, None]
                   + self.day_p[day_ids] + self.implicit(users, timestamps))
        bias = self.user_bias[users] + self.drift_bias[users] * dev + self.day_bias[day_ids]
        return factors, bias, bins

    def forward(self, users, items, timestamps):
        factors, bias, bins = self.user_state(users, timestamps)
        return (self.mean_rating + bias + self.item_bias[items] + self.item_bin_bias[items, bins]
                + (factors * self.q[items]).sum(dim=1))

    def fit(self, loss, optimizer):
        self.train()
        data = self.data

        for _ in range(self.n_epochs):
            order = np.random.permutation(data.n_interactions)
            total = 0.0

            for start in range(0, data.n_interactions, self.batch_size):
                ids = order[start:start + self.batch_size]
                optimizer.zero_grad()
                predictions = self(data.users[ids], data.items[ids], data.timestamps[ids])
                value = loss(predictions, data.ratings[ids])

                value.backward()
                optimizer.step()

                batch_loss = value.item()
                self.step_loss_history.append(batch_loss)
                total += batch_loss * len(ids)

            self.loss_history.append(total / data.n_interactions)

        return self

    def score(self, users, timestamps=None):
        times = torch.as_tensor(timestamps).reshape(-1).expand(len(users))
        factors, bias, bins = self.user_state(users, times)
        return (self.mean_rating + bias[:, None] + self.item_bias[None, :]
                + self.item_bin_bias[:, bins].T + factors @ self.q.T)
