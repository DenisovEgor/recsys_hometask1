import csv
import gzip
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hw1.models import TrainData

SOURCE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/beer/Beeradvocate.txt.gz'
FIELDS = {'review/profileName': 'user_id', 'beer/beerId': 'item_id',
          'review/overall': 'rating', 'review/time': 'timestamp',
          'beer/name': 'name', 'beer/style': 'style',
          'beer/brewerId': 'brewery', 'beer/ABV': 'abv'}


def download_beer(cache='.cache/beeradvocate'):
    path = Path(cache) / 'Beeradvocate.txt.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(SOURCE_URL, path)
    return path


def read_beer(path):
    compact = path.parent / 'reviews-v1.csv.gz'
    if not compact.exists():
        record = {}
        with (
            gzip.open(path, 'rt', encoding='utf-8', errors='replace') as source,
            gzip.open(compact, 'wt', newline='') as target,
        ):
            writer = csv.DictWriter(target, fieldnames=list(FIELDS.values()))
            writer.writeheader()

            for line in source:
                line = line.rstrip('\r\n')
                if not line:
                    if record:
                        writer.writerow(record)
                        record = {}
                else:
                    key, _, value = line.partition(': ')
                    if key in FIELDS:
                        record[FIELDS[key]] = value

            if record:
                writer.writerow(record)

    dtypes = {column: 'string' for column in ['user_id', 'name', 'style', 'brewery']}
    return pd.read_csv(compact, dtype=dtypes, low_memory=False)


def seed_everything(seed=52, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def clean_reviews(raw):
    frame = raw.copy()
    for column in ['item_id', 'rating', 'timestamp', 'abv']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')

    frame['user_id'] = frame['user_id'].str.strip().replace('', pd.NA)
    invalid = (frame[['user_id', 'item_id', 'rating', 'timestamp']].isna().any(axis=1)
               | ~frame['rating'].between(.5, 5)
               | (frame['timestamp'] <= 0) | (frame['item_id'] <= 0))

    frame = frame.loc[~invalid].copy()
    frame['item_id'] = frame['item_id'].astype('int64')
    frame['timestamp'] = frame['timestamp'].astype('int64')
    frame.loc[~frame['abv'].between(0, 100), 'abv'] = np.nan

    frame = frame.sort_values(['timestamp', 'user_id', 'item_id'], kind='stable')
    duplicates = int(frame.duplicated(['user_id', 'item_id']).sum())
    frame = frame.drop_duplicates(['user_id', 'item_id'], keep='first').reset_index(drop=True)

    report = {'raw_rows': len(raw), 'invalid_rows_removed': int(invalid.sum()),
              'duplicate_user_item_rows_removed': duplicates, 'clean_rows': len(frame),
              'clean_users': frame.user_id.nunique(), 'clean_items': frame.item_id.nunique()}
    return frame, report


def prepare_split(clean, max_users=4000, max_items=1200, train_fraction=.70,
                  validation_user_fraction=.5, min_user_train=10,
                  min_user_positives=3, min_item_train=15, threshold=4.0, seed=52):
    cutoff = int(clean.timestamp.quantile(train_fraction))
    pool = clean.loc[clean.timestamp < cutoff]

    counts = pool.groupby('item_id').size().rename('count').reset_index()
    counts = counts.loc[counts['count'] >= min_item_train]
    item_ids = counts.sort_values(['count', 'item_id'], ascending=[False, True]).head(max_items).item_id

    restricted = pool.loc[pool.item_id.isin(item_ids)].copy()
    restricted['positive'] = restricted.rating >= threshold
    counts = (restricted.groupby('user_id')
              .agg(count=('rating', 'size'), positives=('positive', 'sum')).reset_index())
    counts = counts.loc[(counts['count'] >= min_user_train) & (counts.positives >= min_user_positives)]
    user_ids = counts.sort_values(['count', 'user_id'], ascending=[False, True]).head(max_users).user_id

    train = restricted.loc[restricted.user_id.isin(user_ids)]
    user_map = {value: index for index, value in enumerate(sorted(train.user_id.unique()))}
    item_map = {value: index for index, value in enumerate(sorted(train.item_id.unique()))}

    sample = clean.loc[clean.user_id.isin(user_map) & clean.item_id.isin(item_map)].copy()
    sample['user_idx'] = sample.user_id.map(user_map).astype('int64')
    sample['item_idx'] = sample.item_id.map(item_map).astype('int64')

    users = np.random.default_rng(seed).permutation(len(user_map))
    n_validation = int(len(users) * validation_user_fraction)
    validation_users = np.sort(users[:n_validation])
    test_users = np.sort(users[n_validation:])

    train = sample.loc[sample.timestamp < cutoff].copy()
    future = sample.loc[sample.timestamp >= cutoff]
    validation = future.loc[future.user_idx.isin(validation_users)].copy()
    test = future.loc[future.user_idx.isin(test_users)].copy()

    metadata = (train.sort_values(['timestamp', 'item_idx', 'user_idx'])
                .drop_duplicates('item_idx').set_index('item_idx')
                .reindex(range(len(item_map))))

    summary = {'n_reviews': len(sample), 'n_users': len(user_map), 'n_items': len(item_map),
               'train_reviews': len(train), 'validation_reviews': len(validation),
               'test_reviews': len(test), 'validation_start': cutoff, 'test_start': cutoff,
               'validation_user_ids': validation_users.tolist(), 'test_user_ids': test_users.tolist(),
               'density': len(sample) / (len(user_map) * len(item_map)),
               'source_users_retained': len(user_map) / clean.user_id.nunique(),
               'source_items_retained': len(item_map) / clean.item_id.nunique()}
    return sample, train, validation, test, metadata, summary


def encode_metadata(metadata):
    styles = metadata['style'].fillna('<unknown>').astype(str)
    breweries = metadata['brewery'].fillna('<unknown>').astype(str)
    style_vocab = {value: index for index, value in enumerate(sorted(styles.unique()))}
    brewery_vocab = {value: index + len(style_vocab) for index, value in enumerate(sorted(breweries.unique()))}
    numeric_id = len(style_vocab) + len(brewery_vocab)

    abv = pd.to_numeric(metadata.abv, errors='coerce')
    median = float(abv.median())
    abv = abv.fillna(median)
    mean, std = float(abv.mean()), float(abv.std(ddof=0))

    indices = np.column_stack([styles.map(style_vocab), breweries.map(brewery_vocab),
                               np.full(len(metadata), numeric_id)]).astype(np.int64)
    values = np.column_stack([np.ones(len(metadata)), np.ones(len(metadata)),
                              ((abv - mean) / std).clip(-5, 5)]).astype(np.float32)

    description = {'style_categories': len(style_vocab), 'brewery_categories': len(brewery_vocab),
                   'abv_imputation_median': median, 'abv_mean': mean, 'abv_std': std,
                   'abv_missing_fraction': float(metadata.abv.isna().mean()),
                   'feature_names': ['style_one_hot', 'brewery_one_hot', 'standardized_ABV']}
    return indices, values, numeric_id + 1, description


def make_train_data(train, metadata, threshold=4.0):
    indices, values, n_features, description = encode_metadata(metadata)
    data = TrainData(n_users=int(train.user_idx.max()) + 1, n_items=len(metadata),
                     users=train.user_idx.to_numpy(), items=train.item_idx.to_numpy(),
                     ratings=train.rating.to_numpy(), timestamps=train.timestamp.to_numpy(),
                     item_feature_indices=indices, item_feature_values=values,
                     n_item_features=n_features, relevance_threshold=threshold)
    return data, description


def relevance_sets(frame, users, threshold=4.0):
    relevant = frame.loc[frame.rating >= threshold]
    groups = relevant.groupby('user_idx').item_idx.agg(set).to_dict()
    return {int(user): groups.get(int(user), set()) for user in users}


def ranking_metrics(recommendations, truth, users, k=10):
    discounts = 1 / np.log2(np.arange(2, k + 2))
    rows = []

    for user, predicted in zip(users, recommendations):
        relevant = truth[user]
        hits = np.array([item in relevant for item in predicted[:k]], dtype=float)
        hit_count = hits.sum()

        dcg = (hits * discounts[:len(hits)]).sum()
        ideal = discounts[:min(k, len(relevant))].sum()

        rows.append({'NDCG@10': dcg / ideal if ideal else 0.0,
                     'Recall@10': hit_count / len(relevant) if relevant else 0.0})

    return pd.DataFrame(rows).mean().to_dict()


def history_mask(data, extra_frame=None):
    mask = data.seen.clone()
    if extra_frame is not None:
        users = torch.tensor(extra_frame.user_idx.to_numpy(copy=True), dtype=torch.long)
        items = torch.tensor(extra_frame.item_idx.to_numpy(copy=True), dtype=torch.long)
        mask[users, items] = True
    return mask


def evaluate(model, truth, users, timestamp, seen_mask):
    recommendations = model.recommend(users, timestamps=timestamp, seen_mask=seen_mask)
    return ranking_metrics(recommendations, truth, users)


def tune_model(name, model_class, grid, data, truth, users, timestamp, seen_mask,
               loss=None, optimizer=None, seed=52):
    records, best_model, best_record = [], None, None

    for trial, parameters in enumerate(grid, 1):
        seed_everything(seed)
        model_parameters = parameters.copy()
        optimizer_parameters = model_parameters.pop('optimizer', {})
        model = model_class(data, **model_parameters)

        fit_kwargs = {}
        if optimizer is not None:
            fit_kwargs = dict(loss=loss, optimizer=optimizer(model.parameters(), **optimizer_parameters))

        started = time.perf_counter()
        model.fit(**fit_kwargs)
        seconds = time.perf_counter() - started

        metrics = evaluate(model, truth, users, timestamp, seen_mask)
        record = dict(model=name, trial=trial, parameters=parameters, fit_seconds=seconds, **metrics)
        records.append(record)

        key = metrics['NDCG@10'], metrics['Recall@10']
        if best_record is None or key > (best_record['NDCG@10'], best_record['Recall@10']):
            best_model, best_record = model, record

        print(f"{name} {trial}/{len(grid)}: Validation NDCG@10={metrics['NDCG@10']:.3f}, "
              f"Recall@10={metrics['Recall@10']:.3f}", flush=True)

    return best_model, best_record, pd.DataFrame(records)
