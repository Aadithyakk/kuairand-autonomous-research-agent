#!/usr/bin/env python3
"""Reproduce the accepted training-only user-neighbor CF correction.

The frozen configuration was selected from an April 19-21 temporal holdout and
then required to improve four disjoint validation-user folds. Validation labels
are read only after predictions are complete so they can be reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from scripts import kuairand_runner as runner


NEIGHBOR_COUNT = 60
SIMILARITY_POWER = 2.0
NEIGHBOR_SMOOTHING = 8.0
ITEM_SMOOTHING = 20.0

tracker = ProcessResourceTracker()
data_dir = ROOT / "external" / "KuaiRand-Pure" / "data"
feature_columns = ["user_id", "video_id", "date"]
train = pl.read_csv(
    data_dir / "log_standard_4_08_to_4_21_pure.csv",
    columns=[*feature_columns, "long_view"],
).with_columns(pl.lit(0).cast(pl.Int8).alias("batch"))
valid = (
    pl.read_csv(
        data_dir / "log_standard_4_22_to_5_08_pure.csv", columns=feature_columns,
    )
    .filter(pl.col("date") <= 20220428)
    .with_columns([
        pl.lit(0).cast(pl.Int64).alias("long_view"),
        pl.lit(1).cast(pl.Int8).alias("batch"),
    ])
)
frame = pl.concat([train, valid], how="vertical")

user_values = frame["user_id"].cast(pl.String).to_numpy()
item_values = frame["video_id"].cast(pl.String).to_numpy()
_, user_codes = np.unique(user_values, return_inverse=True)
_, item_codes = np.unique(item_values, return_inverse=True)
user_codes = user_codes.astype(np.int32, copy=False)
item_codes = item_codes.astype(np.int32, copy=False)
labels = frame["long_view"].to_numpy().astype(np.float32, copy=False)
batches = frame["batch"].to_numpy().astype(np.int8, copy=False)
user_count = int(user_codes.max()) + 1
item_count = int(item_codes.max()) + 1
train_indices = np.flatnonzero(batches == 0)
valid_indices = np.flatnonzero(batches == 1)

train_users = user_codes[train_indices]
train_items = item_codes[train_indices]
train_labels = labels[train_indices]
exposure = sparse.coo_matrix(
    (
        np.ones(len(train_indices), dtype=np.float32),
        (train_users, train_items),
    ),
    shape=(user_count, item_count),
).tocsr()
positive = sparse.coo_matrix(
    (train_labels, (train_users, train_items)),
    shape=(user_count, item_count),
).tocsr()

# Similarity uses only each user's set of exposed training items. Outcomes stay
# confined to ``positive`` and enter only the neighbor rate estimate below. The
# retained artifact key preserves the prototype's historical profile label.
profile = exposure.copy()
profile.data[:] = 1.0
profile.eliminate_zeros()
norm = np.sqrt(np.asarray(profile.multiply(profile).sum(axis=1)).ravel())
profile = (sparse.diags(1.0 / np.maximum(norm, 1e-8)) @ profile).tocsr()

valid_users = user_codes[valid_indices]
unique_valid_users = np.unique(valid_users)
neighbor_model = NearestNeighbors(
    n_neighbors=NEIGHBOR_COUNT + 1,
    metric="cosine",
    algorithm="brute",
    n_jobs=8,
)
neighbor_model.fit(profile)
distances, neighbor_indices = neighbor_model.kneighbors(
    profile[unique_valid_users]
)
neighbor_lookup = np.full(user_count, -1, dtype=np.int32)
neighbor_lookup[unique_valid_users] = np.arange(
    len(unique_valid_users), dtype=np.int32
)

neighbors = np.empty(
    (len(unique_valid_users), NEIGHBOR_COUNT), dtype=np.int32
)
weights = np.zeros(
    (len(unique_valid_users), NEIGHBOR_COUNT), dtype=np.float64
)
for row, user in enumerate(unique_valid_users):
    keep = neighbor_indices[row] != user
    selected_neighbors = neighbor_indices[row][keep][:NEIGHBOR_COUNT]
    selected_weights = np.maximum(
        1.0 - distances[row][keep][:NEIGHBOR_COUNT], 0.0
    )
    neighbors[row, :len(selected_neighbors)] = selected_neighbors
    weights[row, :len(selected_weights)] = np.power(
        selected_weights, SIMILARITY_POWER
    )
    if len(selected_neighbors) < NEIGHBOR_COUNT:
        neighbors[row, len(selected_neighbors):] = user

global_prior = float(train_labels.mean())
item_exposure = np.asarray(exposure.sum(axis=0)).ravel().astype(np.float64)
item_positive = np.asarray(positive.sum(axis=0)).ravel().astype(np.float64)
item_prior = (
    item_positive + ITEM_SMOOTHING * global_prior
) / (item_exposure + ITEM_SMOOTHING)

valid_items = item_codes[valid_indices]
scores = np.empty(len(valid_indices), dtype=np.float32)
grouped: dict[int, list[int]] = {}
for row, user in enumerate(valid_users):
    grouped.setdefault(int(user), []).append(row)
for user, raw_rows in grouped.items():
    rows = np.asarray(raw_rows, dtype=np.int64)
    items = valid_items[rows]
    neighbor_row = int(neighbor_lookup[user])
    local_neighbors = neighbors[neighbor_row]
    local_weights = weights[neighbor_row]
    local_exposure = exposure[local_neighbors][:, items].toarray().astype(
        np.float64
    )
    local_positive = positive[local_neighbors][:, items].toarray().astype(
        np.float64
    )
    denominator = local_weights @ local_exposure
    numerator = local_weights @ local_positive
    scores[rows] = (
        (numerator + NEIGHBOR_SMOOTHING * item_prior[items])
        / (denominator + NEIGHBOR_SMOOTHING)
    ).astype(np.float32)

output = ROOT / "runtime" / "user-neighbor-cf-n60.npz"
np.savez_compressed(output, **{"positive_p2.0_s8.0": scores})
metrics = runner.evaluate_module.evaluate(
    user_values[valid_indices].tolist(),
    (
        pl.read_csv(
            data_dir / "log_standard_4_22_to_5_08_pure.csv",
            columns=["date", "long_view"],
        )
        .filter(pl.col("date") <= 20220428)["long_view"]
        .to_numpy()
        .astype(np.float32, copy=False)
    ),
    scores,
)
print("VALID", metrics, flush=True)
print("OUTPUT", output, flush=True)
print("RESOURCE_USAGE", tracker.finish(), flush=True)
