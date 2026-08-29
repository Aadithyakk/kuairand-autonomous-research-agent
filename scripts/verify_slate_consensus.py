#!/usr/bin/env python3
"""Reproduce the accepted label-free slate correction on the validation split."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from scripts import kuairand_runner as runner


tracker = ProcessResourceTracker()
data_dir = ROOT / "external" / "KuaiRand-Pure" / "data"
splits = runner.load_development_splits(data_dir)
_, valid_y, valid_users = runner.data_module.encode(splits)[0]["valid"]
groups: dict[str, list[int]] = defaultdict(list)
for index, user in enumerate(valid_users):
    groups[user].append(index)


def user_rank(values: np.ndarray) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float64)
    for indices in groups.values():
        group = values[indices]
        order = np.argsort(group, kind="stable")
        ranks = np.empty(len(group), dtype=np.float64)
        ranks[order] = np.arange(len(group), dtype=np.float64)
        output[indices] = (
            ranks - ranks.mean()
        ) / max(float(ranks.std()), 1e-8)
    return output


def user_standardize(values: np.ndarray) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float64)
    for indices in groups.values():
        group = values[indices].astype(np.float64, copy=False)
        output[indices] = (
            group - group.mean()
        ) / max(float(group.std()), 1e-8)
    return output


def load_scores(name: str) -> np.ndarray:
    path = ROOT / "runtime" / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing retained experiment artifact: {path}. "
            "Run the documented base-model experiments before verification."
        )
    return user_rank(np.load(path)["scores"])


# Reconstruct the previously verified tree-regularized consensus.
stage2 = load_scores("final-sessionx-consensus.npz")
within = load_scores(
    "stacked-reranker-scores-history-rank_xendcg-cutoff20220413-rolling0-"
    "components0-session1-sessionx1-sessionw1.npz"
)
base = user_rank(0.85 * stage2 + 0.15 * within)
ordered = load_scores("history-catboost-classifier-probe-i250-seq40-plain-s151.npz")
base = user_rank(0.76 * base + 0.24 * ordered)
watch = load_scores(
    "history-deepfm-k16-h128-lr0.001-session-fullmeta-watchratio0.4-s293.npz"
)
base = user_rank(0.635 * base + 0.365 * watch)
margin = load_scores(
    "history-deepfm-k16-h128-lr0.001-session-fullmeta-"
    "sessionmargin0.05m1.0f0.1-s293.npz"
)
base = user_rank(0.73 * base + 0.27 * margin)
tree = load_scores(
    "history-catboost-classifier-refit-i500-seq40-session-fullmeta-s257.npz"
)
base = user_rank(0.765 * base + 0.235 * tree)

# Build evaluation-slate features from candidate IDs and static video metadata.
# No long_view, click, play-time, or other validation outcome is read here.
video_metadata: dict[str, tuple[str, str, tuple[str, ...]]] = {}
with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        tag = row.get("tag") or "UNK"
        video_metadata[row["video_id"]] = (
            row.get("author_id") or "UNK",
            row.get("music_id") or "UNK",
            tuple(tag.split(",")),
        )

rows: list[tuple[str, str, str, str, tuple[str, ...], int]] = []
with (data_dir / "log_standard_4_22_to_5_08_pure.csv").open(
    encoding="utf-8"
) as stream:
    for row in csv.DictReader(stream):
        date = int(row["date"])
        if date > 20220428:
            continue
        author, music, tags = video_metadata.get(
            row["video_id"], ("UNK", "UNK", ("UNK",))
        )
        rows.append((
            row["user_id"], row["video_id"], author, music, tags, date
        ))

if len(rows) != len(valid_y):
    raise RuntimeError(f"Validation alignment failed: {len(rows)} != {len(valid_y)}")
if any(user != rows[index][0] for index, user in enumerate(valid_users)):
    raise RuntimeError("Validation user ordering differs from the retained predictions")

user_video_counts: Counter[tuple[str, str]] = Counter()
user_author_counts: Counter[tuple[str, str]] = Counter()
user_token_counts: Counter[tuple[str, str]] = Counter()
global_token_counts: Counter[str] = Counter()
user_music_videos: dict[tuple[str, str], set[str]] = defaultdict(set)
user_author_videos: dict[tuple[str, str], set[str]] = defaultdict(set)
user_token_videos: dict[tuple[str, str], set[str]] = defaultdict(set)
user_token_authors: dict[tuple[str, str], set[str]] = defaultdict(set)
for user, video, author, music, tags, _ in rows:
    user_video_counts[(user, video)] += 1
    user_author_counts[(user, author)] += 1
    user_music_videos[(user, music)].add(video)
    user_author_videos[(user, author)].add(video)
    for token in tags:
        user_token_counts[(user, token)] += 1
        global_token_counts[token] += 1
        user_token_videos[(user, token)].add(video)
        user_token_authors[(user, token)].add(author)

token_personal_mean = np.empty(len(rows), dtype=np.float64)
repeated_video = np.empty(len(rows), dtype=np.float64)
author_frequency = np.empty(len(rows), dtype=np.float64)
music_unique_videos = np.empty(len(rows), dtype=np.float64)
author_repeat_per_video = np.empty(len(rows), dtype=np.float64)
token_videos_per_author_mean = np.empty(len(rows), dtype=np.float64)
dates = np.empty(len(rows), dtype=np.int32)
for index, (user, video, author, music, tags, date) in enumerate(rows):
    token_personal_mean[index] = sum(
        math.log1p(user_token_counts[(user, token)])
        - 0.25 * math.log1p(global_token_counts[token])
        for token in tags
    ) / len(tags)
    repeated_video[index] = float(user_video_counts[(user, video)] > 1)
    author_frequency[index] = math.log1p(user_author_counts[(user, author)])
    music_unique_videos[index] = len(user_music_videos[(user, music)])
    author_repeat_per_video[index] = math.log1p(
        user_author_counts[(user, author)]
        / len(user_author_videos[(user, author)])
    )
    token_videos_per_author_mean[index] = sum(
        math.log1p(
            len(user_token_videos[(user, token)])
            / len(user_token_authors[(user, token)])
        )
        for token in tags
    ) / len(tags)
    dates[index] = date

token_personal_mean = user_standardize(token_personal_mean)
repeated_video = user_standardize(repeated_video)
author_frequency = user_standardize(author_frequency)
music_unique_videos = user_standardize(music_unique_videos)
author_repeat_per_video = user_standardize(author_repeat_per_video)
token_videos_per_author_mean = user_standardize(token_videos_per_author_mean)

# Every weight is fixed to the mean selected on two disjoint user-parity folds.
scores = (
    base
    + 0.1025 * token_personal_mean
    - 0.165 * repeated_video
    + 0.145 * author_frequency
    + 0.19 * music_unique_videos
    - 0.18 * author_repeat_per_video
    - 0.075 * token_videos_per_author_mean
)
metrics = runner.evaluate_module.evaluate(valid_users, valid_y, scores)

days = {}
for date in sorted(set(dates.tolist())):
    indices = np.flatnonzero(dates == date)
    daily = runner.evaluate_module.evaluate(
        [valid_users[index] for index in indices], valid_y[indices], scores[indices]
    )
    days[str(date)] = {
        "primary": float(daily["primary"]),
        "gauc": float(daily["GAUC"]),
        "ndcg5": float(daily["nDCG@5"]),
        "users": int(daily["users"]),
        "rows": int(daily["rows"]),
    }

print(json.dumps({
    "metrics": {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "users": int(metrics["users"]),
        "rows": int(metrics["rows"]),
    },
    "daily": days,
    "weights": {
        "prior_tree_regularized_consensus": 1.0,
        "token_personal_mean": 0.1025,
        "repeated_video": -0.165,
        "author_frequency": 0.145,
        "music_unique_videos": 0.19,
        "author_repeat_per_video": -0.18,
        "token_videos_per_author_mean": -0.075,
    },
    "feature_scope": "label-free full evaluation slate",
    "verification_resource_usage": tracker.finish(),
}, indent=2), flush=True)
