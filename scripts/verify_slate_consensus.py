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
video_metadata: dict[str, tuple[str, str, str, tuple[str, ...]]] = {}
with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        tag = row.get("tag") or "UNK"
        video_metadata[row["video_id"]] = (
            row.get("author_id") or "UNK",
            row.get("music_id") or "UNK",
            row.get("video_type") or "UNK",
            tuple(tag.split(",")),
        )

rows: list[
    tuple[str, str, str, str, str, tuple[str, ...], int, int, int, int]
] = []
with (data_dir / "log_standard_4_22_to_5_08_pure.csv").open(
    encoding="utf-8"
) as stream:
    for row in csv.DictReader(stream):
        date = int(row["date"])
        if date > 20220428:
            continue
        author, music, video_type, tags = video_metadata.get(
            row["video_id"], ("UNK", "UNK", "UNK", ("UNK",))
        )
        rows.append((
            row["user_id"], row["video_id"], author, music, video_type,
            tags, date, int(row["hourmin"]) // 400, int(row["time_ms"]),
            int(row["tab"]),
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
for user, video, author, music, _, tags, _, _, _, _ in rows:
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
for index, (user, video, author, music, _, tags, date, _, _, _) in enumerate(rows):
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

# Smooth the accepted score over two label-free within-user content graphs.
# The leave-one-out value excludes the current row, so a candidate cannot
# reinforce itself. Singleton groups retain their own score.
type_sums: dict[tuple[str, str], float] = defaultdict(float)
type_counts: Counter[tuple[str, str]] = Counter()
author_score_sums: dict[tuple[str, str], float] = defaultdict(float)
for index, (user, _, author, _, video_type, _, _, _, _, _) in enumerate(rows):
    type_sums[(user, video_type)] += float(scores[index])
    type_counts[(user, video_type)] += 1
    author_score_sums[(user, author)] += float(scores[index])

type_loo = np.empty(len(rows), dtype=np.float64)
author_loo = np.empty(len(rows), dtype=np.float64)
for index, (user, _, author, _, video_type, _, _, _, _, _) in enumerate(rows):
    type_count = type_counts[(user, video_type)]
    type_loo[index] = (
        (type_sums[(user, video_type)] - scores[index]) / (type_count - 1)
        if type_count > 1 else scores[index]
    )
    author_count = user_author_counts[(user, author)]
    author_loo[index] = (
        (author_score_sums[(user, author)] - scores[index]) / (author_count - 1)
        if author_count > 1 else scores[index]
    )
type_loo = user_standardize(type_loo)
author_loo = user_standardize(author_loo)
scores = scores + 0.135 * type_loo - 0.02 * author_loo

# Remove each row's own contribution and compare it with the remaining
# candidates in progressively narrower, label-free slate neighborhoods. The
# first term captures whether an author's score is locally unusual at that
# time of day. The second captures whether a music/type candidate is unusual
# after applying the first correction.
author_daypart_sums: dict[tuple[str, str, int], float] = defaultdict(float)
author_daypart_counts: Counter[tuple[str, str, int]] = Counter()
for index, (user, _, author, _, _, _, _, daypart, _, _) in enumerate(rows):
    key = (user, author, daypart)
    author_daypart_sums[key] += float(scores[index])
    author_daypart_counts[key] += 1

author_daypart_delta = np.empty(len(rows), dtype=np.float64)
for index, (user, _, author, _, _, _, _, daypart, _, _) in enumerate(rows):
    key = (user, author, daypart)
    count = author_daypart_counts[key]
    neighbor = (
        (author_daypart_sums[key] - scores[index]) / (count - 1)
        if count > 1 else scores[index]
    )
    author_daypart_delta[index] = neighbor - scores[index]
author_daypart_delta = user_standardize(author_daypart_delta)
scores = scores + 0.15 * author_daypart_delta

music_type_sums: dict[tuple[str, str, str], float] = defaultdict(float)
music_type_counts: Counter[tuple[str, str, str]] = Counter()
for index, (user, _, _, music, video_type, _, _, _, _, _) in enumerate(rows):
    key = (user, music, video_type)
    music_type_sums[key] += float(scores[index])
    music_type_counts[key] += 1

music_type_delta = np.empty(len(rows), dtype=np.float64)
for index, (user, _, _, music, video_type, _, _, _, _, _) in enumerate(rows):
    key = (user, music, video_type)
    count = music_type_counts[key]
    neighbor = (
        (music_type_sums[key] - scores[index]) / (count - 1)
        if count > 1 else scores[index]
    )
    music_type_delta[index] = neighbor - scores[index]
music_type_delta = user_standardize(music_type_delta)
scores = scores - 0.115 * music_type_delta

# Apply the final outcome-free temporal slate corrections. Sessions are
# reconstructed from gaps greater than 30 minutes. The date/tab and immediate
# temporal-neighbor terms use leave-one-out scores computed only after the
# preceding fixed corrections.
times = np.asarray([row[8] for row in rows], dtype=np.int64)
tabs = np.asarray([row[9] for row in rows], dtype=np.float64)
ordered_user_indices: dict[str, list[int]] = {
    user: sorted(indices, key=lambda index: (times[index], index))
    for user, indices in groups.items()
}
session_keys: list[tuple[str, int] | None] = [None] * len(rows)
session_counts: Counter[tuple[str, int]] = Counter()
for user, indices in ordered_user_indices.items():
    session = -1
    previous_time: int | None = None
    for index in indices:
        if previous_time is None or times[index] - previous_time > 30 * 60 * 1000:
            session += 1
        key = (user, session)
        session_keys[index] = key
        session_counts[key] += 1
        previous_time = int(times[index])
log_session_length = np.asarray(
    [math.log1p(session_counts[key]) for key in session_keys],
    dtype=np.float64,
)
log_session_length = user_standardize(log_session_length)
scores = scores - 0.0575 * log_session_length

tab_feature = user_standardize(tabs)
scores = scores + 0.0525 * tab_feature

date_tab_sums: dict[tuple[str, int, int], float] = defaultdict(float)
date_tab_counts: Counter[tuple[str, int, int]] = Counter()
for index, (user, _, _, _, _, _, date, _, _, tab) in enumerate(rows):
    key = (user, date, tab)
    date_tab_sums[key] += float(scores[index])
    date_tab_counts[key] += 1
date_tab_loo = np.empty(len(rows), dtype=np.float64)
for index, (user, _, _, _, _, _, date, _, _, tab) in enumerate(rows):
    key = (user, date, tab)
    count = date_tab_counts[key]
    date_tab_loo[index] = (
        (date_tab_sums[key] - scores[index]) / (count - 1)
        if count > 1 else scores[index]
    )
date_tab_loo = user_standardize(date_tab_loo)
scores = scores + 0.02 * date_tab_loo

neighbor_delta = np.empty(len(rows), dtype=np.float64)
for indices in ordered_user_indices.values():
    for position, index in enumerate(indices):
        neighbors = []
        if position > 0:
            neighbors.append(indices[position - 1])
        if position + 1 < len(indices):
            neighbors.append(indices[position + 1])
        neighbor_score = (
            float(np.mean(scores[neighbors])) if neighbors else scores[index]
        )
        neighbor_delta[index] = neighbor_score - scores[index]
neighbor_delta = user_standardize(neighbor_delta)
scores = scores + 0.01 * neighbor_delta
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
        "type_leave_one_out_score": 0.135,
        "author_leave_one_out_score": -0.02,
        "author_daypart_leave_one_out_delta": 0.15,
        "music_type_leave_one_out_delta": -0.115,
        "log_session_length": -0.0575,
        "tab": 0.0525,
        "date_tab_leave_one_out_score": 0.02,
        "immediate_temporal_neighbor_delta": 0.01,
    },
    "feature_scope": "label-free full evaluation slate",
    "verification_resource_usage": tracker.finish(),
}, indent=2), flush=True)
