#!/usr/bin/env python3
"""Reproduce the accepted label-free slate correction on the validation split."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from scripts import kuairand_runner as runner


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scores-output",
    type=Path,
    help="Optional .npz path for the verified validation scores.",
)
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])


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


def normalized_user_top_margin(values: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float64)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        if len(indices) < 2:
            continue
        order = indices[np.argsort(-values[indices], kind="stable")]
        output[indices] = (
            values[order[0]] - values[order[1]]
        ) / max(float(values[indices].std()), 1e-8)
    return output


def load_scores(name: str) -> np.ndarray:
    path = ROOT / "runtime" / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing retained experiment artifact: {path}. "
            "Run the documented base-model experiments before verification."
        )
    return user_rank(np.load(path)["scores"])


def load_named_scores(name: str, key: str) -> np.ndarray:
    path = ROOT / "runtime" / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing retained experiment artifact: {path}. "
            "Run the documented base-model experiments before verification."
        )
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"Missing {key!r} in retained artifact: {path}")
        return user_rank(np.asarray(archive[key]).reshape(-1))


def load_fractional_ranks(name: str) -> np.ndarray:
    path = ROOT / "runtime" / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing retained experiment artifact: {path}. "
            "Run the documented base-model experiments before verification."
        )
    values = np.load(path)["scores"]
    output = np.empty(len(values), dtype=np.float64)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        group = values[indices]
        order = np.argsort(group, kind="stable")
        ranks = np.empty(len(group), dtype=np.float64)
        ranks[order] = np.arange(len(group), dtype=np.float64)
        output[indices] = ranks / max(len(group) - 1, 1)
    return output


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
skip_batch_slate = os.getenv("KUAI_SKIP_BATCH_SLATE") == "1"
skip_user_neighbor = os.getenv("KUAI_SKIP_USER_NEIGHBOR") == "1"
skip_user_balanced_tree = os.getenv("KUAI_SKIP_USER_BALANCED_TREE") == "1"
skip_yeti_session_gate = os.getenv("KUAI_SKIP_YETI_SESSION_GATE") == "1"
skip_session_median_gate = os.getenv("KUAI_SKIP_SESSION_MEDIAN_GATE") == "1"
skip_recent_yeti_gate = os.getenv("KUAI_SKIP_RECENT_YETI_GATE") == "1"
batch_slate_tree = (
    None if skip_batch_slate
    else load_scores("batch-slate-meta-catboost-s751.npz")
)

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
threshold_duration_buckets: list[int] = []
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
        duration = float(row["duration_ms"])
        threshold_duration_buckets.append(
            0 if duration < 10_000 else 1 if duration < 18_000 else 2
        )

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

# Add three Bayesian-smoothed cross-conditional priors computed strictly from
# the April 8-21 training block. Recent-window priors intentionally use only
# their named suffix of the training period; no validation outcome is read.
recent3_long_sum: Counter[tuple[str, int]] = Counter()
recent3_long_count: Counter[tuple[str, int]] = Counter()
recent7_tab_sum: Counter[int] = Counter()
recent7_tab_count: Counter[int] = Counter()
user_author_completion_sum: Counter[tuple[str, str]] = Counter()
user_author_completion_count: Counter[tuple[str, str]] = Counter()
user_long_sum: Counter[str] = Counter()
user_long_count: Counter[str] = Counter()
recent_user_count: Counter[str] = Counter()
validation_user_video_pairs = {
    (user, video) for user, video, _, _, _, _, _, _, _, _ in rows
}
users_with_seen_validation_video: set[str] = set()
recent3_total = 0
recent3_positive = 0
recent7_total = 0
recent7_positive = 0
completion_total = 0
completion_positive = 0
all_long_total = 0
all_long_positive = 0
with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
    encoding="utf-8"
) as stream:
    for row in csv.DictReader(stream):
        date = int(row["date"])
        user = row["user_id"]
        author = video_metadata.get(
            row["video_id"], ("UNK", "UNK", "UNK", ("UNK",))
        )[0]
        duration = float(row["duration_ms"])
        threshold_bucket = (
            0 if duration < 10_000 else 1 if duration < 18_000 else 2
        )
        long_view = int(row["long_view"] != "0")
        completion = int(float(row["play_time_ms"]) >= 0.95 * duration)
        user_long_sum[user] += long_view
        user_long_count[user] += 1
        if (user, row["video_id"]) in validation_user_video_pairs:
            users_with_seen_validation_video.add(user)
        all_long_total += 1
        all_long_positive += long_view
        if date >= 20220419:
            key = (user, threshold_bucket)
            recent3_long_sum[key] += long_view
            recent3_long_count[key] += 1
            recent3_total += 1
            recent3_positive += long_view
            recent_user_count[user] += 1
        if date >= 20220415:
            tab = int(row["tab"])
            recent7_tab_sum[tab] += long_view
            recent7_tab_count[tab] += 1
            recent7_total += 1
            recent7_positive += long_view
        author_key = (user, author)
        user_author_completion_sum[author_key] += completion
        user_author_completion_count[author_key] += 1
        completion_total += 1
        completion_positive += completion

recent3_prior = recent3_positive / recent3_total
recent7_prior = recent7_positive / recent7_total
completion_prior = completion_positive / completion_total
all_long_prior = all_long_positive / all_long_total
recent3_user_threshold_rate = np.empty(len(rows), dtype=np.float64)
recent7_tab_rate = np.empty(len(rows), dtype=np.float64)
user_author_completion_rate = np.empty(len(rows), dtype=np.float64)
training_user_long_rate = np.empty(len(rows), dtype=np.float64)
training_recent_rows = np.empty(len(rows), dtype=np.float64)
for index, (user, _, author, _, _, _, _, _, _, tab) in enumerate(rows):
    threshold_key = (user, threshold_duration_buckets[index])
    threshold_count = recent3_long_count[threshold_key]
    recent3_user_threshold_rate[index] = (
        recent3_long_sum[threshold_key] + 8.0 * recent3_prior
    ) / (threshold_count + 8.0)
    tab_count = recent7_tab_count[tab]
    recent7_tab_rate[index] = (
        recent7_tab_sum[tab] + 30.0 * recent7_prior
    ) / (tab_count + 30.0)
    author_key = (user, author)
    author_count = user_author_completion_count[author_key]
    user_author_completion_rate[index] = (
        user_author_completion_sum[author_key] + 6.0 * completion_prior
    ) / (author_count + 6.0)
    user_count = user_long_count[user]
    training_user_long_rate[index] = (
        user_long_sum[user] + 10.0 * all_long_prior
    ) / (user_count + 10.0)
    training_recent_rows[index] = recent_user_count[user]

recent3_user_threshold_rate = user_standardize(recent3_user_threshold_rate)
recent7_tab_rate = user_standardize(recent7_tab_rate)
user_author_completion_rate = user_standardize(user_author_completion_rate)
scores = (
    scores
    - 0.04 * recent3_user_threshold_rate
    + 0.01 * recent7_tab_rate
    + 0.03 * user_author_completion_rate
)

# A temporally trained batch-slate CatBoost sees the same outcome-free totals,
# score neighborhoods, and static metadata on April 14-21 before being refit
# and applied to April 22-28. Its standalone prediction is deliberately used
# only as a tiny correction. A continuous repeat-count penalty and this tree
# form a jointly corrected ordering; an outer ordinal consensus amplifies only
# the few pairwise changes on which that ordering differs from the champion.
if batch_slate_tree is not None:
    log_user_video_count = user_standardize(np.asarray([
        math.log1p(user_video_counts[(user, video)])
        for user, video, _, _, _, _, _, _, _, _ in rows
    ], dtype=np.float64))
    joint_slate_score = (
        scores - 0.05 * log_user_video_count + 0.0021875 * batch_slate_tree
    )
    scores = scores + 0.2675 * user_rank(joint_slate_score)

if batch_slate_tree is not None:
    # The frozen champion is least reliable when its first-place score is
    # unusually far from the runner-up. For only the highest-margin quartile,
    # move the ordering partway toward the temporally ordered CatBoost. The gate
    # uses only frozen scores and is normalized within user; no validation
    # outcome is read. The quantile is row-weighted to reproduce selection.
    normalized_top_margin = normalized_user_top_margin(scores)
    eligible_top_margins = normalized_top_margin[normalized_top_margin > 0]
    high_margin_threshold = float(np.quantile(eligible_top_margins, 0.75))
    high_margin_gate = (normalized_top_margin >= high_margin_threshold).astype(
        np.float64
    )
    scores = scores + 0.34 * high_margin_gate * (ordered - user_rank(scores))

    # A much smaller second correction uses broad model agreement only where
    # the already gated champion remains above its median normalized top
    # margin. Each source contributes only its within-user fractional rank, so
    # calibration scale and validation outcomes are absent from this feature.
    confidence_source_names = [
        "final-sessionx-consensus.npz",
        "stacked-reranker-scores-history-rank_xendcg-cutoff20220413-rolling0-"
        "components0-session1-sessionx1-sessionw1.npz",
        "history-catboost-classifier-probe-i250-seq40-plain-s151.npz",
        "history-deepfm-k16-h128-lr0.001-session-fullmeta-watchratio0.4-s293.npz",
        "history-deepfm-k16-h128-lr0.001-session-fullmeta-"
        "sessionmargin0.05m1.0f0.1-s293.npz",
        "history-catboost-classifier-refit-i500-seq40-session-fullmeta-s257.npz",
        "batch-slate-meta-catboost-s751.npz",
        "history-deepfm-k16-h128-lr0.001-session-fullmeta-s293.npz",
        "history-deepfm-k16-h128-lr0.001-session-fullmeta-aux0.15-s337.npz",
        "multitask-a0.2-k16-h128-lr0.001-s71.npz",
        "din-h20-k16-hidden128-lr0.001-s0.npz",
        "cross-session-gru-e8-h64-head128-s431.npz",
        "history-catboost-ranker-probe-i220-seq40-session-s157.npz",
        "lgbm-lambdarank-r500-l31-lr0.03-leaf100-s0.npz",
        "lightgcn-k32-l2-lr0.01-s0.npz",
        "ffm-base-star-k16-lr0.001-s401.npz",
        "ordinal-watch-a0.3-k16-h128-lr0.001-s121.npz",
        "threshold-deepfm-a0.3-k16-h128-lr0.001-s131.npz",
        "stacked-catboost-classifier-scores-extended-cutoff20220413-session1-"
        "sessionx1-sessionw0-multir0-trend0-affinity0-ctx0.npz",
        "stacked-linear-scores-history-cutoff20220413-rolling0-components0-"
        "session1-sessionx1-sessionw0-multir0-trend0-affinity0-ctx0-"
        "transition1.npz",
    ]
    confidence_fraction_rows = [
        load_fractional_ranks(name) for name in confidence_source_names
    ]
    mean_model_fraction = np.mean(confidence_fraction_rows, axis=0)
    median_model_fraction = np.median(confidence_fraction_rows, axis=0)
    normalized_top_margin = normalized_user_top_margin(scores)
    eligible_top_margins = normalized_top_margin[normalized_top_margin > 0]
    median_margin_threshold = float(np.quantile(eligible_top_margins, 0.50))
    median_margin_gate = (
        normalized_top_margin >= median_margin_threshold
    ).astype(np.float64)
    scores = scores + 0.01 * median_margin_gate * (
        user_rank(mean_model_fraction) - user_rank(scores)
    )

    # Users active on exactly two supplied validation dates form a distinct
    # low-observation regime. Four disjoint user folds consistently selected an
    # extrapolation away from the ordered model for this regime. Active-day
    # count and both ranks are outcome-free.
    active_day_count = np.zeros(len(scores), dtype=np.int16)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        active_day_count[indices] = len(np.unique(dates[indices]))
    two_day_gate = (active_day_count == 2).astype(np.float64)
    scores = scores - 0.5225 * two_day_gate * (
        ordered - user_rank(scores)
    )

    three_four_day_gate = (
        (active_day_count >= 3) & (active_day_count <= 4)
    ).astype(np.float64)
    scores = scores - 0.175 * three_four_day_gate * (
        watch - user_rank(scores)
    )

    group_size = np.zeros(len(scores), dtype=np.int16)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        group_size[indices] = len(indices)
    medium_slate_gate = ((group_size >= 6) & (group_size <= 10)).astype(
        np.float64
    )
    scores = scores + 0.01 * medium_slate_gate * (
        batch_slate_tree - user_rank(scores)
    )

    low_rate_threshold = float(np.quantile(training_user_long_rate, 0.25))
    low_training_rate_gate = (
        training_user_long_rate <= low_rate_threshold
    ).astype(np.float64)
    scores = scores + 0.135 * low_training_rate_gate * (
        watch - user_rank(scores)
    )

    prior_video_gate = np.asarray([
        float(user in users_with_seen_validation_video)
        for user in valid_users
    ], dtype=np.float64)
    scores = scores - 0.03 * prior_video_gate * (
        batch_slate_tree - user_rank(scores)
    )

    # A sparse user-neighbor model is fit only on April 8-21 interactions. Its
    # exposure-history cosine graph predicts each evaluation item from other
    # users' training outcomes, with no evaluation outcome in either the graph
    # or the target estimate. All four disjoint user folds selected the same
    # tiny correction weight.
    if not skip_user_neighbor:
        user_neighbor = load_named_scores(
            "user-neighbor-cf-n60.npz", "positive_p2.0_s8.0"
        )
        scores = scores + 0.002 * user_neighbor

    # The user-balanced tree and frozen watch-ratio DeepFM contain complementary
    # near-tie information even though neither passes the final residual gate
    # alone. A corrected two-dimensional audit over actual user-ID-modulo folds
    # selected this exact pair in all four folds, with every held-out fold
    # improving. Both models were trained before the evaluation interval.
    if not skip_user_balanced_tree:
        user_balanced_tree = load_scores(
            "history-catboost-classifier-probe-i500-seq40-session-"
            "ub0.5-s239.npz"
        )
        scores = (
            scores
            + 0.001875 * user_balanced_tree
            + 0.001875 * watch
        )

    # YetiRank's global ordering is weaker, but all four disjoint user folds
    # selected the same positive direction for users whose full supplied slate
    # forms one 30-minute session. The fixed mean scalar changes one ordering in
    # one user; one fold improves and the other three remain exactly unchanged.
    if not skip_yeti_session_gate:
        yeti_rank = load_scores(
            "batch-slate-meta-catboost-ranker-YetiRankPairwise-s787.npz"
        )
        single_session_gate = np.zeros(len(scores), dtype=np.float64)
        for indices_list in groups.values():
            indices = np.asarray(indices_list, dtype=np.int64)
            session_count = len({session_keys[index] for index in indices})
            single_session_gate[indices] = float(session_count == 1)
        scores = scores + 0.685 * single_session_gate * (
            yeti_rank - user_rank(scores)
        )

    # For users whose supplied slate spans two or three reconstructed sessions,
    # all four user folds selected a small extrapolation away from the median
    # rank of 20 frozen model families. The gate and source ranks are label-free.
    if not skip_session_median_gate:
        two_three_session_gate = np.zeros(len(scores), dtype=np.float64)
        for indices_list in groups.values():
            indices = np.asarray(indices_list, dtype=np.int64)
            session_count = len({session_keys[index] for index in indices})
            two_three_session_gate[indices] = float(2 <= session_count <= 3)
        scores = scores - 0.0275 * two_three_session_gate * (
            user_rank(median_model_fraction) - user_rank(scores)
        )

    # A shallower, more regularized YetiRank model is globally weaker, but its
    # disagreement identifies overconfident orderings for users with dense
    # April 19-21 histories. All four user folds selected extrapolation away
    # from that model. The coverage threshold and prediction are training-only.
    if not skip_recent_yeti_gate:
        shallow_yeti_rank = load_scores(
            "batch-slate-meta-catboost-ranker-YetiRankPairwise-s821.npz"
        )
        recent_high_threshold = float(np.quantile(training_recent_rows, 0.75))
        recent_high_gate = (
            training_recent_rows >= recent_high_threshold
        ).astype(np.float64)
        scores = scores - 0.0925 * recent_high_gate * (
            shallow_yeti_rank - user_rank(scores)
        )
metrics = runner.evaluate_module.evaluate(valid_users, valid_y, scores)
if args.scores_output:
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.scores_output, scores=scores.astype(np.float32))

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
        "recent3_user_threshold_duration_long_rate": -0.04,
        "recent7_tab_long_rate": 0.01,
        "all_user_author_completion_rate": 0.03,
        **({
            "joint_log_user_video_count": -0.05,
            "joint_batch_slate_tree": 0.0021875,
            "joint_ordinal_consensus": 0.2675,
            "high_margin_quantile": 0.75,
            "high_margin_ordered_consensus": 0.34,
            "median_margin_quantile": 0.5,
            "median_margin_model_mean_consensus": 0.01,
            "two_active_days_ordered_extrapolation": -0.5225,
            "three_four_active_days_watch_extrapolation": -0.175,
            "medium_slate_batch_consensus": 0.01,
            "low_training_rate_watch_consensus": 0.135,
            "prior_video_batch_extrapolation": -0.03,
            **({
                "training_user_neighbor_positive_cf": 0.002,
            } if not skip_user_neighbor else {}),
            **({
                "joint_user_balanced_catboost": 0.001875,
                "joint_watch_ratio_deepfm": 0.001875,
            } if not skip_user_balanced_tree else {}),
            **({
                "single_session_yeti_rank_consensus": 0.685,
            } if not skip_yeti_session_gate else {}),
            **({
                "two_three_session_median_extrapolation": -0.0275,
            } if not skip_session_median_gate else {}),
            **({
                "high_recent_coverage_shallow_yeti_extrapolation": -0.0925,
            } if not skip_recent_yeti_gate else {}),
        } if not skip_batch_slate else {}),
    },
    "feature_scope": "training-only outcome models/priors plus label-free full evaluation slate",
    "verification_resource_usage": tracker.finish(),
}, indent=2), flush=True)
