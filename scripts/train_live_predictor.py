#!/usr/bin/env python3
"""Train and export a leakage-safe browser/API logistic ranker.

The exported April 29 cohort contains only information available at impression
time. Its outcome and engagement columns are never copied into the artifact.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.live_predictor import FORBIDDEN_CANDIDATE_FIELDS


DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
OUTPUT = ROOT / "public" / "live-predictor.json"
HASH_DIM = 1 << 17
HISTORY_END = 20220421
SELECTION_END = 20220427
VALID_DATE = 20220428
TARGET_DATE = 20220429
CATEGORICAL_FIELDS = (
    "user",
    "video",
    "author",
    "music",
    "tab",
    "duration_bucket",
    "hour",
    "weekday",
    "video_type",
    "upload_type",
    "tag",
    "user_tab",
    "user_hour",
    "author_tab",
)
NUMERIC_FIELDS = (
    "log_duration_ms",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "user_history_rate",
    "log_user_history_count",
    "video_history_rate",
    "log_video_history_count",
    "author_history_rate",
    "log_author_history_count",
    "user_tab_history_rate",
    "log_user_tab_history_count",
    "log_followers",
    "log_fans",
    "log_friends",
    "log_register_days",
    "video_aspect_ratio",
)


def stable_hash(token: str) -> int:
    value = 2166136261
    for byte in token.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value % HASH_DIM


@dataclass
class Example:
    date: int
    user_id: str
    video_id: str
    author_id: str
    music_id: str
    tab: str
    duration_ms: float
    hour: int
    time_ms: int
    video_type: str
    upload_type: str
    tag: str
    label: int | None


def read_static() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    videos: dict[str, dict[str, str]] = {}
    with (DATA / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            videos[row["video_id"]] = row
    users: dict[str, dict[str, str]] = {}
    with (DATA / "user_features_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            users[row["user_id"]] = row
    return videos, users


def safe_number(value: str | None) -> float:
    try:
        number = float(value or 0.0)
        return number if math.isfinite(number) and number >= 0 else 0.0
    except ValueError:
        return 0.0


def history_statistics(videos: dict[str, dict[str, str]]) -> dict:
    user = defaultdict(lambda: [0, 0])
    video = defaultdict(lambda: [0, 0])
    author = defaultdict(lambda: [0, 0])
    user_tab = defaultdict(lambda: [0, 0])
    total = positive = 0
    path = DATA / "log_standard_4_08_to_4_21_pure.csv"
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            label = 1 if row["long_view"] != "0" else 0
            user_id, video_id, tab = row["user_id"], row["video_id"], row["tab"]
            author_id = videos.get(video_id, {}).get("author_id", "UNK")
            for mapping, key in (
                (user, user_id),
                (video, video_id),
                (author, author_id),
                (user_tab, f"{user_id}|{tab}"),
            ):
                mapping[key][0] += label
                mapping[key][1] += 1
            positive += label
            total += 1
    return {
        "global_rate": positive / total,
        "rows": total,
        "user": user,
        "video": video,
        "author": author,
        "user_tab": user_tab,
    }


def load_later(videos: dict[str, dict[str, str]]) -> tuple[list[Example], list[Example]]:
    development: list[Example] = []
    target: list[Example] = []
    path = DATA / "log_standard_4_22_to_5_08_pure.csv"
    with path.open(encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        safe = {
            "user_id", "video_id", "date", "hourmin", "time_ms",
            "duration_ms", "is_rand", "tab",
        }
        for values in reader:
            date = int(values[index["date"]])
            if not 20220422 <= date <= TARGET_DATE:
                continue
            row = {name: values[index[name]] for name in safe}
            video = videos.get(row["video_id"], {})
            common = dict(
                date=date,
                user_id=row["user_id"],
                video_id=row["video_id"],
                author_id=video.get("author_id", "UNK"),
                music_id=video.get("music_id", "UNK"),
                tab=row["tab"],
                duration_ms=safe_number(row["duration_ms"]),
                hour=int(row["hourmin"]) // 100,
                time_ms=int(row["time_ms"]),
                video_type=video.get("video_type", "UNK"),
                upload_type=video.get("upload_type", "UNK"),
                tag=video.get("tag", "UNK"),
            )
            if date == TARGET_DATE:
                # Do not index any outcome column for the target date.
                target.append(Example(**common, label=None))
            else:
                development.append(
                    Example(
                        **common,
                        label=1 if values[index["long_view"]] != "0" else 0,
                    )
                )
    return development, target


def smoothed(mapping: dict, key: str, global_rate: float, prior: float = 20.0) -> tuple[float, float]:
    positive, count = mapping.get(key, (0, 0))
    return (positive + prior * global_rate) / (count + prior), math.log1p(count)


def raw_features(
    example: Example,
    histories: dict,
    users: dict[str, dict[str, str]],
    videos: dict[str, dict[str, str]],
) -> tuple[list[int], list[float]]:
    duration_bucket = min(9, int(math.log1p(max(0.0, example.duration_ms)) / 1.2))
    weekday = int(str(example.date)[-2:]) % 7
    tokens = (
        f"user={example.user_id}",
        f"video={example.video_id}",
        f"author={example.author_id}",
        f"music={example.music_id}",
        f"tab={example.tab}",
        f"duration_bucket={duration_bucket}",
        f"hour={example.hour}",
        f"weekday={weekday}",
        f"video_type={example.video_type}",
        f"upload_type={example.upload_type}",
        f"tag={example.tag}",
        f"user_tab={example.user_id}|{example.tab}",
        f"user_hour={example.user_id}|{example.hour}",
        f"author_tab={example.author_id}|{example.tab}",
    )
    if len(tokens) != len(CATEGORICAL_FIELDS):
        raise RuntimeError("Categorical feature schema drift")
    global_rate = histories["global_rate"]
    user_rate, user_count = smoothed(histories["user"], example.user_id, global_rate)
    video_rate, video_count = smoothed(histories["video"], example.video_id, global_rate)
    author_rate, author_count = smoothed(histories["author"], example.author_id, global_rate)
    user_tab_rate, user_tab_count = smoothed(
        histories["user_tab"], f"{example.user_id}|{example.tab}", global_rate
    )
    user = users.get(example.user_id, {})
    video = videos.get(example.video_id, {})
    width = safe_number(video.get("server_width"))
    height = safe_number(video.get("server_height"))
    angle = 2.0 * math.pi * example.hour / 24.0
    weekday_angle = 2.0 * math.pi * weekday / 7.0
    numeric = [
        math.log1p(example.duration_ms),
        math.sin(angle),
        math.cos(angle),
        math.sin(weekday_angle),
        math.cos(weekday_angle),
        user_rate,
        user_count,
        video_rate,
        video_count,
        author_rate,
        author_count,
        user_tab_rate,
        user_tab_count,
        math.log1p(safe_number(user.get("follow_user_num"))),
        math.log1p(safe_number(user.get("fans_user_num"))),
        math.log1p(safe_number(user.get("friend_user_num"))),
        math.log1p(safe_number(user.get("register_days"))),
        width / max(height, 1.0),
    ]
    return [stable_hash(token) for token in tokens], numeric


def matrix(
    rows: list[Example], histories: dict, users: dict, videos: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    indices = np.empty((len(rows), len(CATEGORICAL_FIELDS)), dtype=np.int32)
    numeric = np.empty((len(rows), len(NUMERIC_FIELDS)), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.float32)
    user_ids: list[str] = []
    for position, row in enumerate(rows):
        categorical, numbers = raw_features(row, histories, users, videos)
        indices[position] = categorical
        numeric[position] = numbers
        labels[position] = float(row.label or 0)
        user_ids.append(row.user_id)
    return indices, numeric, labels, user_ids


def predict(weights: np.ndarray, numeric_weights: np.ndarray, intercept: float, indices: np.ndarray, numeric: np.ndarray) -> np.ndarray:
    logits = intercept + weights[indices].sum(axis=1) + numeric @ numeric_weights
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def fit(
    indices: np.ndarray,
    numeric: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.zeros(HASH_DIM, dtype=np.float32)
    numeric_weights = np.zeros(numeric.shape[1], dtype=np.float32)
    intercept = 0.0
    accumulator = np.full(HASH_DIM, 1e-4, dtype=np.float32)
    numeric_accumulator = np.full(numeric.shape[1], 1e-4, dtype=np.float32)
    bias_accumulator = 1e-4
    random = np.random.default_rng(seed)
    batch_size = 8192
    learning_rate = 0.18
    l2 = 2e-5
    for _ in range(epochs):
        order = random.permutation(len(labels))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            probability = predict(weights, numeric_weights, intercept, indices[batch], numeric[batch])
            error = probability - labels[batch]
            scale = max(1, len(batch))
            gradient = np.bincount(
                indices[batch].ravel(),
                weights=np.repeat(error, indices.shape[1]),
                minlength=HASH_DIM,
            ).astype(np.float32) / scale
            gradient += l2 * weights
            numeric_gradient = numeric[batch].T @ error / scale + l2 * numeric_weights
            bias_gradient = float(np.mean(error))
            accumulator += gradient * gradient
            numeric_accumulator += numeric_gradient * numeric_gradient
            bias_accumulator += bias_gradient * bias_gradient
            weights -= learning_rate * gradient / np.sqrt(accumulator)
            numeric_weights -= learning_rate * numeric_gradient / np.sqrt(numeric_accumulator)
            intercept -= learning_rate * bias_gradient / math.sqrt(bias_accumulator)
    return weights, numeric_weights, intercept


def auc(labels: list[int], scores: list[float]) -> float:
    order = np.argsort(np.asarray(scores), kind="mergesort")
    sorted_scores = np.asarray(scores)[order]
    sorted_labels = np.asarray(labels)[order]
    ranks = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + end + 1) / 2.0
        start = end
    positives = float(sorted_labels.sum())
    negatives = len(sorted_labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    return float((ranks[sorted_labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def metrics(user_ids: list[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for user_id, label, score in zip(user_ids, labels, scores):
        groups[user_id].append((float(score), int(label)))
    gauc_numerator = gauc_denominator = 0.0
    ndcgs: list[float] = []
    discounts = [math.log2(index + 2) for index in range(5)]
    for rows in groups.values():
        rows.sort(key=lambda item: -item[0])
        group_labels = [label for _, label in rows]
        positives = sum(group_labels)
        if 0 < positives < len(rows):
            gauc_numerator += positives * auc(group_labels, [score for score, _ in rows])
            gauc_denominator += positives
        dcg = sum(label / discounts[index] for index, label in enumerate(group_labels[:5]))
        ideal = sorted(group_labels, reverse=True)[:5]
        idcg = sum(label / discounts[index] for index, label in enumerate(ideal))
        ndcgs.append(0.0 if idcg == 0 else dcg / idcg)
    gauc = gauc_numerator / gauc_denominator if gauc_denominator else 0.5
    ndcg5 = float(np.mean(ndcgs)) if ndcgs else 0.0
    return {
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": (gauc + ndcg5) / 2.0,
        "users": len(groups),
        "rows": len(labels),
    }


def main() -> int:
    videos, users = read_static()
    histories = history_statistics(videos)
    development, target = load_later(videos)
    selection = [row for row in development if row.date <= SELECTION_END]
    validation = [row for row in development if row.date == VALID_DATE]
    selection_indices, selection_numeric, selection_labels, _ = matrix(selection, histories, users, videos)
    valid_indices, valid_numeric, valid_labels, valid_users = matrix(validation, histories, users, videos)
    means = selection_numeric.mean(axis=0)
    scales = selection_numeric.std(axis=0)
    scales[scales < 1e-6] = 1.0
    normalized_selection = (selection_numeric - means) / scales
    normalized_valid = (valid_numeric - means) / scales

    best: tuple[float, int, dict] | None = None
    for epochs in (2, 4, 7, 11, 16):
        weights, numeric_weights, intercept = fit(
            selection_indices, normalized_selection, selection_labels, epochs
        )
        result = metrics(
            valid_users,
            valid_labels,
            predict(weights, numeric_weights, intercept, valid_indices, normalized_valid),
        )
        print(f"epochs={epochs:2d} primary={result['primary']:.6f} GAUC={result['gauc']:.6f} nDCG@5={result['ndcg5']:.6f}", flush=True)
        if best is None or result["primary"] > best[0]:
            best = (result["primary"], epochs, result)
    assert best is not None

    full_indices, full_numeric, full_labels, _ = matrix(development, histories, users, videos)
    final_means = full_numeric.mean(axis=0)
    final_scales = full_numeric.std(axis=0)
    final_scales[final_scales < 1e-6] = 1.0
    weights, numeric_weights, intercept = fit(
        full_indices,
        (full_numeric - final_means) / final_scales,
        full_labels,
        best[1],
    )

    by_user: dict[str, list[Example]] = defaultdict(list)
    for row in target:
        by_user[row.user_id].append(row)
    eligible = [
        (user_id, sorted(rows, key=lambda row: row.time_ms))
        for user_id, rows in by_user.items()
        if 5 <= len(rows) <= 24
    ]
    eligible.sort(key=lambda item: (-len(item[1]), int(item[0])))
    selected = eligible[:24]
    exported_candidates = []
    used_indices: set[int] = set()
    for user_id, rows in selected:
        for exposure_index, row in enumerate(rows):
            categorical, numbers = raw_features(row, histories, users, videos)
            used_indices.update(categorical)
            exported_candidates.append(
                {
                    "user_id": user_id,
                    "video_id": row.video_id,
                    "author_id": row.author_id,
                    "video_type": row.video_type,
                    "tab": row.tab,
                    "hour": row.hour,
                    "duration_seconds": round(row.duration_ms / 1000.0, 3),
                    "exposure_index": exposure_index,
                    "categorical_indices": categorical,
                    "numeric_values": [round(float(value), 8) for value in numbers],
                }
            )

    artifact = {
        "schema_version": 1,
        "model": {
            "name": "KuaiRand deployment logistic surrogate",
            "kind": "hashed logistic regression",
            "hash": "FNV-1a/131072",
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "numeric_fields": list(NUMERIC_FIELDS),
            "selected_epochs": best[1],
            "intercept": round(float(intercept), 10),
            "candidate_weights": {
                str(index): round(float(weights[index]), 10) for index in sorted(used_indices)
            },
            "numeric_means": [round(float(value), 10) for value in final_means],
            "numeric_scales": [round(float(value), 10) for value in final_scales],
            "numeric_weights": [round(float(value), 10) for value in numeric_weights],
        },
        "training": {
            "history_window": "2022-04-08..2022-04-21",
            "fit_window": "2022-04-22..2022-04-28",
            "history_rows": histories["rows"],
            "fit_rows": len(development),
            "label": "long_view",
        },
        "evaluation": {
            "window": "2022-04-28 temporal holdout",
            "note": "Surrogate-only temporal proxy; not comparable to the 0.723415 online champion.",
            **{key: round(float(value), 9) if isinstance(value, float) else value for key, value in best[2].items()},
        },
        "target": {
            "date": "2022-04-29",
            "kind": "logged candidate-slate reranking",
            "cohort_users": len(selected),
            "cohort_rows": len(exported_candidates),
        },
        "integrity": {
            "target_outcomes_accessed": False,
            "target_scores_precomputed": False,
            "target_outcome_columns_excluded": sorted(FORBIDDEN_CANDIDATE_FIELDS),
            "champion_0_723_used": False,
        },
        "users": [
            {"user_id": user_id, "candidate_count": len(rows)} for user_id, rows in selected
        ],
        "candidates": exported_candidates,
    }
    forbidden = FORBIDDEN_CANDIDATE_FIELDS.intersection(
        key for row in exported_candidates for key in row
    )
    if forbidden:
        raise RuntimeError(f"Refusing to export target outcomes: {sorted(forbidden)}")
    OUTPUT.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
