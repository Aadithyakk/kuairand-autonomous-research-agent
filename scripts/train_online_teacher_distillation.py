#!/usr/bin/env python3
"""Distill the verified online rank teacher into a frozen outcome-free residual.

This audit never reads rows after 2022-04-28 from an outcome column. Teacher
targets through April 24 are used for model selection, April 25 is used only to
select the residual recipe, and April 26-28 remain locked until the final refit.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, CatBoostRegressor, Pool


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
VALIDATION_ROWS = 124_909
OUTCOME_COLUMNS = {
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "profile_stay_time", "comment_stay_time",
    "is_profile_enter",
}


def rank_within_user(users: np.ndarray, scores: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"user": users.astype(str), "score": scores, "row": np.arange(len(scores))})
    frame["rank"] = frame.groupby("user", sort=False)["score"].rank(method="first", pct=True)
    return frame.sort_values("row")["rank"].to_numpy(dtype=np.float32)


def evaluate(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict:
    from scripts.kuairand_runner import evaluate_module

    measured = evaluate_module.evaluate(users.astype(str).tolist(), labels.astype(np.float32), scores)
    return {
        "primary": float(measured["primary"]),
        "gauc": float(measured["GAUC"]),
        "ndcg5": float(measured["nDCG@5"]),
        "rows": int(measured["rows"]),
        "users": int(measured["users"]),
    }


def load_public_validation() -> tuple[pd.DataFrame, np.ndarray]:
    path = DATA / "log_standard_4_22_to_5_08_pure.csv"
    feature_columns = ["user_id", "video_id", "date", "hourmin", "time_ms", "duration_ms", "is_rand", "tab"]
    frame = pd.read_csv(path, usecols=feature_columns)
    frame = frame.loc[frame["date"] <= 20220428].reset_index(drop=True)
    labels = []
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["date"]) > 20220428:
                continue
            labels.append(1.0 if row["long_view"] != "0" else 0.0)
    if len(frame) != VALIDATION_ROWS or len(labels) != VALIDATION_ROWS:
        raise RuntimeError(f"Expected {VALIDATION_ROWS} public rows")
    if int(frame["date"].max()) != 20220428:
        raise RuntimeError("Public-row boundary is not April 28; refusing to read outcomes")
    return frame, np.asarray(labels, dtype=np.float32)


def add_training_priors(frame: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    source_columns = ["user_id", "video_id", "long_view", "play_time_ms", "duration_ms"]
    history = pd.read_csv(DATA / "log_standard_4_08_to_4_21_pure.csv", usecols=source_columns)
    history = history.merge(basic[["video_id", "author_id"]], on="video_id", how="left")
    history["watch_ratio"] = np.clip(
        history["play_time_ms"].to_numpy(dtype=np.float64)
        / np.maximum(history["duration_ms"].to_numpy(dtype=np.float64), 1.0),
        0.0,
        2.0,
    )
    global_long = float(history["long_view"].mean())
    global_watch = float(history["watch_ratio"].mean())
    result = frame
    for keys, name, alpha in (
        (["user_id"], "hist_user", 30.0),
        (["video_id"], "hist_video", 30.0),
        (["author_id"], "hist_author", 50.0),
        (["user_id", "author_id"], "hist_user_author", 15.0),
    ):
        grouped = history.groupby(keys, dropna=False).agg(
            prior_count=("long_view", "size"),
            prior_long_sum=("long_view", "sum"),
            prior_watch_sum=("watch_ratio", "sum"),
        ).reset_index()
        grouped[f"{name}_log_count"] = np.log1p(grouped["prior_count"].to_numpy(dtype=np.float64))
        grouped[f"{name}_long"] = (
            grouped["prior_long_sum"] + alpha * global_long
        ) / (grouped["prior_count"] + alpha)
        grouped[f"{name}_watch"] = (
            grouped["prior_watch_sum"] + alpha * global_watch
        ) / (grouped["prior_count"] + alpha)
        grouped = grouped[keys + [f"{name}_log_count", f"{name}_long", f"{name}_watch"]]
        result = result.merge(grouped, on=keys, how="left", sort=False)
        result[f"{name}_log_count"] = result[f"{name}_log_count"].fillna(0.0)
        result[f"{name}_long"] = result[f"{name}_long"].fillna(global_long)
        result[f"{name}_watch"] = result[f"{name}_watch"].fillna(global_watch)
    return result


def add_slate_features(frame: pd.DataFrame) -> pd.DataFrame:
    from backend.kuailab.slate import build_slate_features

    rows = [
        (
            int(row.date), str(row.user_id), str(row.video_id), str(row.author_id), str(row.tab),
            float(row.duration_ms), 0, int(row.hour), int(row.time_ms), 0.0,
        )
        for row in frame.itertuples(index=False)
    ]
    values, names = build_slate_features(rows)
    for index, name in enumerate(names):
        frame[f"slate_{name}"] = values[:, index]
    for column in ("video_id", "author_id", "music_id", "tag"):
        frame[f"slate_{column}_frequency"] = (
            frame.groupby(["user_id", column], sort=False)[column].transform("size").astype(np.float32)
        )
        frame[f"slate_{column}_seen"] = frame.groupby(["user_id", column], sort=False).cumcount().astype(np.float32)
    return frame


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    forbidden = OUTCOME_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise RuntimeError(f"Outcome columns reached feature builder: {sorted(forbidden)}")
    basic_columns = [
        "video_id", "author_id", "video_type", "upload_dt", "upload_type", "visible_status",
        "video_duration", "server_width", "server_height", "music_id", "music_type", "tag",
    ]
    basic = pd.read_csv(DATA / "video_features_basic_pure.csv", usecols=basic_columns)
    users = pd.read_csv(DATA / "user_features_pure.csv")
    feature_frame = frame.copy()
    feature_frame["_row"] = np.arange(len(feature_frame))
    feature_frame = feature_frame.merge(basic, on="video_id", how="left", sort=False)
    feature_frame = feature_frame.merge(users, on="user_id", how="left", sort=False)
    feature_frame = feature_frame.sort_values("_row").reset_index(drop=True)
    feature_frame["hour"] = (feature_frame["hourmin"].fillna(0).astype(int) // 100).clip(0, 23)
    feature_frame["minute"] = (feature_frame["hourmin"].fillna(0).astype(int) % 100).clip(0, 59)
    feature_frame["weekday"] = pd.to_datetime(feature_frame["date"].astype(str)).dt.dayofweek
    feature_frame["hour_sin"] = np.sin(2 * np.pi * feature_frame["hour"] / 24.0)
    feature_frame["hour_cos"] = np.cos(2 * np.pi * feature_frame["hour"] / 24.0)
    feature_frame["log_duration_ms"] = np.log1p(feature_frame["duration_ms"].clip(lower=0))
    feature_frame["aspect_ratio"] = feature_frame["server_height"] / feature_frame["server_width"].replace(0, np.nan)
    upload = pd.to_datetime(feature_frame["upload_dt"], errors="coerce")
    exposure = pd.to_datetime(feature_frame["date"].astype(str), errors="coerce")
    feature_frame["video_age_days"] = (exposure - upload).dt.days.clip(lower=0).fillna(-1)
    feature_frame = add_training_priors(feature_frame, basic)
    feature_frame = add_slate_features(feature_frame)

    categorical = [
        "user_id", "video_id", "author_id", "video_type", "upload_type", "music_id", "music_type",
        "tag", "tab", "is_rand", "weekday", "hour", "user_active_degree", "is_lowactive_period",
        "is_live_streamer", "is_video_author", "follow_user_num_range", "fans_user_num_range",
        "friend_user_num_range", "register_days_range",
    ]
    categorical = [column for column in categorical if column in feature_frame]
    ignored = {"_row", "date", "time_ms", "hourmin", "upload_dt"}
    numeric = [
        column for column in feature_frame.columns
        if column not in set(categorical).union(ignored) and column not in OUTCOME_COLUMNS
    ]
    for column in categorical:
        feature_frame[column] = feature_frame[column].fillna("UNK").astype(str)
    for column in numeric:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return feature_frame[categorical + numeric], categorical, numeric


def scan_weight(users: np.ndarray, labels: np.ndarray, champion: np.ndarray, residual: np.ndarray) -> tuple[float, dict, list[dict]]:
    records = []
    for weight in np.linspace(-0.25, 1.0, 26):
        metrics = evaluate(users, labels, rank_within_user(users, champion) + float(weight) * residual)
        records.append({"weight": float(weight), **metrics})
    best = max(records, key=lambda item: (item["primary"], item["gauc"], item["ndcg5"], -abs(item["weight"])))
    return float(best["weight"]), best, records


def scan_rank_blend(users: np.ndarray, labels: np.ndarray, champion: np.ndarray, student: np.ndarray) -> tuple[float, dict, list[dict]]:
    champion_rank = rank_within_user(users, champion)
    student_rank = rank_within_user(users, student)
    records = []
    for weight in np.linspace(-0.25, 1.0, 26):
        scores = (1.0 - float(weight)) * champion_rank + float(weight) * student_rank
        metrics = evaluate(users, labels, scores)
        records.append({"weight": float(weight), **metrics})
    best = max(records, key=lambda item: (item["primary"], item["gauc"], item["ndcg5"], -abs(item["weight"])))
    return float(best["weight"]), best, records


def train_model(features: pd.DataFrame, target: np.ndarray, indices: np.ndarray, categorical: list[str], depth: int, iterations: int, seed: int) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=depth,
        iterations=iterations,
        learning_rate=0.04,
        l2_leaf_reg=20.0,
        random_seed=seed,
        random_strength=0.5,
        verbose=False,
        allow_writing_files=False,
        thread_count=6,
    )
    model.fit(features.iloc[indices], target[indices], cat_features=categorical)
    return model


def train_ranker(features: pd.DataFrame, target: np.ndarray, users: np.ndarray, indices: np.ndarray, categorical: list[str], depth: int, iterations: int, seed: int, loss: str) -> CatBoostRanker:
    ordered = indices[np.argsort(users[indices], kind="stable")]
    pool = Pool(
        features.iloc[ordered], target[ordered],
        cat_features=categorical, group_id=users[ordered].astype(str),
    )
    model = CatBoostRanker(
        loss_function=loss,
        depth=depth,
        iterations=iterations,
        learning_rate=0.04,
        l2_leaf_reg=20.0,
        random_seed=seed,
        random_strength=0.5,
        verbose=False,
        allow_writing_files=False,
        thread_count=6,
    )
    model.fit(pool)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "teacher-distillation" / "audit.json")
    parser.add_argument("--model-output", type=Path, default=ROOT / "runtime" / "teacher-distillation.cbm")
    args = parser.parse_args()
    started = time.monotonic()
    frame, labels = load_public_validation()
    teacher_file = np.load(args.teacher_scores)
    teacher = np.asarray(teacher_file["selected"], dtype=np.float32)
    champion = np.asarray(np.load(ROOT / "results" / "final-model" / "validation-scores.npz")["scores"], dtype=np.float32)
    if len(teacher) != VALIDATION_ROWS or len(champion) != VALIDATION_ROWS:
        raise RuntimeError("Teacher or champion score alignment failed")
    users = frame["user_id"].astype(str).to_numpy()
    teacher_rank = rank_within_user(users, teacher)
    champion_rank = rank_within_user(users, champion)
    residual_target = teacher_rank - champion_rank
    features, categorical, numeric = build_features(frame)
    dates = frame["date"].to_numpy(dtype=np.int32)
    train_indices = np.flatnonzero(dates <= 20220424)
    selection_indices = np.flatnonzero(dates == 20220425)
    locked_indices = np.flatnonzero(dates >= 20220426)

    configs = []
    for depth, iterations in ((6, 220), (8, 260)):
        model = train_model(features, residual_target, train_indices, categorical, depth, iterations, 2200 + depth)
        prediction = model.predict(features.iloc[selection_indices]).astype(np.float32)
        weight, best, scan = scan_weight(
            users[selection_indices], labels[selection_indices], champion[selection_indices], prediction,
        )
        configs.append({"kind": "residual_regression", "loss": "RMSE", "depth": depth, "iterations": iterations, "weight": weight, "selection": best, "weight_scan": scan})
    for loss, depth, iterations in (("QueryRMSE", 6, 260), ("YetiRankPairwise", 6, 220)):
        model = train_ranker(
            features, teacher_rank, users, train_indices, categorical,
            depth, iterations, 2400 + iterations, loss,
        )
        prediction = model.predict(features.iloc[selection_indices]).astype(np.float32)
        weight, best, scan = scan_rank_blend(
            users[selection_indices], labels[selection_indices], champion[selection_indices], prediction,
        )
        configs.append({"kind": "teacher_rank", "loss": loss, "depth": depth, "iterations": iterations, "weight": weight, "selection": best, "weight_scan": scan})
    chosen = max(configs, key=lambda item: (item["selection"]["primary"], -abs(item["weight"])))

    refit_indices = np.flatnonzero(dates <= 20220425)
    if chosen["kind"] == "teacher_rank":
        model = train_ranker(
            features, teacher_rank, users, refit_indices, categorical,
            int(chosen["depth"]), int(chosen["iterations"]), 2600 + int(chosen["depth"]), str(chosen["loss"]),
        )
    else:
        model = train_model(
            features, residual_target, refit_indices, categorical,
            int(chosen["depth"]), int(chosen["iterations"]), 2300 + int(chosen["depth"]),
        )
    locked_prediction = model.predict(features.iloc[locked_indices]).astype(np.float32)
    if chosen["kind"] == "teacher_rank":
        locked_candidate = (
            (1.0 - float(chosen["weight"])) * rank_within_user(users[locked_indices], champion[locked_indices])
            + float(chosen["weight"]) * rank_within_user(users[locked_indices], locked_prediction)
        )
    else:
        locked_candidate = champion_rank[locked_indices] + float(chosen["weight"]) * locked_prediction
    locked = {
        "champion": evaluate(users[locked_indices], labels[locked_indices], champion[locked_indices]),
        "distilled_residual": evaluate(users[locked_indices], labels[locked_indices], locked_candidate),
        "online_teacher_upper_bound": evaluate(users[locked_indices], labels[locked_indices], teacher[locked_indices]),
    }
    locked["gain_vs_champion"] = {
        key: locked["distilled_residual"][key] - locked["champion"][key]
        for key in ("primary", "gauc", "ndcg5")
    }
    promote = bool(
        locked["gain_vs_champion"]["primary"] > 0
        and locked["gain_vs_champion"]["gauc"] >= 0
        and locked["gain_vs_champion"]["ndcg5"] >= 0
    )
    report = {
        "experiment": "frozen outcome-free distillation of verified online teacher residual",
        "status": "promoted" if promote else "rejected_locked_regression",
        "teacher_primary_full_public_validation": 0.7234153747558594,
        "protocol": {
            "teacher_target_train": "2022-04-22..2022-04-24",
            "configuration_selection": "2022-04-25",
            "refit": "2022-04-22..2022-04-25",
            "locked_test": "2022-04-26..2022-04-28",
            "prediction_features": "identity, supplied metadata, exposure/slate/session structure, and April 8-21 outcome priors",
            "target_period_outcomes_in_features": False,
            "rows_after_2022_04_28_outcomes_parsed": False
        },
        "features": {"categorical": categorical, "numeric": numeric, "count": len(categorical) + len(numeric)},
        "selection_configs": configs,
        "selected": {key: chosen[key] for key in ("kind", "loss", "depth", "iterations", "weight", "selection")},
        "locked": locked,
        "promotion_gate": {
            "passed": promote,
            "rule": "positive locked primary with nonnegative locked GAUC and nDCG@5",
            "champion_modified": False,
        },
        "runtime_seconds": time.monotonic() - started,
        "artifacts": {"model": str(args.model_output.relative_to(ROOT)), "teacher_scores": str(args.teacher_scores)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_output)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": report["selected"], "locked": locked, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
