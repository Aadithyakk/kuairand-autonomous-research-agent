from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime" / "catboost-venv"))

from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.linear_model import LogisticRegression

from backend.kuailab.resources import ProcessResourceTracker
from scripts import kuairand_runner as runner


parser = argparse.ArgumentParser()
parser.add_argument("--residual-baseline", action="store_true")
parser.add_argument(
    "--ranker-objective",
    choices=("PairLogitPairwise", "QuerySoftMax", "YetiRankPairwise"),
    default="",
)
parser.add_argument("--ranker-iterations", type=int, default=500)
parser.add_argument("--ranker-depth", type=int, default=8)
parser.add_argument("--ranker-learning-rate", type=float, default=0.04)
parser.add_argument("--ranker-l2", type=float, default=12.0)
parser.add_argument("--ranker-seed", type=int, default=769)
args = parser.parse_args()

tracker = ProcessResourceTracker()
residual_mode = args.residual_baseline
ranker_mode = bool(args.ranker_objective)
os.environ["KUAI_SKIP_BATCH_SLATE"] = "1"
with contextlib.redirect_stdout(io.StringIO()):
    from scripts import verify_slate_consensus as champion

data_dir = ROOT / "external" / "KuaiRand-Pure" / "data"
meta_base = np.load(
    ROOT / "runtime" / "stacked-reranker"
    / "out-of-time-base-cutoff-20220413.npz"
)["scores"].astype(np.float64)
valid_base = champion.scores.astype(np.float64, copy=True)
valid_y = champion.valid_y.astype(np.float32, copy=False)
valid_users = champion.valid_users

columns = [
    "user_id", "video_id", "date", "hourmin", "time_ms", "duration_ms",
    "tab", "long_view",
]
meta = (
    pl.read_csv(
        data_dir / "log_standard_4_08_to_4_21_pure.csv", columns=columns,
    )
    .filter(pl.col("date") >= 20220414)
    .with_row_index("batch_row")
    .with_columns(pl.lit(0).cast(pl.Int8).alias("batch"))
)
valid = (
    pl.read_csv(
        data_dir / "log_standard_4_22_to_5_08_pure.csv", columns=columns,
    )
    .filter(pl.col("date") <= 20220428)
    .with_row_index("batch_row")
    .with_columns(pl.lit(1).cast(pl.Int8).alias("batch"))
)
if len(meta) != len(meta_base) or len(valid) != len(valid_base):
    raise RuntimeError(
        f"Base alignment failed: meta {len(meta)} != {len(meta_base)}, "
        f"valid {len(valid)} != {len(valid_base)}"
    )
meta = meta.with_columns(pl.Series("base_score", meta_base))
valid = valid.with_columns(pl.Series("base_score", valid_base))
frame = pl.concat([meta, valid], how="vertical")

videos = pl.read_csv(data_dir / "video_features_basic_pure.csv").select([
    "video_id", "author_id", "music_id", "tag", "video_type", "upload_type",
    "visible_status", "music_type", "server_width", "server_height",
])
statistics = pl.read_csv(data_dir / "video_features_statistic_pure.csv").select([
    "video_id", "show_cnt", "show_user_num", "play_cnt", "valid_play_cnt",
    "complete_play_cnt", "long_time_play_cnt", "long_time_play_user_num",
    "play_progress", "like_cnt", "comment_cnt", "share_cnt",
])
frame = (
    frame.join(videos, on="video_id", how="left")
    .join(statistics, on="video_id", how="left")
    .with_columns([
        pl.col("user_id").cast(pl.String),
        pl.col("video_id").cast(pl.String),
        pl.col("author_id").cast(pl.String).fill_null("UNK"),
        pl.col("music_id").cast(pl.String).fill_null("UNK"),
        pl.col("tag").cast(pl.String).fill_null("UNK"),
        pl.col("video_type").cast(pl.String).fill_null("UNK"),
        pl.col("upload_type").cast(pl.String).fill_null("UNK"),
        pl.col("tab").cast(pl.String),
        (pl.col("duration_ms") / 10_000).floor().clip(0, 60)
        .cast(pl.String).alias("duration_bucket"),
        pl.when(pl.col("duration_ms") < 10_000).then(pl.lit("short"))
        .when(pl.col("duration_ms") < 18_000).then(pl.lit("medium"))
        .otherwise(pl.lit("long")).alias("threshold_bucket"),
        (pl.col("hourmin") // 400).cast(pl.String).alias("daypart"),
        (pl.col("duration_ms").clip(1, None).log1p()).alias("log_duration"),
        (pl.col("show_cnt").fill_null(0).clip(0, None).log1p()).alias("log_show"),
        (pl.col("play_cnt").fill_null(0).clip(0, None).log1p()).alias("log_play"),
        (pl.col("long_time_play_cnt") / (pl.col("show_cnt") + 20.0))
        .fill_nan(0).fill_null(0).alias("stat_long_rate"),
        (pl.col("long_time_play_user_num") / (pl.col("show_user_num") + 20.0))
        .fill_nan(0).fill_null(0).alias("stat_long_user_rate"),
        (pl.col("valid_play_cnt") / (pl.col("show_cnt") + 20.0))
        .fill_nan(0).fill_null(0).alias("stat_valid_rate"),
        (pl.col("complete_play_cnt") / (pl.col("play_cnt") + 20.0))
        .fill_nan(0).fill_null(0).alias("stat_complete_rate"),
        (pl.col("like_cnt") / (pl.col("show_cnt") + 20.0))
        .fill_nan(0).fill_null(0).alias("stat_like_rate"),
        (pl.col("comment_cnt") / (pl.col("show_cnt") + 20.0))
        .fill_nan(0).fill_null(0).alias("stat_comment_rate"),
        (pl.col("share_cnt") / (pl.col("show_cnt") + 20.0))
        .fill_nan(0).fill_null(0).alias("stat_share_rate"),
    ])
)

# Normalize the base prediction in exactly the metric's ranking unit.
frame = frame.with_columns([
    pl.col("base_score").rank("ordinal").over(["batch", "user_id"])
    .cast(pl.Float64).alias("base_rank"),
    pl.len().over(["batch", "user_id"]).cast(pl.Float64).alias("user_slate_length"),
]).with_columns(
    (
        (pl.col("base_rank") - 1.0)
        / (pl.col("user_slate_length") - 1.0).clip(1, None)
    ).alias("base_rank_fraction")
)

count_specs = {
    "video": ["batch", "user_id", "video_id"],
    "author": ["batch", "user_id", "author_id"],
    "music": ["batch", "user_id", "music_id"],
    "tag": ["batch", "user_id", "tag"],
    "type": ["batch", "user_id", "video_type"],
    "tab": ["batch", "user_id", "tab"],
    "duration": ["batch", "user_id", "duration_bucket"],
    "date": ["batch", "user_id", "date"],
    "date_tab": ["batch", "user_id", "date", "tab"],
    "author_daypart": ["batch", "user_id", "author_id", "daypart"],
    "music_type": ["batch", "user_id", "music_id", "video_type"],
}
frame = frame.with_columns([
    pl.len().over(keys).cast(pl.Float64).alias(f"{name}_count")
    for name, keys in count_specs.items()
]).with_columns([
    pl.col(f"{name}_count").log1p().alias(f"log_{name}_count")
    for name in count_specs
])

# Candidate-slate diversity: repeated authors/music/tags can represent either
# policy confidence or fatigue, while unique-video counts distinguish the two.
diversity = (
    frame.group_by(["batch", "user_id", "author_id"])
    .agg(pl.col("video_id").n_unique().alias("author_unique_videos"))
)
music_diversity = (
    frame.group_by(["batch", "user_id", "music_id"])
    .agg(pl.col("video_id").n_unique().alias("music_unique_videos"))
)
frame = frame.join(diversity, on=["batch", "user_id", "author_id"], how="left")
frame = frame.join(
    music_diversity, on=["batch", "user_id", "music_id"], how="left",
)

# Full-session length and causal occurrence index are outcome-free. Session
# length is batch-transductive, matching the official offline evaluation unit.
frame = (
    frame.sort(["batch", "user_id", "time_ms", "batch_row"])
    .with_columns(
        (
            (pl.col("time_ms") - pl.col("time_ms").shift(1).over(["batch", "user_id"]))
            > 1_800_000
        ).fill_null(True).cast(pl.Int32).alias("session_start")
    )
    .with_columns(
        pl.col("session_start").cum_sum().over(["batch", "user_id"])
        .alias("session_id")
    )
    .with_columns([
        pl.len().over(["batch", "user_id", "session_id"])
        .cast(pl.Float64).alias("session_length"),
        (pl.col("time_ms").cum_count().over(["batch", "user_id", "session_id"]) - 1)
        .cast(pl.Float64).alias("session_position"),
        (pl.col("video_id").cum_count().over(["batch", "user_id", "video_id"]) - 1)
        .cast(pl.Float64).alias("video_occurrence"),
        (pl.col("author_id").cum_count().over(["batch", "user_id", "author_id"]) - 1)
        .cast(pl.Float64).alias("author_occurrence"),
    ])
    .with_columns([
        pl.col("session_length").log1p().alias("log_session_length"),
        pl.col("session_position").log1p().alias("log_session_position"),
        (pl.col("video_occurrence") / (pl.col("video_count") - 1).clip(1, None))
        .alias("video_occurrence_fraction"),
        (pl.col("author_occurrence") / (pl.col("author_count") - 1).clip(1, None))
        .alias("author_occurrence_fraction"),
    ])
)

# Leave-one-out score neighborhoods and chronological score consensus. These
# use model predictions only; neither current nor neighboring outcomes enter.
for name, keys in {
    "type": ["batch", "user_id", "video_type"],
    "author": ["batch", "user_id", "author_id"],
    "date_tab": ["batch", "user_id", "date", "tab"],
}.items():
    frame = frame.with_columns([
        pl.col("base_rank_fraction").sum().over(keys).alias(f"{name}_score_sum"),
        pl.len().over(keys).cast(pl.Float64).alias(f"{name}_score_count"),
    ]).with_columns(
        pl.when(pl.col(f"{name}_score_count") > 1)
        .then(
            (pl.col(f"{name}_score_sum") - pl.col("base_rank_fraction"))
            / (pl.col(f"{name}_score_count") - 1)
        )
        .otherwise(pl.col("base_rank_fraction"))
        .alias(f"{name}_score_loo")
    )
frame = frame.with_columns([
    pl.col("base_rank_fraction").shift(1).over(["batch", "user_id"])
    .alias("previous_base_score"),
    pl.col("base_rank_fraction").shift(-1).over(["batch", "user_id"])
    .alias("next_base_score"),
]).with_columns(
    pl.mean_horizontal("previous_base_score", "next_base_score")
    .fill_null(pl.col("base_rank_fraction")).alias("neighbor_base_score")
)

frame = frame.sort(["batch", "batch_row"])
categorical = [
    "video_id", "author_id", "music_id", "tag", "video_type", "upload_type",
    "tab", "duration_bucket", "threshold_bucket", "daypart", "visible_status",
    "music_type",
]
numeric = [
    "base_rank_fraction", "log_duration", "log_show", "log_play",
    "stat_long_rate", "stat_long_user_rate", "stat_valid_rate",
    "stat_complete_rate", "stat_like_rate", "stat_comment_rate",
    "stat_share_rate", "play_progress", "server_width", "server_height",
    *[f"log_{name}_count" for name in count_specs],
    "author_unique_videos", "music_unique_videos", "log_session_length",
    "log_session_position", "video_occurrence_fraction",
    "author_occurrence_fraction", "type_score_loo", "author_score_loo",
    "date_tab_score_loo", "neighbor_base_score",
]

feature_data: dict[str, np.ndarray] = {}
for name in categorical:
    values = frame[name].cast(pl.String).fill_null("UNK").to_numpy()
    _, encoded = np.unique(values, return_inverse=True)
    feature_data[name] = encoded.astype(np.int32, copy=False)
for name in numeric:
    feature_data[name] = (
        frame[name].cast(pl.Float64, strict=False).fill_nan(0).fill_null(0)
        .to_numpy().astype(np.float32, copy=False)
    )
features = pd.DataFrame(feature_data)
dates = frame["date"].to_numpy().astype(np.int32)
y = frame["long_view"].to_numpy().astype(np.float32)
users = frame["user_id"].to_numpy().astype(str)
batches = frame["batch"].to_numpy()
meta_indices = np.flatnonzero(batches == 0)
valid_indices = np.flatnonzero(batches == 1)
fit = meta_indices[dates[meta_indices] <= 20220419]
holdout = meta_indices[dates[meta_indices] >= 20220420]
base_fraction = frame["base_rank_fraction"].to_numpy().astype(np.float64)
base_calibrator = LogisticRegression(C=100.0, max_iter=200, random_state=751)
base_calibrator.fit(base_fraction[fit, None], y[fit])
base_logit = base_calibrator.decision_function(base_fraction[:, None]).astype(np.float64)

if ranker_mode:
    parameters = dict(
        loss_function=args.ranker_objective, eval_metric="NDCG:top=5",
        iterations=args.ranker_iterations, depth=args.ranker_depth,
        learning_rate=args.ranker_learning_rate, l2_leaf_reg=args.ranker_l2,
        random_seed=args.ranker_seed, thread_count=8, random_strength=0.25,
        od_type="Iter", od_wait=50, allow_writing_files=False, verbose=25,
    )

    def query_order(indices: np.ndarray) -> np.ndarray:
        return indices[np.argsort(users[indices], kind="stable")]

    fit = query_order(fit)
    holdout = query_order(holdout)
    fit_pool = Pool(
        features.iloc[fit], label=y[fit], cat_features=categorical,
        feature_names=list(features.columns), group_id=users[fit],
        baseline=base_logit[fit],
    )
    holdout_pool = Pool(
        features.iloc[holdout], label=y[holdout], cat_features=categorical,
        feature_names=list(features.columns), group_id=users[holdout],
        baseline=base_logit[holdout],
    )
    model = CatBoostRanker(**parameters)
else:
    parameters = dict(
        loss_function="Logloss", eval_metric="AUC", iterations=500, depth=8,
        learning_rate=0.06, l2_leaf_reg=8.0, random_seed=751, thread_count=8,
        random_strength=0.5, bootstrap_type="Bernoulli", subsample=0.85,
        od_type="Iter", od_wait=50, allow_writing_files=False, verbose=25,
    )
    fit_pool = Pool(
        features.iloc[fit], label=y[fit], cat_features=categorical,
        feature_names=list(features.columns),
        baseline=base_logit[fit] if residual_mode else None,
    )
    holdout_pool = Pool(
        features.iloc[holdout], label=y[holdout], cat_features=categorical,
        feature_names=list(features.columns),
        baseline=base_logit[holdout] if residual_mode else None,
    )
    model = CatBoostClassifier(**parameters)
model.fit(fit_pool, eval_set=holdout_pool, use_best_model=True)
trees = max(model.tree_count_, 1)
holdout_scores = (
    model.predict(holdout_pool)
    if ranker_mode else model.predict(holdout_pool, prediction_type="RawFormulaVal")
)
holdout_metrics = runner.evaluate_module.evaluate(
    users[holdout].tolist(), y[holdout], holdout_scores,
)
holdout_base = frame["base_rank_fraction"].to_numpy()[holdout]
holdout_blends = []
for alpha in np.arange(0.0, 1.001, 0.02):
    score = (1.0 - alpha) * holdout_base + alpha * holdout_scores
    metrics = runner.evaluate_module.evaluate(users[holdout].tolist(), y[holdout], score)
    holdout_blends.append((float(metrics["primary"]), float(alpha), metrics))
holdout_best = max(holdout_blends, key=lambda value: value[0])
print("HOLDOUT", holdout_metrics, "BLEND", holdout_best, "TREES", trees, flush=True)

final_parameters = dict(parameters)
final_parameters.update({"iterations": trees, "od_type": None, "od_wait": None, "verbose": 0})
final = CatBoostRanker(**final_parameters) if ranker_mode else CatBoostClassifier(
    **final_parameters
)
final_indices = query_order(meta_indices) if ranker_mode else meta_indices
final.fit(Pool(
    features.iloc[final_indices], label=y[final_indices],
    cat_features=categorical, feature_names=list(features.columns),
    group_id=users[final_indices] if ranker_mode else None,
    baseline=(
        base_logit[final_indices]
        if ranker_mode or residual_mode else None
    ),
))
valid_pool = Pool(
    features.iloc[valid_indices], cat_features=categorical,
    feature_names=list(features.columns),
    baseline=(
        base_logit[valid_indices]
        if ranker_mode or residual_mode else None
    ),
)
scores = (
    final.predict(valid_pool)
    if ranker_mode else final.predict(valid_pool, prediction_type="RawFormulaVal")
).astype(np.float32)
metrics = runner.evaluate_module.evaluate(valid_users, valid_y, scores)
np.savez_compressed(
    ROOT / "runtime" / (
        f"batch-slate-meta-catboost-ranker-{args.ranker_objective}-"
        f"s{args.ranker_seed}.npz"
        if ranker_mode else (
            "batch-slate-meta-catboost-residual-s751.npz"
            if residual_mode else "batch-slate-meta-catboost-s751.npz"
        )
    ), scores=scores,
    holdout_scores=holdout_scores, holdout_alpha=holdout_best[1], trees=trees,
)
final.save_model(ROOT / "runtime" / (
    f"batch-slate-meta-catboost-ranker-{args.ranker_objective}-"
    f"s{args.ranker_seed}.cbm"
    if ranker_mode else (
        "batch-slate-meta-catboost-residual-s751.cbm"
        if residual_mode else "batch-slate-meta-catboost-s751.cbm"
    )
))
print("VALID", metrics, "RESOURCE_USAGE", tracker.finish(), flush=True)
