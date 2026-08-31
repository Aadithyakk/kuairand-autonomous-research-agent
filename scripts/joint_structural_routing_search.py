#!/usr/bin/env python3
"""Joint retuning of high-margin and active-day structural routing gates."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
CHAMPION_ROOT = ROOT
DATA = CHAMPION_ROOT / "external" / "KuaiRand-Pure" / "data"
RUNTIME = CHAMPION_ROOT / "runtime"
PRE_BATCH = HERE / "pre_batch_structural_scores.npz"
IMPROVED_CHAMPION = HERE / "joint_terminal_gate_search_scores.npz"
REPORT = HERE / "joint_structural_routing_search_results.json"
OUTPUT = HERE / "joint_structural_routing_search_scores.npz"

sys.path.insert(0, str(SCRIPT_DIR))
from joint_terminal_gate_search import (  # noqa: E402
    CONFIDENCE_SOURCES,
    apply_terminal,
    exact_metrics,
    factorize,
    fast_metrics,
    ordering_audit,
    rank_fraction,
    rank_ordinal,
)


CURRENT = (0.34, -0.5225, -0.175)
HIGH_GRID = [0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]
TWO_DAY_GRID = [-0.65, -0.60, -0.55, -0.5225, -0.50, -0.45, -0.40]
THREE_FOUR_GRID = [-0.25, -0.225, -0.20, -0.175, -0.15, -0.125, -0.10]
TERMINAL_WEIGHTS = (0.7717, -0.0275, -0.0922)


def load_rank(name: str, codes: np.ndarray, counts: np.ndarray, key: str = "scores") -> np.ndarray:
    with np.load(RUNTIME / name) as archive:
        values = np.asarray(archive[key], dtype=np.float64)
    return rank_ordinal(values, codes, counts)


def user_standardize(values: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sums = np.bincount(codes, weights=values, minlength=len(counts))
    squares = np.bincount(codes, weights=values * values, minlength=len(counts))
    means = sums / counts
    variance = np.maximum(squares / counts - means * means, 0.0)
    scale = np.maximum(np.sqrt(variance), 1e-8)
    return (values - means[codes]) / scale[codes]


def normalized_top_margin(values: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float64)
    for code in range(len(counts)):
        indices = np.flatnonzero(codes == code)
        if len(indices) < 2:
            continue
        ordered = indices[np.argsort(-values[indices], kind="stable")]
        output[indices] = (
            values[ordered[0]] - values[ordered[1]]
        ) / max(float(values[indices].std()), 1e-8)
    return output


def build_static_inputs(
    rows: pd.DataFrame, users: np.ndarray, codes: np.ndarray, counts: np.ndarray,
) -> dict:
    batch = load_rank("batch-slate-meta-catboost-s751.npz", codes, counts)
    ordered = load_rank(
        "history-catboost-classifier-probe-i250-seq40-plain-s151.npz",
        codes, counts,
    )
    watch = load_rank(
        "history-deepfm-k16-h128-lr0.001-session-fullmeta-watchratio0.4-s293.npz",
        codes, counts,
    )
    neighbor = load_rank(
        "user-neighbor-cf-n60.npz", codes, counts,
        key="positive_p2.0_s8.0",
    )
    user_balanced = load_rank(
        "history-catboost-classifier-probe-i500-seq40-session-ub0.5-s239.npz",
        codes, counts,
    )
    yeti = load_rank(
        "batch-slate-meta-catboost-ranker-YetiRankPairwise-s787.npz",
        codes, counts,
    )
    shallow = load_rank(
        "batch-slate-meta-catboost-ranker-YetiRankPairwise-s821.npz",
        codes, counts,
    )

    fractional_sources = []
    for name in CONFIDENCE_SOURCES:
        with np.load(RUNTIME / name) as archive:
            values = np.asarray(archive["scores"], dtype=np.float64)
        fractional_sources.append(rank_fraction(values, codes, counts))
    source_matrix = np.stack(fractional_sources)
    mean_source = rank_ordinal(source_matrix.mean(axis=0), codes, counts)
    median_source = rank_ordinal(np.median(source_matrix, axis=0), codes, counts)

    pair_counts = Counter(zip(users.tolist(), rows["video_id"].astype(str).tolist()))
    log_user_video = np.asarray(
        [np.log1p(pair_counts[(user, str(video))]) for user, video in zip(users, rows["video_id"])],
        dtype=np.float64,
    )
    log_user_video = user_standardize(log_user_video, codes, counts)

    active_days = np.zeros(len(rows), dtype=np.int16)
    group_size = counts[codes].astype(np.int16)
    single_gate = np.zeros(len(rows), dtype=np.float64)
    terminal_median_gate = np.zeros(len(rows), dtype=np.float64)
    for code in range(len(counts)):
        indices = np.flatnonzero(codes == code)
        active_days[indices] = rows.loc[indices, "date"].nunique()
        ordered_indices = indices[
            np.argsort(rows.loc[indices, "time_ms"].to_numpy(), kind="stable")
        ]
        times = rows.loc[ordered_indices, "time_ms"].to_numpy(dtype=np.int64)
        sessions = 1 + int(np.sum(np.diff(times) > 1_800_000))
        single_gate[indices] = float(sessions == 1)
        terminal_median_gate[indices] = float(2 <= sessions <= 3)

    training = pd.read_csv(
        DATA / "log_standard_4_08_to_4_21_pure.csv",
        usecols=["user_id", "video_id", "date", "long_view"],
        dtype={"user_id": "string", "video_id": "string"},
    )
    user_stats = training.groupby("user_id", sort=False)["long_view"].agg(["sum", "size"])
    prior = float(training["long_view"].mean())
    user_sum = rows["user_id"].map(user_stats["sum"]).fillna(0).to_numpy(dtype=float)
    user_count = rows["user_id"].map(user_stats["size"]).fillna(0).to_numpy(dtype=float)
    training_rate = (user_sum + 10.0 * prior) / (user_count + 10.0)
    low_rate_threshold = float(np.quantile(training_rate, 0.25))

    valid_pairs = set(zip(users.tolist(), rows["video_id"].astype(str).tolist()))
    seen_users = {
        str(user) for user, video in zip(training["user_id"], training["video_id"])
        if (str(user), str(video)) in valid_pairs
    }
    prior_video_gate = np.asarray(
        [float(user in seen_users) for user in users], dtype=np.float64
    )

    recent_counts = (
        training.loc[training["date"] >= 20220419]
        .groupby("user_id", sort=False)
        .size()
    )
    recent_rows = rows["user_id"].map(recent_counts).fillna(0).to_numpy(dtype=float)
    recent_threshold = float(np.quantile(recent_rows, 0.75))

    return {
        "batch": batch,
        "ordered": ordered,
        "watch": watch,
        "neighbor": neighbor,
        "user_balanced": user_balanced,
        "yeti": yeti,
        "shallow": shallow,
        "mean_source": mean_source,
        "median_source": median_source,
        "log_user_video": log_user_video,
        "two_day_gate": (active_days == 2).astype(np.float64),
        "three_four_gate": ((active_days >= 3) & (active_days <= 4)).astype(np.float64),
        "medium_slate_gate": ((group_size >= 6) & (group_size <= 10)).astype(np.float64),
        "low_rate_gate": (training_rate <= low_rate_threshold).astype(np.float64),
        "prior_video_gate": prior_video_gate,
        "single_gate": single_gate,
        "terminal_median_gate": terminal_median_gate,
        "coverage_gate": (recent_rows >= recent_threshold).astype(np.float64),
    }


def main() -> None:
    rows = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=["user_id", "video_id", "date", "time_ms", "long_view"],
        dtype={"user_id": "string", "video_id": "string"},
    )
    rows = rows.loc[rows["date"] <= 20220428].reset_index(drop=True)
    users = rows["user_id"].astype(str).to_numpy()
    labels = rows["long_view"].to_numpy(dtype=np.float64)
    codes, _, counts = factorize(users)
    with np.load(PRE_BATCH) as archive:
        pre_batch = np.asarray(archive["scores"], dtype=np.float64)
    with np.load(IMPROVED_CHAMPION) as archive:
        improved = np.asarray(archive["selected"], dtype=np.float64)
    x = build_static_inputs(rows, users, codes, counts)

    # Stages through the joint batch-slate ordinal consensus are fixed.
    joint = pre_batch - 0.05 * x["log_user_video"] + 0.0021875 * x["batch"]
    structural_base = pre_batch + 0.2675 * rank_ordinal(joint, codes, counts)
    structural_base_rank = rank_ordinal(structural_base, codes, counts)
    margin = normalized_top_margin(structural_base, codes, counts)
    positive_margins = margin[margin > 0]
    high_gate = (margin >= np.quantile(positive_margins, 0.75)).astype(np.float64)

    def score(weights: tuple[float, float, float]) -> np.ndarray:
        high_weight, two_day_weight, three_four_weight = weights
        stage = structural_base + high_weight * high_gate * (
            x["ordered"] - structural_base_rank
        )
        # This confidence gate is recomputed after the high-margin correction.
        stage_margin = normalized_top_margin(stage, codes, counts)
        positive = stage_margin[stage_margin > 0]
        median_gate = (stage_margin >= np.quantile(positive, 0.50)).astype(float)
        stage = stage + 0.01 * median_gate * (
            x["mean_source"] - rank_ordinal(stage, codes, counts)
        )
        stage = stage + two_day_weight * x["two_day_gate"] * (
            x["ordered"] - rank_ordinal(stage, codes, counts)
        )
        stage = stage + three_four_weight * x["three_four_gate"] * (
            x["watch"] - rank_ordinal(stage, codes, counts)
        )
        stage = stage + 0.01 * x["medium_slate_gate"] * (
            x["batch"] - rank_ordinal(stage, codes, counts)
        )
        stage = stage + 0.135 * x["low_rate_gate"] * (
            x["watch"] - rank_ordinal(stage, codes, counts)
        )
        stage = stage - 0.03 * x["prior_video_gate"] * (
            x["batch"] - rank_ordinal(stage, codes, counts)
        )
        stage = (
            stage + 0.002 * x["neighbor"]
            + 0.001875 * x["user_balanced"]
            + 0.001875 * x["watch"]
        )
        stage_rank = rank_ordinal(stage, codes, counts)
        return apply_terminal(
            stage, stage_rank, x["yeti"], x["median_source"], x["shallow"],
            x["single_gate"], x["terminal_median_gate"], x["coverage_gate"],
            codes, counts, TERMINAL_WEIGHTS,
        )

    reconstruction = score(CURRENT)
    champion_exact = exact_metrics(users, labels, improved)
    reconstruction_exact = exact_metrics(users, labels, reconstruction)
    if max(abs(champion_exact[key] - reconstruction_exact[key]) for key in champion_exact) > 3e-6:
        raise RuntimeError(
            f"Structural reconstruction drifted: {reconstruction_exact} vs {champion_exact}"
        )
    baseline_fast = fast_metrics(labels, reconstruction, codes, counts)
    fold_ids = np.asarray([int(user) % 4 for user in users], dtype=np.int8)
    fold_cache = {}
    for fold in range(4):
        mask = fold_ids == fold
        fc, _, fn = factorize(users[mask])
        fold_cache[fold] = (mask, fc, fn, fast_metrics(labels[mask], reconstruction[mask], fc, fn))

    best_global = None
    promising = []
    trials = 0
    for high_weight in HIGH_GRID:
        for two_weight in TWO_DAY_GRID:
            for three_weight in THREE_FOUR_GRID:
                trials += 1
                weights = (high_weight, two_weight, three_weight)
                candidate = score(weights)
                candidate_metrics = fast_metrics(labels, candidate, codes, counts)
                row = {
                    "weights": list(weights),
                    "metrics": candidate_metrics,
                    "delta": {key: candidate_metrics[key] - baseline_fast[key] for key in baseline_fast},
                }
                if best_global is None or candidate_metrics["primary"] > best_global["metrics"]["primary"]:
                    best_global = row
                if (
                    candidate_metrics["primary"] > baseline_fast["primary"]
                    and candidate_metrics["gauc"] >= baseline_fast["gauc"]
                    and candidate_metrics["ndcg5"] >= baseline_fast["ndcg5"]
                ):
                    stable = True
                    fold_rows = []
                    for fold, (mask, fc, fn, fold_base) in fold_cache.items():
                        fm = fast_metrics(labels[mask], candidate[mask], fc, fn)
                        fd = {key: fm[key] - fold_base[key] for key in fold_base}
                        stable &= all(value >= -1e-12 for value in fd.values())
                        fold_rows.append({"fold": fold, "delta": fd})
                    row["stable"] = bool(stable)
                    row["folds"] = fold_rows
                    promising.append(row)

    stable_rows = [row for row in promising if row["stable"]]
    selected = None
    selected_scores = reconstruction
    if stable_rows:
        best_primary = max(row["metrics"]["primary"] for row in stable_rows)
        plateau = [row for row in stable_rows if abs(row["metrics"]["primary"] - best_primary) <= 1e-12]
        selected = min(
            plateau,
            key=lambda row: sum(abs(value - current) for value, current in zip(row["weights"], CURRENT)),
        )
        selected_scores = score(tuple(selected["weights"]))
        selected["exact_metrics"] = exact_metrics(users, labels, selected_scores)
        selected["exact_delta"] = {
            key: selected["exact_metrics"][key] - champion_exact[key]
            for key in champion_exact
        }

    audit = ordering_audit(codes, counts, reconstruction, selected_scores)
    np.savez_compressed(
        OUTPUT,
        champion=improved.astype(np.float32),
        reconstruction=reconstruction.astype(np.float32),
        selected=selected_scores.astype(np.float32),
    )
    report = {
        "experiment": "joint high-margin and active-day routing retune",
        "decision": "promote" if selected is not None else "retain_improved_champion",
        "current_weights": list(CURRENT),
        "terminal_weights_fixed": list(TERMINAL_WEIGHTS),
        "grids": {"high_margin": HIGH_GRID, "two_day": TWO_DAY_GRID, "three_four_day": THREE_FOUR_GRID},
        "trials": trials,
        "champion_exact": champion_exact,
        "reconstruction_exact": reconstruction_exact,
        "best_global": best_global,
        "globally_promising_count": len(promising),
        "stable_candidate_count": len(stable_rows),
        "selected": selected,
        "ordering_audit": audit,
        "hidden_test_accessed": False,
        "artifacts": {"report": str(REPORT), "scores": str(OUTPUT)},
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "decision": report["decision"],
        "trials": trials,
        "champion": champion_exact,
        "reconstruction": reconstruction_exact,
        "best_global": best_global,
        "globally_promising_count": len(promising),
        "stable_candidate_count": len(stable_rows),
        "selected": selected,
        "ordering_audit": audit,
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
