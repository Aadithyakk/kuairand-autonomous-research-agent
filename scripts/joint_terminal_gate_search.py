#!/usr/bin/env python3
"""Joint local retuning of the champion's final three routing corrections."""
from __future__ import annotations

import json
import os
import math
import sys
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
PRETERMINAL = HERE / "pre_terminal_routing_scores.npz"
CHAMPION = HERE / "champion_scores_apr22_28.npz"
REPORT = HERE / "joint_terminal_gate_search_results.json"
OUTPUT = HERE / "joint_terminal_gate_search_scores.npz"

sys.path.insert(0, str(CHAMPION_ROOT))
from scripts import kuairand_runner as runner  # noqa: E402


CONFIDENCE_SOURCES = [
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
    "session1-sessionx1-sessionw0-multir0-trend0-affinity0-ctx0-transition1.npz",
]

W1_GRID = [round(0.770 + 0.0001 * index, 6) for index in range(31)]
W2_GRID = [-0.0275]
W3_GRID = [round(-0.0925 + 0.0001 * index, 6) for index in range(11)]
CURRENT_WEIGHTS = (0.685, -0.0275, -0.0925)


def factorize(users: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes, unique = pd.factorize(users, sort=False)
    counts = np.bincount(codes)
    return codes.astype(np.int32), np.asarray(unique), counts


def rank_ordinal(values: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Stable within-user ordinal z-ranks, matching verify_slate_consensus."""
    n = len(values)
    index = np.arange(n, dtype=np.int64)
    order = np.lexsort((index, np.asarray(values, dtype=np.float64), codes))
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    starts = np.repeat(boundaries, counts)
    local_rank = np.arange(n, dtype=np.float64) - starts
    means = (counts - 1.0) / 2.0
    standard_deviation = np.sqrt(np.maximum((counts.astype(float) ** 2 - 1.0) / 12.0, 0.0))
    standard_deviation = np.maximum(standard_deviation, 1e-8)
    z_sorted = (local_rank - means[sorted_codes]) / standard_deviation[sorted_codes]
    output = np.empty(n, dtype=np.float64)
    output[order] = z_sorted
    return output


def rank_fraction(values: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n = len(values)
    index = np.arange(n, dtype=np.int64)
    order = np.lexsort((index, np.asarray(values, dtype=np.float64), codes))
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    starts = np.repeat(boundaries, counts)
    local_rank = np.arange(n, dtype=np.float64) - starts
    denominator = np.maximum(counts[sorted_codes] - 1, 1)
    output = np.empty(n, dtype=np.float64)
    output[order] = local_rank / denominator
    return output


def fast_metrics(labels: np.ndarray, scores: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> dict:
    """Vectorized equivalent of the official no-tie evaluator."""
    n = len(scores)
    index = np.arange(n, dtype=np.int64)
    ascending = np.lexsort((index, np.asarray(scores, dtype=np.float64), codes))
    sorted_codes = codes[ascending]
    boundaries = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    starts = np.repeat(boundaries, counts)
    local_rank_one = np.arange(n, dtype=np.float64) - starts + 1.0
    sorted_labels = labels[ascending]
    positive_count = np.bincount(codes, weights=labels, minlength=len(counts))
    negative_count = counts - positive_count
    positive_rank_sum = np.bincount(
        sorted_codes,
        weights=local_rank_one * sorted_labels,
        minlength=len(counts),
    )
    valid = (positive_count > 0) & (negative_count > 0)
    auc = np.full(len(counts), 0.5, dtype=np.float64)
    auc[valid] = (
        positive_rank_sum[valid]
        - positive_count[valid] * (positive_count[valid] + 1.0) / 2.0
    ) / (positive_count[valid] * negative_count[valid])
    gauc = float(np.sum(positive_count[valid] * auc[valid]) / np.sum(positive_count[valid]))

    descending = np.lexsort((index, -np.asarray(scores, dtype=np.float64), codes))
    desc_codes = codes[descending]
    desc_boundaries = np.flatnonzero(np.r_[True, desc_codes[1:] != desc_codes[:-1]])
    desc_starts = np.repeat(desc_boundaries, counts)
    positions = np.arange(n, dtype=np.int64) - desc_starts
    top = positions < 5
    discount = np.zeros(n, dtype=np.float64)
    discount[top] = 1.0 / np.log2(positions[top] + 2.0)
    dcg = np.bincount(
        desc_codes,
        weights=labels[descending] * discount,
        minlength=len(counts),
    )
    discount5 = np.asarray([1.0 / math.log2(position + 2.0) for position in range(5)])
    cumulative = np.r_[0.0, np.cumsum(discount5)]
    idcg = cumulative[np.minimum(positive_count.astype(int), 5)]
    ndcg_by_user = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    ndcg = float(ndcg_by_user.mean())
    return {"primary": 0.5 * (gauc + ndcg), "gauc": gauc, "ndcg5": ndcg}


def exact_metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict:
    result = runner.evaluate_module.evaluate(
        users.tolist(), labels.astype(np.float32), scores.astype(np.float32)
    )
    return {
        "primary": float(result["primary"]),
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
    }


def ordering_audit(
    codes: np.ndarray, counts: np.ndarray,
    control: np.ndarray, candidate: np.ndarray,
) -> dict:
    changed_pairs = 0
    changed_users = 0
    top5_membership_users = 0
    for code in range(len(counts)):
        indices = np.flatnonzero(codes == code)
        old_order = indices[np.argsort(-control[indices], kind="stable")]
        new_order = indices[np.argsort(-candidate[indices], kind="stable")]
        old_position = {int(index): position for position, index in enumerate(old_order)}
        new_position = {int(index): position for position, index in enumerate(new_order)}
        local_changes = 0
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1:]:
                local_changes += int(
                    (old_position[int(left)] < old_position[int(right)])
                    != (new_position[int(left)] < new_position[int(right)])
                )
        changed_pairs += local_changes
        changed_users += int(local_changes > 0)
        top5_membership_users += int(
            set(old_order[:5].tolist()) != set(new_order[:5].tolist())
        )
    return {
        "changed_pair_orderings": changed_pairs,
        "changed_users": changed_users,
        "users_with_changed_top5_membership": top5_membership_users,
    }


def load_rank(name: str, codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    with np.load(RUNTIME / name) as archive:
        values = np.asarray(archive["scores"], dtype=np.float64)
    return rank_ordinal(values, codes, counts)


def apply_terminal(
    preterminal: np.ndarray,
    preterminal_rank: np.ndarray,
    yeti: np.ndarray,
    median: np.ndarray,
    shallow: np.ndarray,
    single_gate: np.ndarray,
    median_gate: np.ndarray,
    coverage_gate: np.ndarray,
    codes: np.ndarray,
    counts: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    w1, w2, w3 = weights
    stage1 = preterminal + w1 * single_gate * (yeti - preterminal_rank)
    stage1_rank = rank_ordinal(stage1, codes, counts)
    stage2 = stage1 + w2 * median_gate * (median - stage1_rank)
    stage2_rank = rank_ordinal(stage2, codes, counts)
    return stage2 + w3 * coverage_gate * (shallow - stage2_rank)


def main() -> None:
    rows = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=["user_id", "date", "time_ms", "long_view"],
        dtype={"user_id": "string"},
    )
    rows = rows.loc[rows["date"] <= 20220428].reset_index(drop=True)
    users = rows["user_id"].astype(str).to_numpy()
    labels = rows["long_view"].to_numpy(dtype=np.float64)
    codes, _, counts = factorize(users)
    with np.load(PRETERMINAL) as archive:
        preterminal = np.asarray(archive["scores"], dtype=np.float64)
    with np.load(CHAMPION) as archive:
        champion = np.asarray(archive["scores"], dtype=np.float64)
    preterminal_rank = rank_ordinal(preterminal, codes, counts)
    yeti = load_rank("batch-slate-meta-catboost-ranker-YetiRankPairwise-s787.npz", codes, counts)
    shallow = load_rank("batch-slate-meta-catboost-ranker-YetiRankPairwise-s821.npz", codes, counts)

    fractional_sources = []
    for name in CONFIDENCE_SOURCES:
        with np.load(RUNTIME / name) as archive:
            values = np.asarray(archive["scores"], dtype=np.float64)
        fractional_sources.append(rank_fraction(values, codes, counts))
    median_fraction = np.median(np.stack(fractional_sources), axis=0)
    median = rank_ordinal(median_fraction, codes, counts)

    single_gate = np.zeros(len(rows), dtype=np.float64)
    median_gate = np.zeros(len(rows), dtype=np.float64)
    for code in range(len(counts)):
        indices = np.flatnonzero(codes == code)
        ordered = indices[np.argsort(rows.loc[indices, "time_ms"].to_numpy(), kind="stable")]
        times = rows.loc[ordered, "time_ms"].to_numpy(dtype=np.int64)
        session_count = 1 + int(np.sum(np.diff(times) > 1_800_000))
        single_gate[indices] = float(session_count == 1)
        median_gate[indices] = float(2 <= session_count <= 3)

    training = pd.read_csv(
        DATA / "log_standard_4_08_to_4_21_pure.csv",
        usecols=["user_id", "date"],
        dtype={"user_id": "string"},
    )
    recent_counts = (
        training.loc[training["date"] >= 20220419]
        .groupby("user_id", sort=False)
        .size()
    )
    recent_rows = rows["user_id"].map(recent_counts).fillna(0).to_numpy(dtype=np.float64)
    coverage_threshold = float(np.quantile(recent_rows, 0.75))
    coverage_gate = (recent_rows >= coverage_threshold).astype(np.float64)

    reconstruction = apply_terminal(
        preterminal, preterminal_rank, yeti, median, shallow,
        single_gate, median_gate, coverage_gate, codes, counts, CURRENT_WEIGHTS,
    )
    champion_exact = exact_metrics(users, labels, champion)
    reconstruction_exact = exact_metrics(users, labels, reconstruction)
    if max(abs(reconstruction_exact[key] - champion_exact[key]) for key in champion_exact) > 2e-6:
        raise RuntimeError(
            f"Terminal reconstruction drifted: {reconstruction_exact} vs {champion_exact}"
        )

    baseline_fast = fast_metrics(labels, reconstruction, codes, counts)
    fold_ids = np.asarray([int(user) % 4 for user in users], dtype=np.int8)
    fold_cache = {}
    for fold in range(4):
        mask = fold_ids == fold
        fold_codes, _, fold_counts = factorize(users[mask])
        fold_cache[fold] = (
            mask,
            fold_codes,
            fold_counts,
            fast_metrics(labels[mask], reconstruction[mask], fold_codes, fold_counts),
        )

    globally_promising = []
    best_global = None
    trials = 0
    for w1 in W1_GRID:
        stage1 = preterminal + w1 * single_gate * (yeti - preterminal_rank)
        stage1_rank = rank_ordinal(stage1, codes, counts)
        for w2 in W2_GRID:
            stage2 = stage1 + w2 * median_gate * (median - stage1_rank)
            stage2_rank = rank_ordinal(stage2, codes, counts)
            for w3 in W3_GRID:
                trials += 1
                candidate = stage2 + w3 * coverage_gate * (shallow - stage2_rank)
                candidate_metrics = fast_metrics(labels, candidate, codes, counts)
                row = {
                    "weights": [w1, w2, w3],
                    "metrics": candidate_metrics,
                    "delta": {
                        key: candidate_metrics[key] - baseline_fast[key]
                        for key in baseline_fast
                    },
                }
                if best_global is None or candidate_metrics["primary"] > best_global["metrics"]["primary"]:
                    best_global = row
                if (
                    candidate_metrics["primary"] > baseline_fast["primary"]
                    and candidate_metrics["gauc"] >= baseline_fast["gauc"]
                    and candidate_metrics["ndcg5"] >= baseline_fast["ndcg5"]
                ):
                    fold_rows = []
                    stable = True
                    for fold, (mask, fold_codes, fold_counts, fold_base) in fold_cache.items():
                        fold_candidate = fast_metrics(
                            labels[mask], candidate[mask], fold_codes, fold_counts
                        )
                        fold_delta = {
                            key: fold_candidate[key] - fold_base[key]
                            for key in fold_base
                        }
                        fold_rows.append({"fold": fold, "delta": fold_delta})
                        stable &= all(fold_delta[key] >= -1e-12 for key in fold_delta)
                    row["folds"] = fold_rows
                    row["stable"] = bool(stable)
                    globally_promising.append(row)

    stable_candidates = [row for row in globally_promising if row["stable"]]
    selected = None
    selected_scores = reconstruction
    if stable_candidates:
        best_stable_primary = max(
            row["metrics"]["primary"] for row in stable_candidates
        )
        primary_plateau = [
            row for row in stable_candidates
            if abs(row["metrics"]["primary"] - best_stable_primary) <= 1e-12
        ]
        selected = min(
            primary_plateau,
            key=lambda row: sum(
                abs(value - current)
                for value, current in zip(row["weights"], CURRENT_WEIGHTS)
            ),
        )
        selected_scores = apply_terminal(
            preterminal, preterminal_rank, yeti, median, shallow,
            single_gate, median_gate, coverage_gate, codes, counts,
            tuple(selected["weights"]),
        )
        selected["exact_metrics"] = exact_metrics(users, labels, selected_scores)
        selected["exact_delta_vs_champion"] = {
            key: selected["exact_metrics"][key] - champion_exact[key]
            for key in champion_exact
        }

    np.savez_compressed(
        OUTPUT,
        champion=champion.astype(np.float32),
        reconstruction=reconstruction.astype(np.float32),
        selected=selected_scores.astype(np.float32),
    )
    ordering_changes = ordering_audit(
        codes, counts, reconstruction, selected_scores
    )
    report = {
        "experiment": "joint local retuning of three terminal champion gates",
        "decision": "promote" if selected is not None else "retain_frozen_champion",
        "current_weights": list(CURRENT_WEIGHTS),
        "grid": {"single_session_yeti": W1_GRID, "session_median": W2_GRID, "coverage_yeti": W3_GRID},
        "trials": trials,
        "champion_exact": champion_exact,
        "reconstruction_exact": reconstruction_exact,
        "reconstruction_fast": baseline_fast,
        "best_global": best_global,
        "globally_promising_count": len(globally_promising),
        "stable_candidate_count": len(stable_candidates),
        "selected": selected,
        "ordering_audit": ordering_changes,
        "protocol": {
            "gate": "both aggregate metrics and every metric in every actual-user-ID fold must be nonnegative",
            "zero_preference": "retain existing weights unless a strictly higher primary passes the gate",
            "new_model_training": False,
            "hidden_test_accessed": False,
        },
        "artifacts": {"scores": str(OUTPUT), "report": str(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "decision": report["decision"],
        "trials": trials,
        "champion": champion_exact,
        "reconstruction": reconstruction_exact,
        "best_global": best_global,
        "globally_promising_count": len(globally_promising),
        "stable_candidate_count": len(stable_candidates),
        "selected": selected,
        "ordering_audit": ordering_changes,
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
