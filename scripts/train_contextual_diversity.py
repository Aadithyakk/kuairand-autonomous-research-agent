#!/usr/bin/env python3
"""Leak-free contextual-diversity reranking screen for KuaiRand-Pure.

A frozen DeepFM score stream, trained only through 2022-04-13, is reranked
with a small MMR-style redundancy penalty.  Lambda is selected on April 14
and then locked for a single April 15-21 screen.  This script deliberately
does not open any standard-log file containing April 22 or later labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from backend.kuailab.champion import load_champion_scores
from scripts import kuairand_runner as runner


SOURCE_CACHE = (
    ROOT / "runtime" / "stacked-reranker"
    / "out-of-time-components-cutoff-20220413.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "parallel-methods" / "contextual-diversity-screen.json"
)
DEFAULT_SCORES = (
    ROOT / "runtime" / "parallel-contextual-diversity" / "screen-scores.npz"
)
CONFIRMATION_OUTPUT = (
    ROOT / "results" / "parallel-methods"
    / "contextual-diversity-confirmation.json"
)
CONFIRMATION_SCORES = (
    ROOT / "runtime" / "parallel-contextual-diversity"
    / "confirmation-scores.npz"
)
LAMBDA_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
LOCKED_LAMBDA = 0.08
LOCKED_POOL_SIZE = 10


def metric_record(users: Sequence[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    result = runner.evaluate_module.evaluate(users, labels, scores)
    return {
        "primary": float(result["primary"]),
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "users": int(result["users"]),
        "rows": int(result["rows"]),
    }


def metric_delta(candidate: dict, control: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - control[key])
        for key in ("primary", "gauc", "ndcg5")
    }


def fractional_user_rank(users: Sequence[str], scores: np.ndarray) -> np.ndarray:
    """Return stable within-user ranks in (0, 1), where larger is better."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(users) != len(values):
        raise ValueError(f"User/score alignment failed: {len(users)} != {len(values)}")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[str(user)].append(index)
    output = np.empty(len(values), dtype=np.float64)
    for raw_indices in groups.values():
        indices = np.asarray(raw_indices, dtype=np.int64)
        order = np.argsort(values[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[order] = np.arange(len(indices), dtype=np.float64)
        output[indices] = (ranks + 0.5) / len(indices)
    return output


def load_rows(data_dir: Path) -> list[tuple]:
    """Load April 8-21 rows and static, outcome-free video metadata."""
    metadata: dict[str, tuple[str, str, str, frozenset[str]]] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_tags = row.get("tag") or "UNK"
            metadata[row["video_id"]] = (
                row.get("author_id") or "UNK",
                row.get("music_id") or "UNK",
                row.get("video_type") or "UNK",
                frozenset(token for token in raw_tags.split(",") if token),
            )

    rows = []
    # This is the only interaction-log file opened. It ends on April 21.
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            author, music, video_type, tags = metadata.get(
                row["video_id"], ("UNK", "UNK", "UNK", frozenset({"UNK"}))
            )
            rows.append((
                int(row["date"]), str(row["user_id"]), str(row["video_id"]),
                author, music, video_type, tags,
                1 if row["long_view"] != "0" else 0,
            ))
    if len(rows) != 1_141_112:
        raise RuntimeError(f"Unexpected April 8-21 row count: {len(rows)}")
    return rows


def load_confirmation_rows(data_dir: Path) -> list[tuple]:
    """Load April 22-28 while never parsing April 29+ outcome fields."""
    metadata: dict[str, tuple[str, str, str, frozenset[str]]] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_tags = row.get("tag") or "UNK"
            metadata[row["video_id"]] = (
                row.get("author_id") or "UNK",
                row.get("music_id") or "UNK",
                row.get("video_type") or "UNK",
                frozenset(token for token in raw_tags.split(",") if token),
            )

    rows = []
    later_log = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    with later_log.open(encoding="utf-8") as stream:
        header = next(stream)
        fieldnames = next(csv.reader([header]))
        if fieldnames[:3] != ["user_id", "video_id", "date"]:
            raise RuntimeError(f"Unexpected later-log header: {fieldnames[:3]}")
        for line in stream:
            # The file is not sorted by date. Split only through the third
            # comma first, so later-date outcome fields are never parsed.
            prefix = line.split(",", 3)
            if len(prefix) != 4:
                raise RuntimeError("Malformed interaction row")
            date = int(prefix[2])
            if date > 20220428:
                continue
            if date < 20220422:
                raise RuntimeError(f"Unexpected earlier date in later log: {date}")
            values = next(csv.reader([line]))
            row = dict(zip(fieldnames, values, strict=True))
            author, music, video_type, tags = metadata.get(
                row["video_id"], ("UNK", "UNK", "UNK", frozenset({"UNK"}))
            )
            rows.append((
                date, str(row["user_id"]), str(row["video_id"]), author,
                music, video_type, tags,
                1 if row["long_view"] != "0" else 0,
            ))
    if len(rows) != 124_909:
        raise RuntimeError(f"Unexpected April 22-28 row count: {len(rows)}")
    return rows


def user_fold(user: str) -> int:
    """Stable actual-user-ID fold assignment used by champion audits."""
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode("utf-8")).hexdigest()[:8], 16) % 4


def similarity(left: tuple, right: tuple) -> float:
    """Outcome-free similarity over author, music, type, and tag tokens."""
    author = 1.0 if left[3] == right[3] and left[3] != "UNK" else 0.0
    music = 1.0 if left[4] == right[4] and left[4] != "UNK" else 0.0
    video_type = 1.0 if left[5] == right[5] and left[5] != "UNK" else 0.0
    left_tags, right_tags = left[6], right[6]
    union = left_tags | right_tags
    tag_jaccard = len(left_tags & right_tags) / len(union) if union else 0.0
    return 0.35 * author + 0.20 * music + 0.10 * video_type + 0.35 * tag_jaccard


def rerank_mmr(
    rows: Sequence[tuple], users: Sequence[str], base_scores: np.ndarray,
    penalty: float, *, pool_size: int = 10,
) -> np.ndarray:
    """Greedily reorder only each user's top pool, preserving all other ranks."""
    base = fractional_user_rank(users, base_scores)
    if penalty == 0.0:
        return base.copy()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[str(user)].append(index)
    output = base.copy()
    for raw_indices in groups.values():
        indices = np.asarray(raw_indices, dtype=np.int64)
        descending = indices[np.argsort(-base[indices], kind="stable")]
        pool = descending[: min(pool_size, len(descending))].tolist()
        if len(pool) < 2:
            continue
        selected: list[int] = []
        while pool:
            best_index = pool[0]
            best_value = -np.inf
            for candidate in pool:
                redundancy = (
                    max(similarity(rows[candidate], rows[prior]) for prior in selected)
                    if selected else 0.0
                )
                value = base[candidate] - penalty * redundancy
                # Strict comparison preserves the original stable order on ties.
                if value > best_value:
                    best_value = value
                    best_index = candidate
            selected.append(best_index)
            pool.remove(best_index)
        original_values = np.sort(base[descending[: len(selected)]])[::-1]
        output[np.asarray(selected, dtype=np.int64)] = original_values
    return output


def select_lambda(grid: list[dict], *, minimum_gain: float = 1e-5) -> float:
    """Select only a material fit gain; otherwise explicitly prefer lambda zero."""
    control = next(item for item in grid if item["lambda"] == 0.0)
    eligible = [
        item for item in grid
        if item["metrics"]["primary"]
        >= control["metrics"]["primary"] + minimum_gain
    ]
    if not eligible:
        return 0.0
    # Deterministic and conservative: best primary, then smallest lambda.
    best = max(
        eligible,
        key=lambda item: (item["metrics"]["primary"], -item["lambda"]),
    )
    return float(best["lambda"])


def run_confirmation(
    output: Path, scores_output: Path, tracker: ProcessResourceTracker,
) -> int:
    """Apply the pre-registered screen choice once to April 22-28."""
    rows = load_confirmation_rows(ROOT / "external" / "KuaiRand-Pure" / "data")
    users = [row[1] for row in rows]
    # Match the organizer encoder's label dtype exactly; its pure-Python
    # evaluator inherits NumPy scalar arithmetic from these values.
    labels = np.asarray([row[7] for row in rows], dtype=np.float32)
    champion_scores, manifest = load_champion_scores(expected_rows=len(rows))
    control_rank_scores = fractional_user_rank(users, champion_scores)
    candidate_scores = rerank_mmr(
        rows, users, champion_scores, LOCKED_LAMBDA,
        pool_size=LOCKED_POOL_SIZE,
    )

    control = metric_record(users, labels, champion_scores)
    candidate = metric_record(users, labels, candidate_scores)
    global_delta = metric_delta(candidate, control)
    manifest_metrics = manifest.get("validation_metrics", {})
    alignment_matches_manifest = all(
        abs(control[key] - float(manifest_metrics[key])) <= 1e-9
        for key in ("primary", "gauc", "ndcg5")
    )
    if not alignment_matches_manifest:
        raise RuntimeError(
            "Frozen champion metric reproduction failed; refusing confirmation: "
            f"reproduced={control}, manifest={manifest_metrics}"
        )

    fold_ids = np.asarray([user_fold(user) for user in users], dtype=np.int8)
    user_array = np.asarray(users, dtype=object)
    folds = []
    for fold in range(4):
        mask = fold_ids == fold
        fold_users = user_array[mask].tolist()
        fold_control = metric_record(
            fold_users, labels[mask], champion_scores[mask]
        )
        fold_candidate = metric_record(
            fold_users, labels[mask], candidate_scores[mask]
        )
        delta = metric_delta(fold_candidate, fold_control)
        folds.append({
            "fold": fold,
            "user_assignment": "integer user_id modulo 4 (SHA-256 fallback)",
            "control": fold_control,
            "candidate": fold_candidate,
            "delta": delta,
            "all_metric_deltas_nonnegative": all(
                delta[key] >= -1e-12 for key in ("primary", "gauc", "ndcg5")
            ),
        })

    global_nonnegative = all(
        global_delta[key] >= -1e-12 for key in ("primary", "gauc", "ndcg5")
    )
    folds_nonnegative = all(
        item["all_metric_deltas_nonnegative"] for item in folds
    )
    passed = global_nonnegative and folds_nonnegative

    output.parent.mkdir(parents=True, exist_ok=True)
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scores_output,
        validation_dates=np.asarray([row[0] for row in rows], dtype=np.int32),
        control_scores=champion_scores.astype(np.float32),
        control_rank_scores=control_rank_scores.astype(np.float32),
        candidate_scores=candidate_scores.astype(np.float32),
        locked_lambda=np.float32(LOCKED_LAMBDA),
        locked_pool_size=np.int32(LOCKED_POOL_SIZE),
        validation_labels=labels.astype(np.float32),
    )
    report = {
        "experiment": "locked CDM/MMR contextual-diversity confirmation",
        "status": (
            "passed_four_fold_confirmation"
            if passed else "rejected_at_four_fold_confirmation"
        ),
        "protocol": {
            "screen_decision_source": "results/parallel-methods/contextual-diversity-screen.json",
            "locked_lambda": LOCKED_LAMBDA,
            "locked_pool_size": LOCKED_POOL_SIZE,
            "retuned_on_confirmation": False,
            "confirmation": "2022-04-22..2022-04-28",
            "later_log_access": (
                "The later log is not date-contiguous. Every line was filtered using "
                "only its first three identity/date fields; outcome fields were fully "
                "parsed only for April 22-28 and never for April 29+."
            ),
            "hidden_test_accessed": False,
            "outcome_free_similarity": (
                "0.35 author + 0.20 music + 0.10 video type + 0.35 tag Jaccard"
            ),
            "reranking_scope": (
                "greedy MMR within each user's top 10; lower ranks unchanged"
            ),
            "fold_protocol": (
                "four disjoint actual-user-ID folds; fixed candidate evaluated "
                "without any fold-specific tuning"
            ),
        },
        "data": {
            "rows": len(rows),
            "users": len(set(users)),
            "minimum_date": min(row[0] for row in rows),
            "maximum_date": max(row[0] for row in rows),
            "champion_scores": "results/final-model/validation-scores.npz",
        },
        "alignment_validation": {
            "exact_row_count_match": len(champion_scores) == len(rows) == 124_909,
            "champion_metrics_match_manifest": alignment_matches_manifest,
            "manifest_validation_metrics": manifest_metrics,
            "reproduced_champion_metrics": control,
        },
        "global": {
            "control": control,
            "candidate": candidate,
            "delta": global_delta,
            "all_metric_deltas_nonnegative": global_nonnegative,
            "changed_rank_rows": int(np.count_nonzero(
                candidate_scores != control_rank_scores
            )),
        },
        "folds": folds,
        "confirmation_gate": {
            "required": (
                "primary, GAUC, and nDCG@5 deltas must all be nonnegative "
                "globally and in every actual-user-ID fold"
            ),
            "global_all_metrics_nonnegative": global_nonnegative,
            "all_four_folds_all_metrics_nonnegative": folds_nonnegative,
            "passed": passed,
        },
        "recommendation": (
            "Eligible for champion integration review."
            if passed else
            "Reject the diversity reranker and retain the exact frozen champion."
        ),
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "scores": str(scores_output.resolve().relative_to(ROOT)),
            "report": str(output.resolve().relative_to(ROOT)),
        },
        "resource_usage": tracker.finish(),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--pool-size", type=int, default=10)
    args = parser.parse_args()
    tracker = ProcessResourceTracker()

    if args.confirmation:
        if args.pool_size != LOCKED_POOL_SIZE:
            raise ValueError(
                f"Confirmation pool size is locked at {LOCKED_POOL_SIZE}; "
                f"received {args.pool_size}"
            )
        return run_confirmation(
            args.output or CONFIRMATION_OUTPUT,
            args.scores_output or CONFIRMATION_SCORES,
            tracker,
        )

    args.output = args.output or DEFAULT_OUTPUT
    args.scores_output = args.scores_output or DEFAULT_SCORES

    all_rows = load_rows(ROOT / "external" / "KuaiRand-Pure" / "data")
    meta_rows = [row for row in all_rows if row[0] > 20220413]
    fit_mask = np.asarray([row[0] == 20220414 for row in meta_rows], dtype=bool)
    screen_mask = np.asarray([row[0] >= 20220415 for row in meta_rows], dtype=bool)
    if not fit_mask.any() or not screen_mask.any():
        raise RuntimeError("Expected both April 14 fit rows and April 15-21 screen rows")

    with np.load(SOURCE_CACHE, allow_pickle=False) as archive:
        if "deep" not in archive.files:
            raise KeyError("Aligned cache is missing the required DeepFM stream")
        base_all = np.asarray(archive["deep"], dtype=np.float64)
    if len(base_all) != len(meta_rows):
        raise RuntimeError(
            f"Artifact/date alignment failed: {len(base_all)} scores != {len(meta_rows)} rows"
        )

    fit_rows = [row for row, keep in zip(meta_rows, fit_mask, strict=True) if keep]
    screen_rows = [row for row, keep in zip(meta_rows, screen_mask, strict=True) if keep]
    fit_users = [row[1] for row in fit_rows]
    fit_y = np.asarray([row[7] for row in fit_rows], dtype=np.float64)
    fit_base = base_all[fit_mask]
    screen_users = [row[1] for row in screen_rows]
    screen_y = np.asarray([row[7] for row in screen_rows], dtype=np.float64)
    screen_base = base_all[screen_mask]

    fit_control_scores = rerank_mmr(
        fit_rows, fit_users, fit_base, 0.0, pool_size=args.pool_size
    )
    screen_control_scores = rerank_mmr(
        screen_rows, screen_users, screen_base, 0.0, pool_size=args.pool_size
    )
    fit_control = metric_record(fit_users, fit_y, fit_control_scores)
    screen_control = metric_record(screen_users, screen_y, screen_control_scores)
    plausible = (
        0.55 <= fit_control["primary"] <= 0.70
        and 0.55 <= screen_control["primary"] <= 0.70
        and 0.55 <= screen_control["gauc"] <= 0.75
        and 0.50 <= screen_control["ndcg5"] <= 0.70
    )
    if not plausible:
        raise RuntimeError(
            "Base standalone metrics are implausible; refusing to use a potentially "
            f"misaligned score stream: fit={fit_control}, screen={screen_control}"
        )

    fit_grid = []
    for value in LAMBDA_GRID:
        scores = rerank_mmr(
            fit_rows, fit_users, fit_base, value, pool_size=args.pool_size
        )
        metrics = metric_record(fit_users, fit_y, scores)
        fit_grid.append({
            "lambda": value,
            "metrics": metrics,
            "delta_vs_lambda_zero": metric_delta(metrics, fit_control),
        })
    chosen_lambda = select_lambda(fit_grid)

    # The selected lambda is now locked. No screen label enters selection.
    selected_fit_scores = rerank_mmr(
        fit_rows, fit_users, fit_base, chosen_lambda, pool_size=args.pool_size
    )
    selected_screen_scores = rerank_mmr(
        screen_rows, screen_users, screen_base, chosen_lambda,
        pool_size=args.pool_size,
    )
    selected_fit = metric_record(fit_users, fit_y, selected_fit_scores)
    selected_screen = metric_record(screen_users, screen_y, selected_screen_scores)
    fit_gain = metric_delta(selected_fit, fit_control)
    screen_gain = metric_delta(selected_screen, screen_control)
    passed = (
        chosen_lambda > 0.0
        and screen_gain["primary"] >= 1e-5
        and screen_gain["gauc"] >= -1e-12
        and screen_gain["ndcg5"] >= -1e-12
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        screen_dates=np.asarray([row[0] for row in screen_rows], dtype=np.int32),
        control_scores=screen_control_scores.astype(np.float32),
        selected_scores=selected_screen_scores.astype(np.float32),
        selected_lambda=np.float32(chosen_lambda),
        screen_labels=screen_y.astype(np.float32),
    )
    report = {
        "experiment": "CDM/MMR-inspired contextual diversity reranker",
        "status": "passed_train_only_screen" if passed else "rejected_at_train_only_screen",
        "protocol": {
            "source_model_fit": "2022-04-08..2022-04-13",
            "lambda_fit_and_selection": "2022-04-14",
            "locked_screen": "2022-04-15..2022-04-21",
            "confirmation_labels_read": False,
            "hidden_test_accessed": False,
            "outcome_free_similarity": "0.35 author + 0.20 music + 0.10 video type + 0.35 tag Jaccard",
            "reranking_scope": f"greedy MMR within each user's top {args.pool_size}; lower ranks unchanged",
            "selection_rule": "require fit primary gain >= 0.00001; otherwise prefer lambda=0",
            "alignment": (
                "Exact cache length equality against CSV rows dated after the cache's "
                "20220413 model cutoff, plus plausible standalone metrics, was required."
            ),
        },
        "data": {
            "selection_rows": len(fit_rows),
            "selection_users": len(set(fit_users)),
            "screen_rows": len(screen_rows),
            "screen_users": len(set(screen_users)),
            "source_cache": str(SOURCE_CACHE.relative_to(ROOT)),
            "source_stream": "deep (frozen out-of-time DeepFM)",
        },
        "alignment_validation": {
            "exact_row_count_match": True,
            "standalone_metrics_plausible": plausible,
            "selection_control": fit_control,
            "screen_control": screen_control,
        },
        "selection_grid": fit_grid,
        "selected_lambda": chosen_lambda,
        "selected_slice": {
            "control": fit_control,
            "candidate": selected_fit,
            "delta": fit_gain,
            "changed_rows": int(np.count_nonzero(
                selected_fit_scores != fit_control_scores
            )),
        },
        "held_out_screen": {
            "control": screen_control,
            "candidate": selected_screen,
            "delta": screen_gain,
            "changed_rows": int(np.count_nonzero(
                selected_screen_scores != screen_control_scores
            )),
        },
        "screen_gate": {
            "required": "nonzero lambda, primary >= +0.00001, GAUC and nDCG@5 nonnegative",
            "passed": passed,
        },
        "recommendation": (
            "Eligible for a separately authorized later-label confirmation."
            if passed else
            "Do not confirm or add this reranker; retain the frozen base ordering."
        ),
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "scores": str(args.scores_output.resolve().relative_to(ROOT)),
            "report": str(args.output.resolve().relative_to(ROOT)),
        },
        "resource_usage": tracker.finish(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
