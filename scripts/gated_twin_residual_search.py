#!/usr/bin/env python3
"""Screen TWIN-lite history sources inside deterministic routing regimes."""
from __future__ import annotations

import json
import os
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
CHAMPION = Path(os.environ.get(
    "KUAI_TWIN_CHAMPION",
    str(HERE / "gated_metric_tradeoff_pass12_scores.npz"),
))
TWIN_CACHE = Path(os.environ.get(
    "KUAI_RESIDUAL_CACHE",
    str(HERE / "twin_lite_history_features.npz"),
))
REPORT = Path(os.environ.get(
    "KUAI_TWIN_REPORT",
    str(HERE / "gated_twin_residual_search_results.json"),
))
OUTPUT = Path(os.environ.get(
    "KUAI_TWIN_OUTPUT",
    str(HERE / "gated_twin_residual_search_scores.npz"),
))

sys.path.insert(0, str(SCRIPT_DIR))
from joint_structural_routing_search import (  # noqa: E402
    build_static_inputs, normalized_top_margin, user_standardize,
)
from joint_terminal_gate_search import (  # noqa: E402
    exact_metrics, factorize, fast_metrics, ordering_audit, rank_ordinal,
)


DEFAULT_WEIGHT_GRID = [
    -0.02, -0.01, -0.005, -0.0025, -0.001, -0.0005, -0.00025, -0.0001,
    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02,
]


def weight_grid() -> list[float]:
    configured = os.environ.get("KUAI_RESIDUAL_WEIGHT_GRID")
    if not configured:
        return DEFAULT_WEIGHT_GRID
    return [float(value) for value in configured.split(",") if value.strip()]


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
    with np.load(CHAMPION) as archive:
        champion = np.asarray(archive["selected"], dtype=np.float64)
    with np.load(TWIN_CACHE) as archive:
        raw = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    source_filter = os.environ.get("KUAI_RESIDUAL_SOURCE_FILTER")
    if source_filter:
        allowed = {name.strip() for name in source_filter.split(",") if name.strip()}
        raw = {name: values for name, values in raw.items() if name in allowed}
    configured_transforms = {
        value.strip() for value in os.environ.get(
            "KUAI_RESIDUAL_TRANSFORMS", "z,rank"
        ).split(",") if value.strip()
    }
    sources = {}
    for name, values in raw.items():
        if "raw" in configured_transforms:
            sources[f"{name}_raw"] = values
        if "z" in configured_transforms:
            sources[f"{name}_z"] = user_standardize(values, codes, counts)
        if "rank" in configured_transforms:
            sources[f"{name}_rank"] = rank_ordinal(values, codes, counts)

    x = build_static_inputs(rows, users, codes, counts)
    group_size = counts[codes]
    small_gate = (group_size <= 5).astype(np.float64)
    margin = normalized_top_margin(champion, codes, counts)
    positive_margin = margin[margin > 0]
    gates = {
        "all": np.ones(len(rows), dtype=np.float64),
        "date_from_apr23": (rows["date"].to_numpy() >= 20220423).astype(np.float64),
        "date_from_apr24": (rows["date"].to_numpy() >= 20220424).astype(np.float64),
        "date_from_apr25": (rows["date"].to_numpy() >= 20220425).astype(np.float64),
        "date_through_apr23": (rows["date"].to_numpy() <= 20220423).astype(np.float64),
        "date_through_apr24": (rows["date"].to_numpy() <= 20220424).astype(np.float64),
        "two_active_days": x["two_day_gate"],
        "three_four_active_days": x["three_four_gate"],
        "medium_slate": x["medium_slate_gate"],
        "not_medium_slate": 1.0 - x["medium_slate_gate"],
        "low_training_rate": x["low_rate_gate"],
        "not_low_training_rate": 1.0 - x["low_rate_gate"],
        "seen_validation_video_in_training": x["prior_video_gate"],
        "single_session": x["single_gate"],
        "not_single_session": 1.0 - x["single_gate"],
        "two_three_sessions": x["terminal_median_gate"],
        "not_two_three_sessions": 1.0 - x["terminal_median_gate"],
        "high_recent_coverage": x["coverage_gate"],
        "not_high_recent_coverage": 1.0 - x["coverage_gate"],
        "high_score_margin": (
            margin >= np.quantile(positive_margin, 0.75)
        ).astype(np.float64),
        "median_score_margin": (
            margin >= np.quantile(positive_margin, 0.50)
        ).astype(np.float64),
        "small_slate_le5": small_gate,
        "large_slate_ge11": (group_size >= 11).astype(np.float64),
    }
    if os.environ.get("KUAI_TWIN_EXPANDED_GATES") == "1":
        row_index = np.arange(len(champion), dtype=np.int64)
        descending = np.lexsort((row_index, -champion, codes))
        descending_codes = codes[descending]
        boundaries = np.flatnonzero(
            np.r_[True, descending_codes[1:] != descending_codes[:-1]]
        )
        starts = np.repeat(boundaries, counts)
        sorted_positions = np.arange(len(champion), dtype=np.int64) - starts
        champion_position = np.empty(len(champion), dtype=np.int64)
        champion_position[descending] = sorted_positions
        gates.update({
            "slate_size_2": (group_size == 2).astype(np.float64),
            "slate_size_3": (group_size == 3).astype(np.float64),
            "slate_size_4": (group_size == 4).astype(np.float64),
            "slate_size_5": (group_size == 5).astype(np.float64),
            "small_slate_le3": (group_size <= 3).astype(np.float64),
            "small_slate_le4": (group_size <= 4).astype(np.float64),
            "small_and_two_active_days": small_gate * x["two_day_gate"],
            "small_and_three_four_active_days": small_gate * x["three_four_gate"],
            "small_and_low_training_rate": small_gate * x["low_rate_gate"],
            "small_and_not_low_training_rate": small_gate * (1.0 - x["low_rate_gate"]),
            "small_and_seen_video": small_gate * x["prior_video_gate"],
            "small_and_single_session": small_gate * x["single_gate"],
            "small_and_two_three_sessions": small_gate * x["terminal_median_gate"],
            "small_and_high_recent_coverage": small_gate * x["coverage_gate"],
            "small_and_high_score_margin": small_gate * gates["high_score_margin"],
            "small_and_median_score_margin": small_gate * gates["median_score_margin"],
            "current_top5": (champion_position < 5).astype(np.float64),
            "current_top10": (champion_position < 10).astype(np.float64),
            "current_rank_3_to_8": (
                (champion_position >= 2) & (champion_position < 8)
            ).astype(np.float64),
            "current_rank_4_to_7": (
                (champion_position >= 3) & (champion_position < 7)
            ).astype(np.float64),
            "current_rank_5_to_6": (
                (champion_position >= 4) & (champion_position < 6)
            ).astype(np.float64),
        })

    baseline_fast = fast_metrics(labels, champion, codes, counts)
    champion_exact = exact_metrics(users, labels, champion)
    fold_ids = np.asarray([int(user) % 4 for user in users], dtype=np.int8)
    fold_cache = {}
    for fold in range(4):
        mask = fold_ids == fold
        fc, _, fn = factorize(users[mask])
        fold_cache[fold] = (
            mask, fc, fn, fast_metrics(labels[mask], champion[mask], fc, fn)
        )

    best_global = None
    promising = []
    per_source_best = {}
    trials = 0
    weights = weight_grid()
    for source_name, source in sources.items():
        source_best = None
        for gate_name, gate in gates.items():
            gated = gate * source
            if np.all(gated == 0):
                continue
            for weight in weights:
                trials += 1
                candidate = champion + weight * gated
                metrics = fast_metrics(labels, candidate, codes, counts)
                row = {
                    "source": source_name, "gate": gate_name, "weight": weight,
                    "metrics": metrics,
                    "delta": {key: metrics[key] - baseline_fast[key] for key in baseline_fast},
                }
                if source_best is None or metrics["primary"] > source_best["metrics"]["primary"]:
                    source_best = row
                if best_global is None or metrics["primary"] > best_global["metrics"]["primary"]:
                    best_global = row
                if (
                    metrics["primary"] > baseline_fast["primary"]
                    and metrics["gauc"] >= baseline_fast["gauc"]
                    and metrics["ndcg5"] >= baseline_fast["ndcg5"]
                ):
                    stable = True
                    folds = []
                    for fold, (mask, fc, fn, fold_base) in fold_cache.items():
                        fm = fast_metrics(labels[mask], candidate[mask], fc, fn)
                        delta = {key: fm[key] - fold_base[key] for key in fold_base}
                        stable &= all(value >= -1e-12 for value in delta.values())
                        folds.append({"fold": fold, "delta": delta})
                    row["stable"] = bool(stable)
                    row["folds"] = folds
                    promising.append(row)
        per_source_best[source_name] = source_best
        print(json.dumps({"source_done": source_name, "best": source_best}), flush=True)

    stable_rows = [row for row in promising if row["stable"]]
    stable_rows_sorted = sorted(
        stable_rows, key=lambda row: row["metrics"]["primary"], reverse=True
    )
    selected = max(stable_rows, key=lambda row: row["metrics"]["primary"]) if stable_rows else None
    selected_scores = champion
    if selected is not None:
        selected_scores = (
            champion + selected["weight"] * gates[selected["gate"]] * sources[selected["source"]]
        )
        selected["exact_metrics"] = exact_metrics(users, labels, selected_scores)
        selected["exact_delta"] = {
            key: selected["exact_metrics"][key] - champion_exact[key] for key in champion_exact
        }

    audit = ordering_audit(codes, counts, champion, selected_scores)
    np.savez_compressed(
        OUTPUT, champion=champion.astype(np.float32), selected=selected_scores.astype(np.float32)
    )
    report = {
        "experiment": os.environ.get(
            "KUAI_RESIDUAL_EXPERIMENT",
            "gated TWIN-lite target-history residual screen",
        ),
        "decision": "promote" if selected is not None else "retain_champion",
        "trials": trials, "source_count": len(sources), "gate_count": len(gates),
        "champion_exact": champion_exact, "best_global": best_global,
        "per_source_best": per_source_best,
        "globally_promising_count": len(promising),
        "stable_candidate_count": len(stable_rows),
        "top_stable_candidates": stable_rows_sorted[:100],
        "selected": selected, "ordering_audit": audit,
        "hidden_test_accessed": False,
        "artifacts": {"report": str(REPORT), "scores": str(OUTPUT)},
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "per_source_best"}, indent=2))


if __name__ == "__main__":
    main()
