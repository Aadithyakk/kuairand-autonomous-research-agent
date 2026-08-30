#!/usr/bin/env python3
"""Compact JRC screen with complete-user listwise contexts.

The lambda grid is selected with April 9-11 training and April 12-14
validation.  The exact lambda-zero control and selected arm are then refit on
April 9-14 and screened on April 15-21.  April 22+ is opened only if the
predeclared all-metric gate passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.champion import load_champion_scores, within_user_rank
from backend.kuailab.resources import ProcessResourceTracker
from scripts import kuairand_runner as runner


DEFAULT_OUTPUT = ROOT / "results" / "calibrated-ranking" / "jrc-screen.json"
DEFAULT_SCORES = (
    ROOT / "runtime" / "parallel-calibrated-ranking" / "jrc"
    / "screen-scores.npz"
)
# Fixed before any model fitting.  Zero is an explicit control and ties prefer
# the smaller lambda.
LAMBDA_GRID = (0.0, 0.001, 0.0025, 0.005)


def load_video_authors(data_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            mapping[str(row["video_id"])] = str(row["author_id"])
    return mapping


def load_train_only_rows(data_dir: Path, authors: dict[str, str]) -> list[tuple]:
    rows: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            video = str(row["video_id"])
            rows.append((
                int(row["date"]), str(row["user_id"]), video,
                authors.get(video, "UNK"), str(row["tab"]),
                float(row["duration_ms"]),
                1 if row["long_view"] != "0" else 0,
            ))
    if len(rows) != 1_141_112:
        raise RuntimeError(f"Unexpected April 8-21 row count: {len(rows)}")
    return rows


def load_confirmation_rows(data_dir: Path, authors: dict[str, str]) -> list[tuple]:
    """Load April 22-28 only; stop before accessing any later outcome."""
    rows: list[tuple] = []
    with (data_dir / "log_standard_4_22_to_5_08_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        header = next(stream)
        fieldnames = next(csv.reader([header]))
        if fieldnames[:3] != ["user_id", "video_id", "date"]:
            raise RuntimeError(f"Unexpected later-log header: {fieldnames[:3]}")
        for line in stream:
            prefix = line.split(",", 3)
            if len(prefix) != 4:
                raise RuntimeError("Malformed later-log row")
            date = int(prefix[2])
            if date > 20220428:
                continue  # Do not parse any outcome-bearing April 29+ row.
            if date < 20220422:
                raise RuntimeError(f"Unexpected later-log date: {date}")
            values = next(csv.reader([line]))
            row = dict(zip(fieldnames, values, strict=True))
            video = str(row["video_id"])
            rows.append((
                date, str(row["user_id"]), video, authors.get(video, "UNK"),
                str(row["tab"]), float(row["duration_ms"]),
                1 if row["long_view"] != "0" else 0,
            ))
    if len(rows) != 124_909:
        raise RuntimeError(f"Unexpected April 22-28 row count: {len(rows)}")
    return rows


def as_metrics(result: dict) -> dict[str, float | int]:
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


def binary_logloss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 20,
) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    assignments = np.minimum((p * bins).astype(np.int64), bins - 1)
    total = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            total += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    return total


def group_indices(users: Sequence[object]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def complete_group_batches(
    groups: list[np.ndarray], random: np.random.Generator, row_budget: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pack shuffled complete user groups without splitting any context."""
    group_order = random.permutation(len(groups))
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    current: list[np.ndarray] = []
    current_rows = 0
    for group_number in group_order:
        indices = groups[int(group_number)]
        if current and current_rows + len(indices) > row_budget:
            batch_indices = np.concatenate(current)
            ids = np.concatenate([
                np.full(len(group), number, dtype=np.int64)
                for number, group in enumerate(current)
            ])
            batches.append((batch_indices, ids))
            current, current_rows = [], 0
        current.append(indices)
        current_rows += len(indices)
    if current:
        batch_indices = np.concatenate(current)
        ids = np.concatenate([
            np.full(len(group), number, dtype=np.int64)
            for number, group in enumerate(current)
        ])
        batches.append((batch_indices, ids))
    return batches


def encode_rows(
    train_rows: list[tuple], valid_rows: list[tuple], extra_rows: list[tuple] | None = None,
) -> tuple[dict, int]:
    splits = {
        "train": train_rows,
        "valid": valid_rows,
        "test": [] if extra_rows is None else extra_rows,
    }
    return runner.data_module.encode(splits)


def build_model(dimension: int, seed: int):
    import torch

    torch.manual_seed(seed)

    class CompactJRC(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(dimension, 8, sparse=True)
            self.linear = torch.nn.Embedding(dimension, 2, sparse=True)
            self.hidden = torch.nn.Linear(5 * 8, 32)
            self.output = torch.nn.Linear(32, 2)
            torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(self.linear.weight)

        def forward(self, fields):
            embedded = self.embedding(fields).flatten(1)
            return self.linear(fields).sum(1) + self.output(torch.relu(self.hidden(embedded)))

    return CompactJRC()


def segmented_logsumexp(values, group_ids, group_count: int):
    import torch

    maximum = torch.full(
        (group_count,), -torch.inf, dtype=values.dtype, device=values.device
    )
    maximum.scatter_reduce_(0, group_ids, values, reduce="amax", include_self=True)
    exponential_sum = torch.zeros(
        group_count, dtype=values.dtype, device=values.device
    )
    exponential_sum.scatter_add_(0, group_ids, torch.exp(values - maximum[group_ids]))
    return maximum + torch.log(exponential_sum.clamp_min(1e-12))


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_users: list[str],
    dimension: int,
    *,
    ranking_lambda: float,
    seed: int,
    epochs: int = 4,
    row_budget: int = 8192,
):
    import torch
    import torch.nn.functional as functional

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    model = build_model(dimension, seed)
    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight, model.linear.weight], lr=0.003
    )
    dense_optimizer = torch.optim.Adam(
        [*model.hidden.parameters(), *model.output.parameters()], lr=0.003
    )
    groups = group_indices(train_users)
    random = np.random.default_rng(seed)
    x = torch.as_tensor(train_x, dtype=torch.long)
    y = torch.as_tensor(train_y, dtype=torch.long)
    history = []
    for epoch in range(1, epochs + 1):
        sums = {"loss": 0.0, "point_ce": 0.0, "jrc_listwise": 0.0}
        rows_seen = 0
        batches = complete_group_batches(groups, random, row_budget)
        for indices_array, group_ids_array in batches:
            indices = torch.as_tensor(indices_array, dtype=torch.long)
            group_ids = torch.as_tensor(group_ids_array, dtype=torch.long)
            xb, yb = x[indices], y[indices]
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_ce = functional.cross_entropy(logits, yb)
            group_count = int(group_ids[-1]) + 1
            negative_normalizer = segmented_logsumexp(
                logits[:, 0], group_ids, group_count
            )
            positive_normalizer = segmented_logsumexp(
                logits[:, 1], group_ids, group_count
            )
            selected_logit = torch.where(yb > 0, logits[:, 1], logits[:, 0])
            selected_normalizer = torch.where(
                yb > 0,
                positive_normalizer[group_ids],
                negative_normalizer[group_ids],
            )
            listwise = torch.mean(selected_normalizer - selected_logit)
            loss = point_ce + ranking_lambda * listwise
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.hidden.parameters(), *model.output.parameters()], 5.0
            )
            sparse_optimizer.step()
            dense_optimizer.step()
            count = len(indices_array)
            rows_seen += count
            for key, value in (
                ("loss", loss), ("point_ce", point_ce), ("jrc_listwise", listwise),
            ):
                sums[key] += float(value.detach()) * count
        history.append({
            "epoch": epoch,
            "complete_user_groups": len(groups),
            "batches": len(batches),
            **{key: value / rows_seen for key, value in sums.items()},
        })
    return model, {
        "seed": seed,
        "epochs": epochs,
        "row_budget": row_budget,
        "ranking_lambda": ranking_lambda,
        "optimizer": "SparseAdam+Adam",
        "learning_rate": 0.003,
        "complete_user_groups": len(groups),
        "history": history,
    }


def predict_model(model, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    scores, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(values), 131_072):
            fields = torch.as_tensor(values[start:start + 131_072], dtype=torch.long)
            logits = model(fields)
            difference = logits[:, 1] - logits[:, 0]
            scores.append(difference.numpy())
            probabilities.append(torch.sigmoid(difference).numpy())
    return np.concatenate(scores), np.concatenate(probabilities)


def evaluate_model(
    model, values: np.ndarray, labels: np.ndarray, users: list[str],
) -> tuple[dict, np.ndarray, np.ndarray]:
    scores, probabilities = predict_model(model, values)
    metrics = as_metrics(runner.evaluate_module.evaluate(users, labels, scores))
    metrics["calibration_logloss"] = binary_logloss(labels, probabilities)
    metrics["calibration_ece20"] = expected_calibration_error(labels, probabilities)
    return metrics, scores, probabilities


def actual_user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode("utf-8")).hexdigest()[:8], 16) % 4


def confirmation_audit(
    *,
    train_rows: list[tuple],
    confirmation_rows: list[tuple],
    selected_lambda: float,
    project_root: Path,
) -> tuple[dict, np.ndarray]:
    """One fixed 5% residual and actual-user-ID fold audit, without retuning."""
    encoded, dimension = encode_rows(train_rows, confirmation_rows)
    train_x, train_y, train_users = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model, training = train_model(
        train_x,
        train_y,
        train_users,
        dimension,
        ranking_lambda=selected_lambda,
        seed=16_309,
    )
    candidate_metrics, candidate_raw, candidate_probabilities = evaluate_model(
        model, valid_x, valid_y, valid_users
    )
    champion_raw, manifest = load_champion_scores(
        project_root=project_root, expected_rows=len(valid_y)
    )
    champion_rank = within_user_rank(valid_users, champion_raw)
    candidate_rank = within_user_rank(valid_users, candidate_raw)
    fixed_scores = champion_rank + 0.05 * (candidate_rank - champion_rank)
    baseline = as_metrics(
        runner.evaluate_module.evaluate(valid_users, valid_y, champion_rank)
    )
    fixed = as_metrics(
        runner.evaluate_module.evaluate(valid_users, valid_y, fixed_scores)
    )
    users_array = np.asarray(valid_users, dtype=object)
    folds = np.asarray([actual_user_fold(user) for user in valid_users], dtype=np.int8)
    fold_results = []
    for fold in range(4):
        mask = folds == fold
        fold_control = as_metrics(
            runner.evaluate_module.evaluate(
                users_array[mask].tolist(), valid_y[mask], champion_rank[mask]
            )
        )
        fold_candidate = as_metrics(
            runner.evaluate_module.evaluate(
                users_array[mask].tolist(), valid_y[mask], fixed_scores[mask]
            )
        )
        fold_results.append({
            "fold": fold,
            "users": int(fold_control["users"]),
            "delta": metric_delta(fold_candidate, fold_control),
        })
    return {
        "opened": True,
        "train": "2022-04-09..2022-04-14",
        "validation": "2022-04-22..2022-04-28",
        "april_29_plus_outcomes_accessed": False,
        "selected_lambda_fixed": selected_lambda,
        "residual_weight_fixed": 0.05,
        "champion_manifest_metrics": manifest["validation_metrics"],
        "standalone_candidate": candidate_metrics,
        "standalone_calibration_logloss": binary_logloss(
            valid_y, candidate_probabilities
        ),
        "champion_control": baseline,
        "fixed_residual": fixed,
        "delta": metric_delta(fixed, baseline),
        "actual_user_id_folds": fold_results,
        "all_fold_primary_nonnegative": all(
            item["delta"]["primary"] >= -1e-12 for item in fold_results
        ),
        "training": training,
    }, fixed_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    args = parser.parse_args()
    tracker = ProcessResourceTracker()
    incidents: list[str] = []

    data_dir = ROOT / "external" / "KuaiRand-Pure" / "data"
    authors = load_video_authors(data_dir)
    rows = load_train_only_rows(data_dir, authors)
    selection_train_rows = [row for row in rows if row[0] <= 20220411]
    selection_valid_rows = [
        row for row in rows if 20220412 <= row[0] <= 20220414
    ]
    refit_rows = [row for row in rows if row[0] <= 20220414]
    screen_rows = [row for row in rows if 20220415 <= row[0] <= 20220421]
    if min(map(len, (selection_train_rows, selection_valid_rows, refit_rows, screen_rows))) == 0:
        raise RuntimeError("Expected four non-empty temporal data slices")
    if min(row[0] for row in rows) == 20220409:
        incidents.append(
            "The organizer file contains no April 8 rows; the requested April 8-11 fit is "
            "therefore the available April 9-11 data."
        )

    selection_encoded, selection_dimension = encode_rows(
        selection_train_rows, selection_valid_rows
    )
    selection_train_x, selection_train_y, selection_train_users = (
        selection_encoded["train"]
    )
    selection_valid_x, selection_valid_y, selection_valid_users = (
        selection_encoded["valid"]
    )
    selection_trials = []
    for ranking_lambda in LAMBDA_GRID:
        model, training = train_model(
            selection_train_x,
            selection_train_y,
            selection_train_users,
            selection_dimension,
            ranking_lambda=ranking_lambda,
            seed=7_331,
        )
        metrics, _, _ = evaluate_model(
            model, selection_valid_x, selection_valid_y, selection_valid_users
        )
        selection_trials.append({
            "lambda": ranking_lambda,
            "metrics": metrics,
            "training": training,
        })
    selection_control = selection_trials[0]["metrics"]
    for trial in selection_trials:
        trial["delta_vs_lambda_zero"] = metric_delta(
            trial["metrics"], selection_control
        )
        trial["eligible"] = (
            trial["delta_vs_lambda_zero"]["gauc"] >= -1e-12
            and trial["delta_vs_lambda_zero"]["ndcg5"] >= -1e-12
        )
    eligible = [trial for trial in selection_trials if trial["eligible"]]
    selected_trial = max(
        eligible,
        key=lambda trial: (trial["metrics"]["primary"], -trial["lambda"]),
    )
    selected_lambda = float(selected_trial["lambda"])

    refit_encoded, refit_dimension = encode_rows(refit_rows, screen_rows)
    refit_x, refit_y, refit_users = refit_encoded["train"]
    screen_x, screen_y, screen_users = refit_encoded["valid"]
    control_model, control_training = train_model(
        refit_x,
        refit_y,
        refit_users,
        refit_dimension,
        ranking_lambda=0.0,
        seed=12_971,
    )
    candidate_model, candidate_training = train_model(
        refit_x,
        refit_y,
        refit_users,
        refit_dimension,
        ranking_lambda=selected_lambda,
        seed=12_971,
    )
    control_metrics, control_scores, control_probabilities = evaluate_model(
        control_model, screen_x, screen_y, screen_users
    )
    candidate_metrics, candidate_scores, candidate_probabilities = evaluate_model(
        candidate_model, screen_x, screen_y, screen_users
    )
    delta = metric_delta(candidate_metrics, control_metrics)
    screen_passed = (
        delta["primary"] >= 0.0001
        and delta["gauc"] > 0.0
        and delta["ndcg5"] > 0.0
    )
    if selected_lambda == 0.0:
        incidents.append(
            "Lambda selection preferred the exact pointwise control; the locked train-only "
            "screen was still executed according to the predeclared protocol."
        )
    if selected_lambda == 0.0 and not np.array_equal(control_scores, candidate_scores):
        incidents.append(
            "Two separately trained lambda-zero models were not bitwise equal despite matched "
            "initialization and order; metric equality is reported instead."
        )

    confirmation = {
        "opened": False,
        "reason": "Train-only gate failed, so April 22-28 remained sealed.",
    }
    confirmation_scores = np.empty(0, dtype=np.float32)
    if screen_passed:
        confirmation_rows = load_confirmation_rows(data_dir, authors)
        confirmation, confirmation_scores = confirmation_audit(
            train_rows=refit_rows,
            confirmation_rows=confirmation_rows,
            selected_lambda=selected_lambda,
            project_root=ROOT,
        )

    arrays = (
        control_scores, candidate_scores, control_probabilities, candidate_probabilities,
    )
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError("Non-finite train-only score detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        screen_dates=np.asarray([row[0] for row in screen_rows], dtype=np.int32),
        screen_users=np.asarray(screen_users, dtype="U32"),
        screen_labels=screen_y,
        control_scores=control_scores.astype(np.float32),
        candidate_scores=candidate_scores.astype(np.float32),
        control_probabilities=control_probabilities.astype(np.float32),
        candidate_probabilities=candidate_probabilities.astype(np.float32),
        confirmation_fixed_scores=confirmation_scores.astype(np.float32),
    )
    report = {
        "experiment": "compact JRC two-logit calibrated ranking",
        "paper": {
            "title": "Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model",
            "authors": "Xiang-Rong Sheng et al.",
            "url": "https://arxiv.org/abs/2208.06164",
            "faithful_core": (
                "Two label-state logits; pointwise probability from click minus non-click; "
                "label-specific listwise softmax over every complete user context."
            ),
        },
        "status": (
            "confirmation_completed" if screen_passed else "rejected_at_train_only_screen"
        ),
        "protocol": {
            "lambda_selection_fit": "2022-04-09..2022-04-11",
            "lambda_selection_validation": "2022-04-12..2022-04-14",
            "matched_refit": "2022-04-09..2022-04-14",
            "locked_screen": "2022-04-15..2022-04-21",
            "lambda_grid_predeclared": list(LAMBDA_GRID),
            "complete_user_groups": True,
            "matched_initialization_seed_and_group_order": True,
            "april_22_28_accessed": bool(screen_passed),
            "april_29_plus_outcomes_accessed": False,
            "hidden_test_accessed": False,
        },
        "data": {
            "selection_fit_rows": len(selection_train_rows),
            "selection_validation_rows": len(selection_valid_rows),
            "refit_rows": len(refit_rows),
            "screen_rows": len(screen_rows),
            "screen_users": len(set(screen_users)),
            "selection_feature_dimension": selection_dimension,
            "refit_feature_dimension": refit_dimension,
        },
        "selection": {
            "trials": selection_trials,
            "selected_lambda": selected_lambda,
            "rule": (
                "Best primary among lambdas with nonnegative GAUC and nDCG@5 deltas; "
                "ties prefer the smaller lambda."
            ),
        },
        "locked_screen": {
            "matched_lambda_zero_control": control_metrics,
            "candidate": candidate_metrics,
            "delta": delta,
            "gate": "primary >= +0.0001 and GAUC > 0 and nDCG@5 > 0",
            "passed": screen_passed,
            "control_training": control_training,
            "candidate_training": candidate_training,
        },
        "confirmation": confirmation,
        "incidents": incidents,
        "recommendation": (
            "Use confirmation and fold evidence to decide promotion."
            if screen_passed else
            "Reject compact JRC; do not use confirmation outcomes for this branch."
        ),
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "scores": str(args.scores_output.resolve().relative_to(ROOT)),
            "report": str(args.output.resolve().relative_to(ROOT)),
        },
        "resource_usage": tracker.finish(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_lambda": selected_lambda,
        "control": control_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "confirmation": confirmation,
        "incidents": incidents,
        "resource_usage": report["resource_usage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
