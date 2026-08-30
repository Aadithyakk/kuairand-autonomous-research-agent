#!/usr/bin/env python3
"""Compact Adaptive Factorization Network interaction screen.

Architecture/regularization selection uses April 9-11 fit and April 12-14
validation.  The selected AFN and exact embedding+MLP control are refit through
April 14 and screened on April 15-21.  April 22+ is conditional on the strict
train-only gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.champion import load_champion_scores, within_user_rank
from backend.kuailab.resources import ProcessResourceTracker
from scripts import kuairand_runner as runner


DEFAULT_OUTPUT = ROOT / "results" / "calibrated-ranking" / "afn-screen.json"
DEFAULT_SCORES = (
    ROOT / "runtime" / "parallel-calibrated-ranking" / "afn"
    / "screen-scores.npz"
)
CONFIGS = (
    {"name": "embedding_mlp_control", "kind": "control", "regularization": 1e-4},
    {"name": "afn_log8_l2_1e-4", "kind": "afn", "regularization": 1e-4},
    {"name": "afn_log8_l2_1e-3", "kind": "afn", "regularization": 1e-3},
)


def load_authors(data_dir: Path) -> dict[str, str]:
    authors: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            authors[str(row["video_id"])] = str(row["author_id"])
    return authors


def load_train_rows(data_dir: Path, authors: dict[str, str]) -> list[tuple]:
    rows = []
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
    rows = []
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


def encode_rows(
    train_rows: list[tuple], validation_rows: list[tuple],
) -> tuple[dict, int]:
    return runner.data_module.encode({
        "train": train_rows, "valid": validation_rows, "test": [],
    })


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
    labels64 = np.asarray(labels, dtype=np.float64)
    probabilities64 = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7
    )
    return float(-np.mean(
        labels64 * np.log(probabilities64)
        + (1.0 - labels64) * np.log(1.0 - probabilities64)
    ))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 20,
) -> float:
    labels64 = np.asarray(labels, dtype=np.float64)
    probabilities64 = np.asarray(probabilities, dtype=np.float64)
    assignments = np.minimum((probabilities64 * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            error += float(mask.mean()) * abs(float(
                labels64[mask].mean() - probabilities64[mask].mean()
            ))
    return error


def build_model(dimension: int, config: dict, seed: int):
    import torch

    torch.manual_seed(seed)

    class CompactInteractions(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kind = str(config["kind"])
            self.embedding = torch.nn.Embedding(dimension, 8, sparse=True)
            torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            if self.kind == "control":
                self.first = torch.nn.Linear(5 * 8, 26)
                self.second = torch.nn.Linear(26, 1)
                self.exponents = None
            else:
                self.exponents = torch.nn.Parameter(torch.empty(8, 5))
                torch.nn.init.normal_(self.exponents, mean=0.20, std=0.03)
                self.first = torch.nn.Linear(8 * 8, 16)
                self.second = torch.nn.Linear(16, 1)
            self.nonfinite_incidents = 0
            self.log_clamp_events = 0
            self.exponent_clip_events = 0

        def forward(self, fields):
            embedded = self.embedding(fields)
            if self.kind == "control":
                transformed = embedded.flatten(1)
            else:
                absolute = torch.abs(embedded)
                self.log_clamp_events += int(torch.count_nonzero(absolute < 1e-7).detach())
                log_embedding = torch.log(torch.clamp(absolute, min=1e-7, max=1e3))
                log_interactions = torch.einsum(
                    "lf,bfd->bld", self.exponents, log_embedding
                )
                self.exponent_clip_events += int(
                    torch.count_nonzero(torch.abs(log_interactions) > 12.0).detach()
                )
                transformed = torch.exp(torch.clamp(log_interactions, -12.0, 12.0)).flatten(1)
            if not torch.isfinite(transformed).all():
                self.nonfinite_incidents += 1
                transformed = torch.nan_to_num(
                    transformed, nan=0.0, posinf=math.exp(12.0), neginf=0.0
                )
            return self.second(torch.relu(self.first(transformed))).squeeze(1)

        def numerical_diagnostics(self) -> dict:
            return {
                "nonfinite_incidents_recovered": self.nonfinite_incidents,
                "embedding_magnitude_floor_events": self.log_clamp_events,
                "pre_exponential_clip_events": self.exponent_clip_events,
                "embedding_abs_floor": 1e-7,
                "embedding_abs_ceiling": 1e3,
                "pre_exponential_clip": [-12.0, 12.0],
            }

    return CompactInteractions()


def parameter_budget(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    embedding = model.embedding.weight.numel()
    return {
        "total_trainable": int(total),
        "shared_embedding": int(embedding),
        "interaction_and_head": int(total - embedding),
    }


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dimension: int,
    config: dict,
    *,
    seed: int,
    epochs: int = 5,
    batch_size: int = 8192,
):
    import torch
    import torch.nn.functional as functional

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    model = build_model(dimension, config, seed)
    dense_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name != "embedding.weight"
    ]
    sparse_optimizer = torch.optim.SparseAdam([model.embedding.weight], lr=0.003)
    dense_optimizer = torch.optim.Adam(dense_parameters, lr=0.003)
    x = torch.as_tensor(train_x, dtype=torch.long)
    y = torch.as_tensor(train_y, dtype=torch.float32)
    random = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = random.permutation(len(train_y))
        total_loss, rows_seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices])
            bce = functional.binary_cross_entropy_with_logits(logits, y[indices])
            if config["kind"] == "afn":
                regularized = model.exponents
            else:
                regularized = model.first.weight
            regularization = float(config["regularization"]) * torch.mean(
                regularized.square()
            )
            loss = bce + regularization
            if not torch.isfinite(loss):
                model.nonfinite_incidents += 1
                raise FloatingPointError("AFN/control loss became non-finite after safeguards")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_parameters, 5.0)
            sparse_optimizer.step()
            dense_optimizer.step()
            count = len(indices)
            total_loss += float(loss.detach()) * count
            rows_seen += count
        history.append({"epoch": epoch, "loss": total_loss / rows_seen})
    return model, {
        "config": config,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.003,
        "loss": "pointwise BCE",
        "parameter_budget": parameter_budget(model),
        "history": history,
        "numerical_diagnostics": model.numerical_diagnostics(),
    }


def predict(model, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(values), 131_072):
            fields = torch.as_tensor(values[start:start + 131_072], dtype=torch.long)
            logits.append(model(fields).numpy())
    raw = np.concatenate(logits)
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    return raw, probability


def evaluate_model(
    model, values: np.ndarray, labels: np.ndarray, users: list[str],
) -> tuple[dict, np.ndarray, np.ndarray]:
    scores, probabilities = predict(model, values)
    metrics = as_metrics(runner.evaluate_module.evaluate(users, labels, scores))
    metrics["calibration_logloss"] = binary_logloss(labels, probabilities)
    metrics["calibration_ece20"] = expected_calibration_error(labels, probabilities)
    return metrics, scores, probabilities


def actual_user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode("utf-8")).hexdigest()[:8], 16) % 4


def run_confirmation(
    *,
    refit_rows: list[tuple],
    confirmation_rows: list[tuple],
    model,
) -> tuple[dict, np.ndarray]:
    encoded, confirmation_dimension = encode_rows(refit_rows, confirmation_rows)
    valid_x, valid_y, valid_users = encoded["valid"]
    if model.embedding.num_embeddings != confirmation_dimension:
        raise RuntimeError(
            f"Confirmation encoder changed dimension: {confirmation_dimension} != "
            f"{model.embedding.num_embeddings}"
        )
    standalone, candidate_raw, _ = evaluate_model(
        model, valid_x, valid_y, valid_users
    )
    champion_raw, manifest = load_champion_scores(
        project_root=ROOT, expected_rows=len(valid_y)
    )
    champion_rank = within_user_rank(valid_users, champion_raw)
    candidate_rank = within_user_rank(valid_users, candidate_raw)
    fixed_scores = champion_rank + 0.05 * (candidate_rank - champion_rank)
    control = as_metrics(
        runner.evaluate_module.evaluate(valid_users, valid_y, champion_rank)
    )
    fixed = as_metrics(
        runner.evaluate_module.evaluate(valid_users, valid_y, fixed_scores)
    )
    folds = np.asarray([actual_user_fold(user) for user in valid_users], dtype=np.int8)
    users_array = np.asarray(valid_users, dtype=object)
    fold_results = []
    for fold in range(4):
        mask = folds == fold
        baseline_fold = as_metrics(
            runner.evaluate_module.evaluate(
                users_array[mask].tolist(), valid_y[mask], champion_rank[mask]
            )
        )
        candidate_fold = as_metrics(
            runner.evaluate_module.evaluate(
                users_array[mask].tolist(), valid_y[mask], fixed_scores[mask]
            )
        )
        fold_results.append({
            "fold": fold,
            "users": baseline_fold["users"],
            "delta": metric_delta(candidate_fold, baseline_fold),
        })
    return {
        "opened": True,
        "validation": "2022-04-22..2022-04-28",
        "april_29_plus_outcomes_accessed": False,
        "standalone": standalone,
        "champion_manifest_metrics": manifest["validation_metrics"],
        "champion_control": control,
        "fixed_residual_weight": 0.05,
        "fixed_residual": fixed,
        "delta": metric_delta(fixed, control),
        "actual_user_id_folds": fold_results,
        "all_fold_primary_nonnegative": all(
            item["delta"]["primary"] >= -1e-12 for item in fold_results
        ),
    }, fixed_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    args = parser.parse_args()
    tracker = ProcessResourceTracker()
    incidents: list[str] = []

    data_dir = ROOT / "external" / "KuaiRand-Pure" / "data"
    authors = load_authors(data_dir)
    rows = load_train_rows(data_dir, authors)
    selection_fit = [row for row in rows if row[0] <= 20220411]
    selection_valid = [row for row in rows if 20220412 <= row[0] <= 20220414]
    refit_rows = [row for row in rows if row[0] <= 20220414]
    screen_rows = [row for row in rows if 20220415 <= row[0] <= 20220421]
    if min(map(len, (selection_fit, selection_valid, refit_rows, screen_rows))) == 0:
        raise RuntimeError("Expected four non-empty temporal slices")
    if min(row[0] for row in rows) == 20220409:
        incidents.append(
            "The organizer file contains no April 8 rows; April 9-11 is the available "
            "realization of the requested April 8-11 fit."
        )

    selection_encoded, selection_dimension = encode_rows(selection_fit, selection_valid)
    train_x, train_y, _ = selection_encoded["train"]
    valid_x, valid_y, valid_users = selection_encoded["valid"]
    selection_trials = []
    for config in CONFIGS:
        model, training = train_model(
            train_x, train_y, selection_dimension, config, seed=8_117
        )
        metrics, _, _ = evaluate_model(model, valid_x, valid_y, valid_users)
        selection_trials.append({
            "config": config,
            "metrics": metrics,
            "training": training,
        })
    control_selection = selection_trials[0]["metrics"]
    for trial in selection_trials:
        trial["delta_vs_control"] = metric_delta(trial["metrics"], control_selection)
        trial["eligible"] = (
            trial["delta_vs_control"]["gauc"] >= -1e-12
            and trial["delta_vs_control"]["ndcg5"] >= -1e-12
        )
    eligible = [trial for trial in selection_trials if trial["eligible"]]
    selected_trial = max(
        eligible,
        key=lambda trial: (
            trial["metrics"]["primary"],
            trial["config"]["kind"] == "control",
            -trial["config"]["regularization"],
        ),
    )
    selected_config = dict(selected_trial["config"])

    refit_encoded, refit_dimension = encode_rows(refit_rows, screen_rows)
    refit_x, refit_y, _ = refit_encoded["train"]
    screen_x, screen_y, screen_users = refit_encoded["valid"]
    control_model, control_training = train_model(
        refit_x, refit_y, refit_dimension, CONFIGS[0], seed=19_331
    )
    candidate_model, candidate_training = train_model(
        refit_x, refit_y, refit_dimension, selected_config, seed=19_331
    )
    control_metrics, control_scores, control_probabilities = evaluate_model(
        control_model, screen_x, screen_y, screen_users
    )
    candidate_metrics, candidate_scores, candidate_probabilities = evaluate_model(
        candidate_model, screen_x, screen_y, screen_users
    )
    delta = metric_delta(candidate_metrics, control_metrics)
    passed = (
        selected_config["kind"] == "afn"
        and delta["primary"] >= 0.0001
        and delta["gauc"] > 0.0
        and delta["ndcg5"] > 0.0
    )
    if selected_config["kind"] == "control":
        incidents.append(
            "Architecture selection preferred the embedding+MLP control; the locked screen "
            "was still executed to preserve the protocol."
        )
    diagnostics = {
        "control": control_model.numerical_diagnostics(),
        "candidate": candidate_model.numerical_diagnostics(),
    }
    if any(item["nonfinite_incidents_recovered"] for item in diagnostics.values()):
        incidents.append(
            "At least one non-finite AFN activation was recovered with nan_to_num; exact "
            "counts are recorded in numerical_diagnostics."
        )

    confirmation = {
        "opened": False,
        "reason": "Train-only gate failed, so April 22-28 remained sealed.",
    }
    confirmation_scores = np.empty(0, dtype=np.float32)
    if passed:
        confirmation_rows = load_confirmation_rows(data_dir, authors)
        confirmation, confirmation_scores = run_confirmation(
            refit_rows=refit_rows,
            confirmation_rows=confirmation_rows,
            model=candidate_model,
        )

    arrays = (
        control_scores, candidate_scores, control_probabilities, candidate_probabilities,
    )
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError("Non-finite output remained after numerical safeguards")
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
        "experiment": "compact Adaptive Factorization Network",
        "paper": {
            "title": "Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions",
            "authors": "Weiyu Cheng, Yanyan Shen, Linpeng Huang",
            "url": "https://arxiv.org/abs/1909.03276",
            "faithful_core": (
                "Field embeddings are mapped to log magnitude, mixed by learned per-neuron "
                "feature exponents, exponentiated, and passed through a feedforward head."
            ),
        },
        "status": "confirmation_completed" if passed else "rejected_at_train_only_screen",
        "protocol": {
            "architecture_selection_fit": "2022-04-09..2022-04-11",
            "architecture_selection_validation": "2022-04-12..2022-04-14",
            "matched_refit": "2022-04-09..2022-04-14",
            "locked_screen": "2022-04-15..2022-04-21",
            "config_grid_predeclared": list(CONFIGS),
            "pointwise_bce_only": True,
            "same_seed_and_row_order": True,
            "april_22_28_accessed": bool(passed),
            "april_29_plus_outcomes_accessed": False,
            "hidden_test_accessed": False,
        },
        "data": {
            "selection_fit_rows": len(selection_fit),
            "selection_validation_rows": len(selection_valid),
            "refit_rows": len(refit_rows),
            "screen_rows": len(screen_rows),
            "screen_users": len(set(screen_users)),
            "selection_feature_dimension": selection_dimension,
            "refit_feature_dimension": refit_dimension,
        },
        "parameter_budget": {
            "control_head": 1093,
            "afn_interaction_and_head": 1097,
            "difference": 4,
            "shared_embedding_dimension": 8,
        },
        "selection": {
            "trials": selection_trials,
            "selected_config": selected_config,
            "rule": (
                "Best primary among configurations with nonnegative GAUC and nDCG@5 deltas; "
                "exact ties prefer the control."
            ),
        },
        "locked_screen": {
            "matched_control": control_metrics,
            "candidate": candidate_metrics,
            "delta": delta,
            "gate": "AFN selected, primary >= +0.0001, GAUC > 0, nDCG@5 > 0",
            "passed": passed,
            "control_training": control_training,
            "candidate_training": candidate_training,
            "numerical_diagnostics": diagnostics,
        },
        "confirmation": confirmation,
        "incidents": incidents,
        "recommendation": (
            "Use confirmation and actual-user fold evidence to decide promotion."
            if passed else
            "Reject compact AFN and keep April 22-28 sealed for this branch."
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
        "selected_config": selected_config,
        "control": control_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "diagnostics": diagnostics,
        "confirmation": confirmation,
        "incidents": incidents,
        "resource_usage": report["resource_usage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
