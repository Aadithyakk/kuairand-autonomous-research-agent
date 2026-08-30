#!/usr/bin/env python3
"""Confidence-aware heteroscedastic ranking with sealed confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "external" / "kuairand-starter-kit"
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
RUNTIME = ROOT / "runtime" / "parallel-calibrated-ranking" / "confidence"
RESULTS = ROOT / "results" / "calibrated-ranking"
PLAN = RESULTS / "confidence-plan.json"
SCREEN_REPORT = RESULTS / "confidence-screen.json"
CONFIRMATION_REPORT = RESULTS / "confidence-confirmation.json"
AUDIT_REPORT = RESULTS / "confidence-audit.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate_module = load_module("confidence_evaluate", STARTER / "evaluate.py")


def author_map(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[row["video_id"]] = row["author_id"]
    return result


def parsed(row: dict[str, str], authors: dict[str, str]) -> tuple:
    return (
        int(row["date"]),
        row["user_id"],
        row["video_id"],
        authors.get(row["video_id"], "UNK"),
        row["tab"],
        float(row["duration_ms"]),
        1 if row["long_view"] != "0" else 0,
    )


def load_train_only(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    authors = author_map(data_dir)
    core: list[tuple] = []
    selection: list[tuple] = []
    locked: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for raw in csv.DictReader(stream):
            row = parsed(raw, authors)
            if row[0] <= 20220411:
                core.append(row)
            elif row[0] <= 20220414:
                selection.append(row)
            elif row[0] <= 20220421:
                locked.append(row)
            else:
                raise RuntimeError(f"Protected train-only date: {row[0]}")
    if (len(core), len(selection), len(locked)) != (559_379, 332_039, 249_694):
        raise RuntimeError(
            f"Unexpected train-only rows: {len(core)}, {len(selection)}, {len(locked)}"
        )
    return core, selection, locked


def load_confirmation(data_dir: Path) -> tuple[list[tuple], list[tuple]]:
    authors = author_map(data_dir)
    train: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for raw in csv.DictReader(stream):
            train.append(parsed(raw, authors))
    valid: list[tuple] = []
    with (data_dir / "log_standard_4_22_to_5_08_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        header = next(stream)
        names = next(csv.reader([header]))
        if names[:3] != ["user_id", "video_id", "date"]:
            raise RuntimeError(f"Unexpected later header: {names[:3]}")
        for line in stream:
            prefix = line.split(",", 3)
            if len(prefix) != 4:
                raise RuntimeError("Malformed later row")
            date = int(prefix[2])
            if date > 20220428:
                continue
            if date < 20220422:
                raise RuntimeError(f"Unexpected confirmation date: {date}")
            values = next(csv.reader([line]))
            valid.append(parsed(dict(zip(names, values, strict=True)), authors))
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError(f"Unexpected confirmation rows: {len(train)}, {len(valid)}")
    return train, valid


def encode(
    train: list[tuple], valid: list[tuple]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    durations = np.asarray([row[5] for row in train], dtype=np.float64)
    edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])

    def values(row: tuple) -> tuple[str, str, str, str, str]:
        duration_bucket = str(int(np.searchsorted(edges, row[5])))
        return str(row[1]), str(row[2]), str(row[3]), str(row[4]), duration_bucket

    vocabs: list[dict[str, int]] = [dict() for _ in range(5)]
    for row in train:
        for field, value in enumerate(values(row)):
            if value not in vocabs[field]:
                vocabs[field][value] = len(vocabs[field])
    unknown = [len(vocab) for vocab in vocabs]
    dims = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)

    def apply(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        x = np.empty((len(rows), 5), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users: list[str] = []
        for index, row in enumerate(rows):
            for field, value in enumerate(values(row)):
                x[index, field] = vocabs[field].get(value, unknown[field]) + offsets[field]
            y[index] = float(row[6])
            users.append(str(row[1]))
        return x, y, users

    train_x, train_y, _ = apply(train)
    valid_x, valid_y, users = apply(valid)
    return train_x, train_y, valid_x, valid_y, users, int(sum(dims))


def make_model(
    dimension: int,
    fields: int,
    embedding_dim: int,
    hidden: int,
    heteroscedastic: bool,
):
    from torch import nn

    class ConfidenceDeepFM(nn.Module):
        def __init__(self):
            super().__init__()
            representation = max(hidden // 2, 8)
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.deep = nn.Sequential(
                nn.Linear(fields * embedding_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, representation),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(representation, 1)
            self.heteroscedastic = heteroscedastic
            if heteroscedastic:
                self.log_variance_head = nn.Linear(representation, 1)
                nn.init.zeros_(self.log_variance_head.weight)
                nn.init.zeros_(self.log_variance_head.bias)
            nn.init.zeros_(self.linear.weight)
            nn.init.normal_(self.embedding.weight, std=0.01)

        def forward(self, x):
            embedding = self.embedding(x)
            summed = embedding.sum(dim=1)
            fm = 0.5 * (
                summed.square() - embedding.square().sum(dim=1)
            ).sum(dim=1)
            representation = self.deep(embedding.flatten(start_dim=1))
            mean = self.linear(x).sum(dim=1).squeeze(-1)
            mean = mean + fm + self.mean_head(representation).squeeze(-1)
            if self.heteroscedastic:
                log_variance = self.log_variance_head(representation).squeeze(-1)
                return mean, log_variance.clamp(min=-4.0, max=3.0)
            return mean, None

    return ConfidenceDeepFM()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def usage(start_wall: float, start_cpu: float) -> dict:
    wall = time.monotonic() - start_wall
    cpu = time.process_time() - start_cpu
    return {
        "wall_seconds": round(wall, 3),
        "cpu_seconds": round(cpu, 3),
        "cpu_hours": round(cpu / 3600, 6),
        "cpu_utilization_percent": round(100 * cpu / max(wall, 1e-9), 2),
        "peak_rss_mb": round(peak_rss_mb(), 3),
        "device": "cpu",
        "gpu_count": 0,
        "gpu_hours": 0.0,
        "peak_gpu_memory_mb": 0.0,
    }


def train_model(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    dimension: int,
    output_dir: Path,
    name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden: int,
    heteroscedastic: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from torch import nn

    start_wall, start_cpu = time.monotonic(), time.process_time()
    train_x_t = torch.from_numpy(train_x.astype(np.int64, copy=False))
    train_y_t = torch.from_numpy(train_y.astype(np.float32, copy=False))
    valid_x_t = torch.from_numpy(valid_x.astype(np.int64, copy=False))
    torch.manual_seed(seed)
    model = make_model(
        dimension, train_x.shape[1], embedding_dim, hidden, heteroscedastic
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    random = np.random.default_rng(seed + 149)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = random.permutation(len(train_y))
        losses, bces, variance_means = [], [], []
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            mean, log_variance = model(train_x_t[indices])
            row_bce = nn.functional.binary_cross_entropy_with_logits(
                mean, train_y_t[indices], reduction="none"
            )
            if heteroscedastic:
                loss = (
                    torch.exp(-log_variance) * row_bce + 0.5 * log_variance
                ).mean()
                variance_means.append(float(torch.exp(log_variance).mean().detach()))
            else:
                loss = row_bce.mean()
                variance_means.append(0.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            bces.append(float(row_bce.mean().detach()))
        history.append(
            {
                "epoch": epoch,
                "objective_loss": float(np.mean(losses)),
                "label_bce": float(np.mean(bces)),
                "predicted_variance_mean": float(np.mean(variance_means)),
            }
        )
    model.eval()
    means, sigmas = [], []
    with torch.no_grad():
        for start in range(0, len(valid_x_t), batch_size * 4):
            mean, log_variance = model(valid_x_t[start:start + batch_size * 4])
            means.append(mean.numpy())
            if heteroscedastic:
                sigmas.append(torch.exp(0.5 * log_variance).numpy())
            else:
                sigmas.append(np.zeros(len(mean), dtype=np.float32))
    mean_scores = np.concatenate(means).astype(np.float32)
    sigma_scores = np.concatenate(sigmas).astype(np.float32)
    checkpoint = output_dir / f"{name}.pt"
    torch.save(model.state_dict(), checkpoint)
    return mean_scores, sigma_scores, {
        "model": "heteroscedastic" if heteroscedastic else "mean_only_control",
        "objective": (
            "attenuated_bce_plus_half_log_variance"
            if heteroscedastic
            else "binary_cross_entropy"
        ),
        "same_mean_initialization_seed": seed,
        "same_batch_order_seed": seed + 149,
        "history": history,
        "uncertainty": {
            "minimum_sigma": float(np.min(sigma_scores)),
            "maximum_sigma": float(np.max(sigma_scores)),
            "mean_sigma": float(np.mean(sigma_scores)),
            "std_sigma": float(np.std(sigma_scores)),
        },
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "resource_usage": usage(start_wall, start_cpu),
    }


def user_standardize(users: list[str], values: np.ndarray) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[str(user)].append(index)
    output = np.zeros(len(values), dtype=np.float32)
    for row_list in groups.values():
        indices = np.asarray(row_list, dtype=np.int64)
        local = values[indices].astype(np.float64)
        deviation = float(local.std())
        if deviation > 1e-8:
            output[indices] = ((local - local.mean()) / deviation).astype(np.float32)
    return output


def confidence_score(
    users: list[str], means: np.ndarray, sigmas: np.ndarray, penalty: float
) -> np.ndarray:
    return means.astype(np.float32) - float(penalty) * user_standardize(users, sigmas)


def metrics(users: list[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    result = evaluate_module.evaluate(users, labels, scores)
    return {
        "GAUC": float(result["GAUC"]),
        "nDCG@5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
        "users": int(result["users"]),
        "rows": int(result["rows"]),
    }


def train_pair(
    *,
    train: list[tuple],
    valid: list[tuple],
    output_dir: Path,
    prefix: str,
    plan: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    train_x, train_y, valid_x, valid_y, users, dimension = encode(train, valid)
    common = {
        "train_x": train_x,
        "train_y": train_y,
        "valid_x": valid_x,
        "dimension": dimension,
        "output_dir": output_dir,
        "seed": int(plan["fixed_seed"]),
        "epochs": int(plan["fixed_epochs"]),
        "batch_size": int(plan["fixed_batch_size"]),
        "embedding_dim": int(plan["fixed_embedding_dim"]),
        "hidden": int(plan["fixed_hidden"]),
    }
    control_mean, _, control_details = train_model(
        name=f"{prefix}-mean-control", heteroscedastic=False, **common
    )
    hetero_mean, hetero_sigma, hetero_details = train_model(
        name=f"{prefix}-heteroscedastic", heteroscedastic=True, **common
    )
    return control_mean, hetero_mean, hetero_sigma, valid_y, users, {
        "control": control_details,
        "heteroscedastic": hetero_details,
        "categorical_dimension": dimension,
    }


def rank(users: list[str], scores: np.ndarray) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[str(user)].append(index)
    output = np.empty(len(scores), dtype=np.float64)
    for row_list in groups.values():
        indices = np.asarray(row_list, dtype=np.int64)
        order = np.argsort(scores[indices], kind="stable")
        values = np.empty(len(indices), dtype=np.float64)
        values[order] = np.arange(len(indices), dtype=np.float64)
        output[indices] = (values - values.mean()) / max(float(values.std()), 1e-8)
    return output


def fold_for(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) % 4


def digest(rows: list[tuple]) -> str:
    value = hashlib.sha256()
    for row in rows:
        value.update(f"{row[0]},{row[1]},{row[2]}\n".encode())
    return value.hexdigest()


def champion(expected_rows: int) -> tuple[np.ndarray, dict]:
    manifest = json.loads(
        (ROOT / "results" / "final-model" / "manifest.json").read_text()
    )
    path = ROOT / manifest["validation_scores"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["validation_scores_sha256"]:
        raise RuntimeError("Frozen champion checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        scores = np.asarray(stored["scores"], dtype=np.float64).reshape(-1)
    if len(scores) != expected_rows or not np.isfinite(scores).all():
        raise RuntimeError("Frozen champion alignment/finiteness failure")
    return scores, manifest


def run_screen(args, plan: dict) -> dict:
    start_wall, start_cpu = time.monotonic(), time.process_time()
    core, selection, locked = load_train_only(args.data_dir)
    select_control, select_mean, select_sigma, select_y, select_users, select_details = (
        train_pair(
            train=core,
            valid=selection,
            output_dir=args.output_dir,
            prefix="selection",
            plan=plan,
        )
    )
    control_selection_metrics = metrics(select_users, select_y, select_control)
    scans = []
    penalties = [float(value) for value in plan["penalty_grid"]]
    for penalty in penalties:
        score = confidence_score(select_users, select_mean, select_sigma, penalty)
        scans.append({"penalty": penalty, "metrics": metrics(select_users, select_y, score)})
    selected = max(
        scans,
        key=lambda item: (
            round(
                item["metrics"]["primary"] - control_selection_metrics["primary"], 12
            ),
            -item["penalty"],
        ),
    )
    selected_penalty = float(selected["penalty"])

    refit_control, refit_mean, refit_sigma, locked_y, locked_users, refit_details = (
        train_pair(
            train=core + selection,
            valid=locked,
            output_dir=args.output_dir,
            prefix="locked-screen",
            plan=plan,
        )
    )
    locked_control_metrics = metrics(locked_users, locked_y, refit_control)
    locked_score = confidence_score(
        locked_users, refit_mean, refit_sigma, selected_penalty
    )
    locked_candidate_metrics = metrics(locked_users, locked_y, locked_score)
    gains = {
        key: locked_candidate_metrics[key] - locked_control_metrics[key]
        for key in ("GAUC", "nDCG@5", "primary")
    }
    passed = bool(
        gains["primary"] >= 0.0001 and gains["GAUC"] > 0 and gains["nDCG@5"] > 0
    )
    score_path = args.output_dir / "screen-scores.npz"
    np.savez_compressed(
        score_path,
        control_scores=refit_control,
        heteroscedastic_mean=refit_mean,
        predicted_sigma=refit_sigma,
        confidence_scores=locked_score,
        labels=locked_y,
        users=np.asarray(locked_users, dtype="U"),
        dates=np.asarray([row[0] for row in locked], dtype=np.int32),
        selected_penalty=np.asarray([selected_penalty], dtype=np.float32),
    )
    details_path = args.output_dir / "screen-training-details.json"
    details_path.write_text(
        json.dumps(
            {"selection": select_details, "refit": refit_details},
            indent=2,
            sort_keys=True,
        )
    )
    report = {
        "experiment": "confidence-aware heteroscedastic pointwise ranking",
        "status": "screen_passed" if passed else "screen_rejected",
        "merits_locked_confirmation": passed,
        "selected_penalty": selected_penalty,
        "penalty_selection": {
            "training": "2022-04-08..2022-04-11",
            "selection": "2022-04-12..2022-04-14",
            "control_metrics": control_selection_metrics,
            "candidate_scans": scans,
            "zero_preference": True,
        },
        "locked_screen": {
            "refit": "2022-04-08..2022-04-14",
            "evaluation": "2022-04-15..2022-04-21",
            "control_metrics": locked_control_metrics,
            "candidate_metrics": locked_candidate_metrics,
            "candidate_gains": gains,
            "acceptance_primary_gain": 0.0001,
        },
        "protocol": {
            "identical_mean_initialization_and_batch_order": True,
            "monte_carlo_inference": False,
            "penalty_grid_predeclared": penalties,
            "penalty_locked_before_screen": True,
            "confirmation_accessed": False,
            "hidden_test_outcome_fields_accessed": False,
            "incidents": [],
        },
        "rows": {
            "core": len(core),
            "selection": len(selection),
            "locked_screen": len(locked),
        },
        "alignment": {
            "rows": len(locked),
            "sha256": digest(locked),
            "finite_scores": bool(
                np.isfinite(refit_control).all()
                and np.isfinite(refit_mean).all()
                and np.isfinite(refit_sigma).all()
                and np.isfinite(locked_score).all()
            ),
        },
        "artifacts": {
            "plan": str(PLAN.relative_to(ROOT)),
            "scores": str(score_path.relative_to(ROOT)),
            "training_details": str(details_path.relative_to(ROOT)),
        },
        "resource_usage_total": usage(start_wall, start_cpu),
    }
    SCREEN_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_confirmation(args, plan: dict, screen: dict) -> dict:
    start_wall, start_cpu = time.monotonic(), time.process_time()
    train, valid = load_confirmation(args.data_dir)
    penalty = float(screen["selected_penalty"])
    control, mean, sigma, labels, users, details = train_pair(
        train=train,
        valid=valid,
        output_dir=args.output_dir,
        prefix="confirmation",
        plan=plan,
    )
    candidate = confidence_score(users, mean, sigma, penalty)
    control_metrics = metrics(users, labels, control)
    candidate_metrics = metrics(users, labels, candidate)
    matched_gains = {
        key: candidate_metrics[key] - control_metrics[key]
        for key in ("GAUC", "nDCG@5", "primary")
    }
    champion_raw, manifest = champion(len(labels))
    champion_rank = rank(users, champion_raw)
    candidate_rank = rank(users, candidate)
    weight = float(plan["fixed_champion_blend_weight"])
    residual = champion_rank + weight * (candidate_rank - champion_rank)
    before = metrics(users, labels, champion_rank)
    after = metrics(users, labels, residual)
    residual_gains = {
        key: after[key] - before[key] for key in ("GAUC", "nDCG@5", "primary")
    }
    folds = np.asarray([fold_for(user) for user in users], dtype=np.int8)
    users_array = np.asarray(users)
    fold_results = []
    for fold in range(4):
        mask = folds == fold
        fold_before = metrics(
            users_array[mask].tolist(), labels[mask], champion_rank[mask]
        )
        fold_after = metrics(
            users_array[mask].tolist(), labels[mask], residual[mask]
        )
        fold_gains = {
            key: fold_after[key] - fold_before[key]
            for key in ("GAUC", "nDCG@5", "primary")
        }
        fold_results.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "users": len(set(users_array[mask].tolist())),
                "champion_metrics": fold_before,
                "residual_metrics": fold_after,
                "gains": fold_gains,
                "all_metrics_nonnegative": all(
                    value >= -1e-12 for value in fold_gains.values()
                ),
            }
        )
    overall_pass = all(
        residual_gains[key] > 0 for key in ("GAUC", "nDCG@5", "primary")
    )
    folds_pass = all(item["all_metrics_nonnegative"] for item in fold_results)
    promotion = bool(overall_pass and folds_pass)
    score_path = args.output_dir / "confirmation-scores.npz"
    np.savez_compressed(
        score_path,
        control_scores=control,
        heteroscedastic_mean=mean,
        predicted_sigma=sigma,
        confidence_scores=candidate,
        champion_scores=champion_rank.astype(np.float32),
        champion_residual_scores=residual.astype(np.float32),
        labels=labels,
        users=np.asarray(users, dtype="U"),
        dates=np.asarray([row[0] for row in valid], dtype=np.int32),
    )
    detail_path = args.output_dir / "confirmation-training-details.json"
    detail_path.write_text(json.dumps(details, indent=2, sort_keys=True))
    audit = {
        "experiment": "fixed confidence-aware residual against frozen champion",
        "selected_penalty_locked": penalty,
        "fixed_champion_blend_weight": weight,
        "champion_manifest_metrics": manifest["validation_metrics"],
        "champion_metrics_recomputed": before,
        "residual_metrics": after,
        "residual_gains": residual_gains,
        "overall_all_three_improve": overall_pass,
        "four_actual_user_id_modulo_folds": fold_results,
        "all_four_folds_all_metrics_nonnegative": folds_pass,
        "merits_promotion": promotion,
        "hidden_test_outcome_fields_accessed": False,
    }
    AUDIT_REPORT.write_text(json.dumps(audit, indent=2, sort_keys=True))
    report = {
        "experiment": "locked confidence-aware heteroscedastic confirmation",
        "status": "promotion_passed" if promotion else "confirmation_rejected",
        "merits_promotion": promotion,
        "protocol": {
            "training": "2022-04-08..2022-04-21",
            "confirmation": "2022-04-22..2022-04-28 one locked scored run",
            "selected_penalty_locked_without_retuning": penalty,
            "monte_carlo_inference": False,
            "confirmation_labels_accessed": True,
            "hidden_test_outcome_fields_accessed": False,
            "incidents": [],
        },
        "rows": {"training": len(train), "confirmation": len(valid)},
        "matched_control_metrics": control_metrics,
        "matched_candidate_metrics": candidate_metrics,
        "matched_candidate_gains": matched_gains,
        "champion_audit": audit,
        "alignment": {
            "rows": len(valid),
            "sha256": digest(valid),
            "finite_scores": bool(
                np.isfinite(control).all()
                and np.isfinite(mean).all()
                and np.isfinite(sigma).all()
                and np.isfinite(candidate).all()
                and np.isfinite(residual).all()
            ),
        },
        "artifacts": {
            "scores": str(score_path.relative_to(ROOT)),
            "training_details": str(detail_path.relative_to(ROOT)),
            "audit": str(AUDIT_REPORT.relative_to(ROOT)),
        },
        "resource_usage_total": usage(start_wall, start_cpu),
    }
    CONFIRMATION_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=RUNTIME)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(args.threads, 8)))
    plan = json.loads(PLAN.read_text())
    screen = run_screen(args, plan)
    result: dict = {"screen": screen}
    if screen["merits_locked_confirmation"]:
        result["confirmation"] = run_confirmation(args, plan, screen)
    result["hardware"] = {
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count() or 1,
        "torch_version": torch.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
