#!/usr/bin/env python3
"""BBP-lite relevance/bias disentanglement with relevance-only inference."""
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
RUNTIME = ROOT / "runtime" / "parallel-calibrated-ranking" / "bbp"
RESULTS = ROOT / "results" / "calibrated-ranking"
PLAN = RESULTS / "bbp-plan.json"
SCREEN_REPORT = RESULTS / "bbp-screen.json"
CONFIRMATION_REPORT = RESULTS / "bbp-confirmation.json"
AUDIT_REPORT = RESULTS / "bbp-audit.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate_module = load_module("bbp_evaluate", STARTER / "evaluate.py")


def authors(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[row["video_id"]] = row["author_id"]
    return result


def parse(row: dict[str, str], author_map: dict[str, str]) -> tuple:
    return (
        int(row["date"]),
        row["user_id"],
        row["video_id"],
        author_map.get(row["video_id"], "UNK"),
        row["tab"],
        float(row["duration_ms"]),
        1 if row["long_view"] != "0" else 0,
        int(row["time_ms"]),
    )


def load_train_only(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    author_map = authors(data_dir)
    core: list[tuple] = []
    selection: list[tuple] = []
    locked: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for raw in csv.DictReader(stream):
            row = parse(raw, author_map)
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
    author_map = authors(data_dir)
    train: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for raw in csv.DictReader(stream):
            train.append(parse(raw, author_map))
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
            valid.append(parse(dict(zip(names, values, strict=True)), author_map))
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError(f"Unexpected confirmation rows: {len(train)}, {len(valid)}")
    return train, valid


def encode_relevance(
    train: list[tuple], valid: list[tuple]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    # Duration and session context are intentionally absent.
    def values(row: tuple) -> tuple[str, str, str, str]:
        return str(row[1]), str(row[2]), str(row[3]), str(row[4])

    vocabs: list[dict[str, int]] = [dict() for _ in range(4)]
    for row in train:
        for field, value in enumerate(values(row)):
            if value not in vocabs[field]:
                vocabs[field][value] = len(vocabs[field])
    unknown = [len(vocab) for vocab in vocabs]
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)

    def apply(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        x = np.empty((len(rows), 4), dtype=np.int32)
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
    return train_x, train_y, valid_x, valid_y, users, int(sum(dimensions))


def behavior_bias_features(rows: list[tuple]) -> tuple[np.ndarray, dict]:
    duration_edges = np.quantile(
        np.asarray([row[5] for row in rows], dtype=np.float64),
        np.linspace(0, 1, 11)[1:-1],
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row[1])].append(index)
    raw = np.zeros((len(rows), 3), dtype=np.float64)
    sessions = 0
    maximum_position = 0
    for row_indices in grouped.values():
        ordered = sorted(row_indices, key=lambda index: (rows[index][7], index))
        previous_time: int | None = None
        previous_date: int | None = None
        session_start = 0
        position = 0
        for index in ordered:
            timestamp = int(rows[index][7])
            date = int(rows[index][0])
            if (
                previous_time is None
                or previous_date != date
                or timestamp < previous_time
                or timestamp - previous_time > 30 * 60 * 1000
            ):
                sessions += 1
                session_start = timestamp
                position = 0
            raw[index] = (
                math.log1p(position),
                float(np.searchsorted(duration_edges, rows[index][5])),
                math.log1p(max(timestamp - session_start, 0) / 1000.0),
            )
            maximum_position = max(maximum_position, position)
            position += 1
            previous_time, previous_date = timestamp, date
    means = raw.mean(axis=0)
    deviations = np.maximum(raw.std(axis=0), 1e-6)
    return ((raw - means) / deviations).astype(np.float32), {
        "fields": [
            "log1p_session_position",
            "duration_decile",
            "log1p_elapsed_session_seconds",
        ],
        "session_gap_minutes": 30,
        "sessions": sessions,
        "maximum_zero_based_position": maximum_position,
        "means": means.tolist(),
        "standard_deviations": deviations.tolist(),
        "outcomes_used": False,
    }


def make_model(
    dimension: int,
    fields: int,
    embedding_dim: int,
    hidden: int,
    representation_dim: int,
):
    from torch import nn

    class BBPLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.relevance_tower = nn.Sequential(
                nn.Linear(fields * embedding_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, representation_dim),
                nn.ReLU(),
            )
            self.relevance_head = nn.Linear(representation_dim, 1)
            self.bias_tower = nn.Sequential(
                nn.Linear(3, representation_dim),
                nn.ReLU(),
                nn.Linear(representation_dim, representation_dim),
                nn.ReLU(),
            )
            self.bias_head = nn.Linear(representation_dim, 1)
            nn.init.zeros_(self.linear.weight)
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.bias_head.weight)
            nn.init.zeros_(self.bias_head.bias)

        def forward(self, x, bias):
            embedding = self.embedding(x)
            summed = embedding.sum(dim=1)
            fm = 0.5 * (
                summed.square() - embedding.square().sum(dim=1)
            ).sum(dim=1)
            relevance_representation = self.relevance_tower(
                embedding.flatten(start_dim=1)
            )
            relevance_logit = self.linear(x).sum(dim=1).squeeze(-1)
            relevance_logit += fm + self.relevance_head(
                relevance_representation
            ).squeeze(-1)
            bias_representation = self.bias_tower(bias)
            bias_logit = self.bias_head(bias_representation).squeeze(-1)
            return (
                relevance_logit,
                bias_logit,
                relevance_representation,
                bias_representation,
            )

    return BBPLite()


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
    train_bias: np.ndarray,
    valid_x: np.ndarray,
    dimension: int,
    output_dir: Path,
    name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden: int,
    representation_dim: int,
    orthogonality_lambda: float,
) -> tuple[np.ndarray, dict]:
    import torch
    from torch import nn

    start_wall, start_cpu = time.monotonic(), time.process_time()
    train_x_t = torch.from_numpy(train_x.astype(np.int64, copy=False))
    train_y_t = torch.from_numpy(train_y.astype(np.float32, copy=False))
    bias_t = torch.from_numpy(train_bias.astype(np.float32, copy=False))
    valid_x_t = torch.from_numpy(valid_x.astype(np.int64, copy=False))
    torch.manual_seed(seed)
    model = make_model(
        dimension, train_x.shape[1], embedding_dim, hidden, representation_dim
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    random = np.random.default_rng(seed + 181)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = random.permutation(len(train_y))
        losses, bces, projections = [], [], []
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            relevance, bias_logit, relevance_rep, bias_rep = model(
                train_x_t[indices], bias_t[indices]
            )
            bce = nn.functional.binary_cross_entropy_with_logits(
                relevance + bias_logit, train_y_t[indices]
            )
            cosine = nn.functional.cosine_similarity(
                relevance_rep, bias_rep, dim=1, eps=1e-8
            )
            projection = cosine.square().mean()  # bounded in [0,1]
            loss = (
                bce + orthogonality_lambda * projection
                if orthogonality_lambda > 0
                else bce
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            bces.append(float(bce.detach()))
            projections.append(float(projection.detach()))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "joint_label_bce": float(np.mean(bces)),
                "bounded_cosine_projection_energy": float(np.mean(projections)),
            }
        )
    # Relevance-only inference: bias values are dummy and bias logit is discarded.
    model.eval()
    chunks = []
    dummy_bias = torch.zeros((min(batch_size * 4, len(valid_x_t)), 3))
    with torch.no_grad():
        for start in range(0, len(valid_x_t), batch_size * 4):
            current = valid_x_t[start:start + batch_size * 4]
            if len(dummy_bias) != len(current):
                dummy_bias = torch.zeros((len(current), 3))
            relevance, _, _, _ = model(current, dummy_bias)
            chunks.append(relevance.numpy())
    scores = np.concatenate(chunks).astype(np.float32)
    checkpoint = output_dir / f"{name}.pt"
    torch.save(model.state_dict(), checkpoint)
    return scores, {
        "objective": "joint_bce_plus_lambda_bounded_cosine_projection",
        "orthogonality_lambda": orthogonality_lambda,
        "inference": "relevance_logit_only",
        "same_initialization_seed": seed,
        "same_batch_order_seed": seed + 181,
        "history": history,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "resource_usage": usage(start_wall, start_cpu),
    }


def train_family(
    *,
    train: list[tuple],
    valid: list[tuple],
    lambdas: list[float],
    output_dir: Path,
    prefix: str,
    plan: dict,
) -> tuple[dict[float, np.ndarray], np.ndarray, list[str], dict]:
    train_x, train_y, valid_x, valid_y, users, dimension = encode_relevance(
        train, valid
    )
    bias, bias_metadata = behavior_bias_features(train)
    scores: dict[float, np.ndarray] = {}
    details: dict[str, dict] = {}
    for penalty in dict.fromkeys(lambdas):
        key = f"{penalty:g}".replace(".", "p")
        values, training = train_model(
            train_x=train_x,
            train_y=train_y,
            train_bias=bias,
            valid_x=valid_x,
            dimension=dimension,
            output_dir=output_dir,
            name=f"{prefix}-lambda-{key}",
            seed=int(plan["fixed_seed"]),
            epochs=int(plan["fixed_epochs"]),
            batch_size=int(plan["fixed_batch_size"]),
            embedding_dim=int(plan["fixed_embedding_dim"]),
            hidden=int(plan["fixed_hidden"]),
            representation_dim=int(plan["fixed_representation_dim"]),
            orthogonality_lambda=penalty,
        )
        scores[penalty] = values
        details[str(penalty)] = training
    return scores, valid_y, users, {
        "models": details,
        "behavior_bias_features": bias_metadata,
        "categorical_dimension": dimension,
    }


def metric(users: list[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    result = evaluate_module.evaluate(users, labels, scores)
    return {
        "GAUC": float(result["GAUC"]),
        "nDCG@5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
        "users": int(result["users"]),
        "rows": int(result["rows"]),
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
        raise RuntimeError("Champion checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        scores = np.asarray(stored["scores"], dtype=np.float64).reshape(-1)
    if len(scores) != expected_rows or not np.isfinite(scores).all():
        raise RuntimeError("Champion alignment/finiteness failure")
    return scores, manifest


def run_screen(args, plan: dict) -> dict:
    start_wall, start_cpu = time.monotonic(), time.process_time()
    core, selection, locked = load_train_only(args.data_dir)
    penalties = [float(value) for value in plan["lambda_grid"]]
    selection_scores, selection_y, selection_users, selection_details = train_family(
        train=core,
        valid=selection,
        lambdas=penalties,
        output_dir=args.output_dir,
        prefix="selection",
        plan=plan,
    )
    scans = [
        {"lambda": penalty, "metrics": metric(selection_users, selection_y, selection_scores[penalty])}
        for penalty in penalties
    ]
    control_selection = next(item for item in scans if item["lambda"] == 0)
    selected = max(
        scans,
        key=lambda item: (
            round(item["metrics"]["primary"] - control_selection["metrics"]["primary"], 12),
            -item["lambda"],
        ),
    )
    selected_lambda = float(selected["lambda"])
    locked_scores, locked_y, locked_users, refit_details = train_family(
        train=core + selection,
        valid=locked,
        lambdas=[0.0, selected_lambda],
        output_dir=args.output_dir,
        prefix="locked-screen",
        plan=plan,
    )
    control_metrics = metric(locked_users, locked_y, locked_scores[0.0])
    candidate_metrics = metric(locked_users, locked_y, locked_scores[selected_lambda])
    gains = {
        key: candidate_metrics[key] - control_metrics[key]
        for key in ("GAUC", "nDCG@5", "primary")
    }
    passed = bool(
        selected_lambda > 0
        and gains["primary"] >= 0.0001
        and gains["GAUC"] > 0
        and gains["nDCG@5"] > 0
    )
    score_path = args.output_dir / "screen-scores.npz"
    np.savez_compressed(
        score_path,
        control_scores=locked_scores[0.0],
        candidate_scores=locked_scores[selected_lambda],
        labels=locked_y,
        users=np.asarray(locked_users, dtype="U"),
        dates=np.asarray([row[0] for row in locked], dtype=np.int32),
        selected_lambda=np.asarray([selected_lambda], dtype=np.float32),
    )
    details_path = args.output_dir / "screen-training-details.json"
    details_path.write_text(
        json.dumps(
            {"selection": selection_details, "refit": refit_details},
            indent=2,
            sort_keys=True,
        )
    )
    report = {
        "experiment": "Behavior Bias-aware Projection lite",
        "status": "screen_passed" if passed else "screen_rejected",
        "merits_locked_confirmation": passed,
        "selected_lambda": selected_lambda,
        "selection": {
            "training": "2022-04-08..2022-04-11",
            "evaluation": "2022-04-12..2022-04-14",
            "scans": scans,
            "zero_preference": True,
        },
        "locked_screen": {
            "refit": "2022-04-08..2022-04-14",
            "evaluation": "2022-04-15..2022-04-21",
            "control_metrics": control_metrics,
            "candidate_metrics": candidate_metrics,
            "candidate_gains": gains,
            "acceptance_primary_gain": 0.0001,
        },
        "protocol": {
            "training_logit": "relevance_plus_behavior_bias",
            "inference_logit": "relevance_only",
            "bounded_penalty": "squared_cosine_similarity",
            "exact_lambda_zero_architecture_control": True,
            "identical_seed_initialization_and_batch_order": True,
            "lambda_grid_predeclared": penalties,
            "lambda_locked_before_screen": True,
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
                np.isfinite(locked_scores[0.0]).all()
                and np.isfinite(locked_scores[selected_lambda]).all()
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
    selected = float(screen["selected_lambda"])
    score_map, labels, users, details = train_family(
        train=train,
        valid=valid,
        lambdas=[0.0, selected],
        output_dir=args.output_dir,
        prefix="confirmation",
        plan=plan,
    )
    control_scores = score_map[0.0]
    candidate_scores = score_map[selected]
    control_metrics = metric(users, labels, control_scores)
    candidate_metrics = metric(users, labels, candidate_scores)
    matched_gains = {
        key: candidate_metrics[key] - control_metrics[key]
        for key in ("GAUC", "nDCG@5", "primary")
    }
    champion_raw, manifest = champion(len(labels))
    champion_rank = rank(users, champion_raw)
    candidate_rank = rank(users, candidate_scores)
    weight = float(plan["fixed_champion_blend_weight"])
    residual = champion_rank + weight * (candidate_rank - champion_rank)
    before = metric(users, labels, champion_rank)
    after = metric(users, labels, residual)
    residual_gains = {
        key: after[key] - before[key] for key in ("GAUC", "nDCG@5", "primary")
    }
    folds = np.asarray([fold_for(user) for user in users], dtype=np.int8)
    users_array = np.asarray(users)
    fold_results = []
    for fold in range(4):
        mask = folds == fold
        fold_before = metric(
            users_array[mask].tolist(), labels[mask], champion_rank[mask]
        )
        fold_after = metric(users_array[mask].tolist(), labels[mask], residual[mask])
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
        control_scores=control_scores,
        candidate_scores=candidate_scores,
        champion_scores=champion_rank.astype(np.float32),
        champion_residual_scores=residual.astype(np.float32),
        labels=labels,
        users=np.asarray(users, dtype="U"),
        dates=np.asarray([row[0] for row in valid], dtype=np.int32),
    )
    details_path = args.output_dir / "confirmation-training-details.json"
    details_path.write_text(json.dumps(details, indent=2, sort_keys=True))
    audit = {
        "experiment": "fixed BBP-lite relevance residual against frozen champion",
        "selected_lambda_locked": selected,
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
        "experiment": "locked BBP-lite confirmation",
        "status": "promotion_passed" if promotion else "confirmation_rejected",
        "merits_promotion": promotion,
        "protocol": {
            "training": "2022-04-08..2022-04-21",
            "confirmation": "2022-04-22..2022-04-28 one locked scored run",
            "selected_lambda_locked_without_retuning": selected,
            "inference_logit": "relevance_only",
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
                np.isfinite(control_scores).all()
                and np.isfinite(candidate_scores).all()
                and np.isfinite(residual).all()
            ),
        },
        "artifacts": {
            "scores": str(score_path.relative_to(ROOT)),
            "training_details": str(details_path.relative_to(ROOT)),
            "audit": str(AUDIT_REPORT.relative_to(ROOT)),
        },
        "resource_usage_total": usage(start_wall, start_cpu),
    }
    CONFIRMATION_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True))
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
