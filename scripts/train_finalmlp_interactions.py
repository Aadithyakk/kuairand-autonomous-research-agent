#!/usr/bin/env python3
"""FinalMLP-style two-stream interaction screen and confirmation.

The default screen opens only the first organizer CSV: Apr 8--11 selects on
Apr 12--14, then Apr 8--14 is refit and evaluated on Apr 15--21.  Explicit
``--confirmation`` promotes the fixed configuration to an Apr 8--14 / Apr
15--21 selection and an Apr 8--21 / Apr 22--28 refit/evaluation.  Its raw-line
date-first loader never parses outcome fields for Apr 29+.  The matched control
and candidate share all fields, embeddings, towers, optimizer settings, and
pointwise BCE; only the candidate adds gated low-rank bilinear aggregation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "external" / "kuairand-starter-kit"
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
DEFAULT_RUNTIME = ROOT / "runtime" / "parallel-finalmlp"
DEFAULT_REPORT = ROOT / "results" / "parallel-methods" / "finalmlp-screen.json"
DEFAULT_CONFIRMATION_REPORT = (
    ROOT / "results" / "parallel-methods" / "finalmlp-confirmation.json"
)
DEFAULT_AUDIT_REPORT = (
    ROOT / "results" / "parallel-methods" / "finalmlp-residual-audit.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_module = load_module("finalmlp_official_data", STARTER / "data.py")
evaluate_module = load_module("finalmlp_official_evaluate", STARTER / "evaluate.py")


def load_train_only_rows(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Load Apr 8--21 only and construct all three train-only windows."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_to_author[row["video_id"]] = row["author_id"]

    core: list[tuple] = []
    selection: list[tuple] = []
    evaluation: list[tuple] = []
    # Deliberately do not name or open log_standard_4_22_to_5_08_pure.csv.
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            date = int(row["date"])
            parsed = (
                date,
                row["user_id"],
                row["video_id"],
                video_to_author.get(row["video_id"], "UNK"),
                row["tab"],
                float(row["duration_ms"]),
                1 if row["long_view"] != "0" else 0,
            )
            if date <= 20220411:
                core.append(parsed)
            elif date <= 20220414:
                selection.append(parsed)
            elif date <= 20220421:
                evaluation.append(parsed)
            else:
                raise RuntimeError(f"Protected date unexpectedly present: {date}")
    if len(core) + len(selection) + len(evaluation) != 1_141_112:
        raise RuntimeError(
            "Unexpected train-only row count: "
            f"{len(core)}+{len(selection)}+{len(evaluation)}"
        )
    if not core or not selection or not evaluation:
        raise RuntimeError("One or more temporal screen windows are empty")
    return core, selection, evaluation


def load_confirmation_rows(
    data_dir: Path,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Load Apr 8--28, stopping before the first possible Apr 29 record."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_to_author[row["video_id"]] = row["author_id"]

    def parse(row: dict[str, str]) -> tuple:
        return (
            int(row["date"]),
            row["user_id"],
            row["video_id"],
            video_to_author.get(row["video_id"], "UNK"),
            row["tab"],
            float(row["duration_ms"]),
            1 if row["long_view"] != "0" else 0,
        )

    core: list[tuple] = []
    selection: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            parsed = parse(row)
            (core if parsed[0] <= 20220414 else selection).append(parsed)

    # The physical file is not sorted by date.  Split only through the third
    # comma first; protected-date outcome fields are never parsed.
    evaluation: list[tuple] = []
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
                raise RuntimeError("Malformed interaction row")
            date = int(prefix[2])
            if date > 20220428:
                continue
            if date < 20220422:
                raise RuntimeError(f"Unexpected earlier date in later log: {date}")
            values = next(csv.reader([line]))
            row = dict(zip(fieldnames, values, strict=True))
            evaluation.append(parse(row))
    if len(core) + len(selection) != 1_141_112 or len(evaluation) != 124_909:
        raise RuntimeError(
            "Unexpected confirmation row counts: "
            f"core={len(core)}, selection={len(selection)}, evaluation={len(evaluation)}"
        )
    return core, selection, evaluation


def encode_selection(core: list[tuple], selection: list[tuple]) -> tuple[dict, int]:
    return data_module.encode({"train": core, "valid": selection, "test": []})


def encode_refit(
    core: list[tuple], selection: list[tuple], evaluation: list[tuple]
) -> tuple[dict, int]:
    return data_module.encode(
        {"train": core + selection, "valid": evaluation, "test": []}
    )


def within_user_ranks(users: list[str], scores: np.ndarray) -> np.ndarray:
    """Stable fractional ranks in [0,1], aligned to input rows."""
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    output = np.empty(len(scores), dtype=np.float32)
    for row_indices in groups.values():
        indices = np.asarray(row_indices, dtype=np.int64)
        order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float32)
        ranks[order] = np.arange(len(indices), dtype=np.float32)
        output[indices] = ranks / max(len(indices) - 1, 1)
    return output


def peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


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


def make_model(
    *, dimension: int, embedding_dim: int, hidden: int, dropout: float,
    bilinear: bool,
):
    import torch
    from torch import nn

    class TwoStreamModel(nn.Module):
        def __init__(self):
            super().__init__()
            tower_dim = max(hidden // 2, 8)
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.user_tower = nn.Sequential(
                nn.Linear(embedding_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, tower_dim),
                nn.ReLU(),
            )
            self.item_context_tower = nn.Sequential(
                nn.Linear(4 * embedding_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, tower_dim),
                nn.ReLU(),
            )
            self.additive_head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(2 * tower_dim, 1)
            )
            self.bilinear = bilinear
            if bilinear:
                # A low-rank factorized bilinear term.  Its gate depends on both
                # streams, while tanh bounds the correction before aggregation.
                rank = max(tower_dim // 2, 8)
                self.user_factor = nn.Linear(tower_dim, rank, bias=False)
                self.item_factor = nn.Linear(tower_dim, rank, bias=False)
                self.interaction_gate = nn.Linear(2 * tower_dim, rank)
                self.interaction_head = nn.Linear(rank, 1, bias=False)
                nn.init.normal_(self.interaction_head.weight, std=0.01)
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)

        def forward(self, x):
            embedded = self.embedding(x)
            user = self.user_tower(embedded[:, 0, :])
            item_context = self.item_context_tower(
                embedded[:, 1:, :].flatten(start_dim=1)
            )
            joined = torch.cat([user, item_context], dim=1)
            result = self.linear(x).sum(dim=1).squeeze(-1)
            result = result + self.additive_head(joined).squeeze(-1)
            if self.bilinear:
                factors = self.user_factor(user) * self.item_factor(item_context)
                gated = torch.tanh(factors) * torch.sigmoid(
                    self.interaction_gate(joined)
                )
                result = result + self.interaction_head(gated).squeeze(-1)
            return result

    return TwoStreamModel()


def train_selected_and_refit(
    *, name: str, bilinear: bool, selection_encoded: dict,
    selection_dimension: int, refit_encoded: dict, refit_dimension: int,
    output_dir: Path, seed: int, embedding_dim: int, hidden: int,
    dropout: float, learning_rate: float, weight_decay: float,
    batch_size: int, epochs: int, patience: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from torch import nn

    started_wall, started_cpu = time.monotonic(), time.process_time()
    core_x, core_y, _ = selection_encoded["train"]
    dev_x, dev_y, dev_users = selection_encoded["valid"]
    full_x, full_y, _ = refit_encoded["train"]
    evaluation_x, evaluation_y, evaluation_users = refit_encoded["valid"]

    def tensors(x: np.ndarray, y: np.ndarray | None = None):
        x_tensor = torch.from_numpy(x.astype(np.int64, copy=False))
        if y is None:
            return x_tensor
        return x_tensor, torch.from_numpy(y.astype(np.float32, copy=False))

    def predict(model, x_tensor):
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), batch_size * 4):
                chunks.append(model(x_tensor[start:start + batch_size * 4]).numpy())
        return np.concatenate(chunks).astype(np.float32)

    def train_epoch(model, optimizer, x_tensor, y_tensor, order):
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[indices])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_tensor[indices]
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    core_x_t, core_y_t = tensors(core_x, core_y)
    dev_x_t = tensors(dev_x)
    torch.manual_seed(seed)
    model = make_model(
        dimension=selection_dimension,
        embedding_dim=embedding_dim,
        hidden=hidden,
        dropout=dropout,
        bilinear=bilinear,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    random = np.random.default_rng(seed)
    best_primary, best_epoch, stale = -1.0, 0, 0
    selection_history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, optimizer, core_x_t, core_y_t, random.permutation(len(core_y))
        )
        dev_logits = predict(model, dev_x_t)
        dev_ranks = within_user_ranks(dev_users, dev_logits)
        metrics = evaluate_module.evaluate(dev_users, dev_y, dev_ranks)
        selection_history.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "GAUC": float(metrics["GAUC"]),
                "nDCG@5": float(metrics["nDCG@5"]),
                "primary": float(metrics["primary"]),
            }
        )
        if float(metrics["primary"]) > best_primary + 1e-5:
            best_primary = float(metrics["primary"])
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_epoch == 0:
        raise RuntimeError(f"{name} temporal selection produced no checkpoint")

    # Recreate vocabulary and weights from the refit window, then train for the
    # selected number of epochs without looking at evaluation metrics.
    full_x_t, full_y_t = tensors(full_x, full_y)
    evaluation_x_t = tensors(evaluation_x)
    torch.manual_seed(seed)
    model = make_model(
        dimension=refit_dimension,
        embedding_dim=embedding_dim,
        hidden=hidden,
        dropout=dropout,
        bilinear=bilinear,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    refit_random = np.random.default_rng(seed + 8_129)
    refit_losses = []
    for _ in range(best_epoch):
        refit_losses.append(
            train_epoch(
                model,
                optimizer,
                full_x_t,
                full_y_t,
                refit_random.permutation(len(full_y)),
            )
        )

    raw_scores = predict(model, evaluation_x_t)
    rank_scores = within_user_ranks(evaluation_users, raw_scores)
    metrics = evaluate_module.evaluate(evaluation_users, evaluation_y, rank_scores)
    checkpoint = output_dir / f"{name}-checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    return raw_scores, rank_scores, {
        "name": name,
        "architecture": (
            "two_stream_gated_low_rank_bilinear"
            if bilinear else "two_stream_additive_concatenation"
        ),
        "objective": "pointwise_binary_cross_entropy",
        "seed": seed,
        "selected_epoch": best_epoch,
        "selection_history": selection_history,
        "refit_losses": refit_losses,
        "evaluation_metrics": {
            "GAUC": float(metrics["GAUC"]),
            "nDCG@5": float(metrics["nDCG@5"]),
            "primary": float(metrics["primary"]),
            "users": int(metrics["users"]),
            "rows": int(metrics["rows"]),
        },
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "resource_usage": usage(started_wall, started_cpu),
    }


def actual_user_fold(user: str) -> int:
    """Deterministic fold from the actual numeric user ID (fallback is stable)."""
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode("utf-8")).hexdigest()[:8], 16) % 4


def audit_champion_residual(
    *, users: list[str], labels: np.ndarray, candidate_ranks: np.ndarray,
    score_artifact: Path, output_path: Path,
) -> dict:
    """Cross-fit a fixed FinalMLP residual weight against frozen champion scores."""
    started_wall, started_cpu = time.monotonic(), time.process_time()
    champion_artifact = ROOT / "results" / "final-model" / "validation-scores.npz"
    with np.load(champion_artifact) as stored:
        champion_raw = np.asarray(stored["scores"], dtype=np.float32)
    if len(champion_raw) != len(labels):
        raise RuntimeError(
            f"Champion has {len(champion_raw)} rows, expected {len(labels)}"
        )
    champion_ranks = within_user_ranks(users, champion_raw)
    candidate_ranks = np.asarray(candidate_ranks, dtype=np.float32)
    users_array = np.asarray(users, dtype=object)
    folds = np.asarray([actual_user_fold(str(user)) for user in users], dtype=np.int8)
    grid = np.round(np.linspace(-0.25, 0.25, 101), 6)

    def measured(mask: np.ndarray, weight: float) -> dict:
        scores = champion_ranks[mask] + weight * (
            candidate_ranks[mask] - champion_ranks[mask]
        )
        result = evaluate_module.evaluate(
            users_array[mask].tolist(), labels[mask], scores
        )
        return {
            "primary": float(result["primary"]),
            "GAUC": float(result["GAUC"]),
            "nDCG@5": float(result["nDCG@5"]),
        }

    selected_weights: list[float] = []
    fold_results = []
    for fold in range(4):
        selection_mask = folds != fold
        holdout_mask = folds == fold
        baseline_selection = measured(selection_mask, 0.0)
        scans = [
            (float(weight), measured(selection_mask, float(weight)))
            for weight in grid
        ]
        selected_weight, selected_metrics = max(
            scans,
            key=lambda item: (
                round(item[1]["primary"] - baseline_selection["primary"], 12),
                -abs(item[0]),
            ),
        )
        # Zero-preferring: a fitted correction must strictly beat no correction.
        if selected_metrics["primary"] <= baseline_selection["primary"] + 1e-12:
            selected_weight = 0.0
            selected_metrics = baseline_selection
        selected_weights.append(selected_weight)
        baseline_holdout = measured(holdout_mask, 0.0)
        selected_holdout = measured(holdout_mask, selected_weight)
        gains = {
            metric: selected_holdout[metric] - baseline_holdout[metric]
            for metric in ("primary", "GAUC", "nDCG@5")
        }
        fold_results.append(
            {
                "fold": fold,
                "rows": int(np.sum(holdout_mask)),
                "users": int(len(set(users_array[holdout_mask].tolist()))),
                "selected_weight": selected_weight,
                "selection_primary_gain": (
                    selected_metrics["primary"] - baseline_selection["primary"]
                ),
                "baseline_metrics": baseline_holdout,
                "residual_metrics": selected_holdout,
                "gains": gains,
                "all_metrics_nonnegative": bool(
                    all(value >= -1e-12 for value in gains.values())
                ),
            }
        )

    fixed_weight = float(np.mean(selected_weights))
    all_rows = np.ones(len(labels), dtype=bool)
    baseline = measured(all_rows, 0.0)
    candidate = measured(all_rows, 1.0)
    fixed = measured(all_rows, fixed_weight)
    fixed_gains = {
        metric: fixed[metric] - baseline[metric]
        for metric in ("primary", "GAUC", "nDCG@5")
    }
    all_folds_all_metrics_nonnegative = bool(
        all(item["all_metrics_nonnegative"] for item in fold_results)
    )
    all_fixed_metrics_nonnegative = bool(
        all(value >= -1e-12 for value in fixed_gains.values())
    )
    audit = {
        "experiment": "FinalMLP rank residual against frozen champion",
        "protocol": (
            "four actual-user-ID modulo folds; choose weight on three folds "
            "from a zero-preferring [-0.25,0.25] grid and report the held-out fourth"
        ),
        "grid": {"minimum": -0.25, "maximum": 0.25, "step": 0.005},
        "champion_artifact": str(champion_artifact.relative_to(ROOT)),
        "candidate_artifact": str(score_artifact.relative_to(ROOT)),
        "baseline_metrics": baseline,
        "candidate_standalone_metrics": candidate,
        "selected_weights": selected_weights,
        "folds": fold_results,
        "fixed_weight": fixed_weight,
        "fixed_residual_metrics": fixed,
        "fixed_residual_gains": fixed_gains,
        "all_fixed_metrics_nonnegative": all_fixed_metrics_nonnegative,
        "all_four_folds_all_metrics_nonnegative": all_folds_all_metrics_nonnegative,
        "merits_promotion": bool(
            fixed_gains["primary"] > 0.0
            and all_fixed_metrics_nonnegative
            and all_folds_all_metrics_nonnegative
        ),
        "confirmation_labels_accessed": True,
        "hidden_test_accessed": False,
        "resource_usage": usage(started_wall, started_cpu),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="Promote the fixed screen configuration to Apr 22--28 confirmation",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=260830)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    if args.report is None:
        args.report = DEFAULT_CONFIRMATION_REPORT if args.confirmation else DEFAULT_REPORT

    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(args.threads, 12)))
    overall_wall, overall_cpu = time.monotonic(), time.process_time()

    if args.confirmation:
        core, selection, evaluation = load_confirmation_rows(args.data_dir.resolve())
    else:
        core, selection, evaluation = load_train_only_rows(args.data_dir.resolve())
    selection_encoded, selection_dimension = encode_selection(core, selection)
    refit_encoded, refit_dimension = encode_refit(core, selection, evaluation)
    common = {
        "selection_encoded": selection_encoded,
        "selection_dimension": selection_dimension,
        "refit_encoded": refit_encoded,
        "refit_dimension": refit_dimension,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "embedding_dim": args.embedding_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    name_prefix = "confirmation-" if args.confirmation else ""
    control_raw, control_ranks, control = train_selected_and_refit(
        name=f"{name_prefix}two-stream-additive-control", bilinear=False, **common
    )
    candidate_raw, candidate_ranks, candidate = train_selected_and_refit(
        name=f"{name_prefix}finalmlp-bilinear", bilinear=True, **common
    )

    score_artifact = args.output_dir / (
        "confirmation-scores.npz" if args.confirmation else "screen-scores.npz"
    )
    np.savez_compressed(
        score_artifact,
        control_raw=control_raw,
        control_ranks=control_ranks,
        candidate_raw=candidate_raw,
        candidate_ranks=candidate_ranks,
    )
    details_artifact = args.output_dir / (
        "confirmation-training-details.json"
        if args.confirmation else "training-details.json"
    )
    details_artifact.write_text(
        json.dumps({"control": control, "candidate": candidate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    def concise(result: dict) -> dict:
        selected = next(
            item
            for item in result["selection_history"]
            if int(item["epoch"]) == int(result["selected_epoch"])
        )
        return {
            "name": result["name"],
            "architecture": result["architecture"],
            "objective": result["objective"],
            "seed": result["seed"],
            "selected_epoch": result["selected_epoch"],
            "selection_metrics": {
                key: selected[key] for key in ("GAUC", "nDCG@5", "primary")
            },
            "evaluation_metrics": result["evaluation_metrics"],
            "checkpoint": result["checkpoint"],
            "resource_usage": result["resource_usage"],
        }

    control_metrics = control["evaluation_metrics"]
    candidate_metrics = candidate["evaluation_metrics"]
    deltas = {
        metric: float(candidate_metrics[metric] - control_metrics[metric])
        for metric in ("GAUC", "nDCG@5", "primary")
    }
    matched_candidate_improves_all_metrics = bool(
        deltas["primary"] > 0.0
        and deltas["GAUC"] > 0.0
        and deltas["nDCG@5"] > 0.0
    )
    protocol = (
        {
            "core_training": "2022-04-08..2022-04-14",
            "epoch_selection": "2022-04-15..2022-04-21",
            "refit": "2022-04-08..2022-04-21",
            "evaluation": "2022-04-22..2022-04-28",
            "confirmation_labels_accessed": True,
            "hidden_test_outcomes_accessed_in_clean_model_run": False,
            "hidden_test_outcomes_used": False,
            "integrity_incident": {
                "aborted_probe_hidden_row_parsed": True,
                "rows": 1,
                "description": (
                    "One aborted preflight parsed one Apr29 row including its "
                    "label into transient memory. The value was not printed, "
                    "scored, stored, trained on, or used. The clean confirmation "
                    "run parses outcome fields only for Apr22-28."
                ),
            },
            "ranking_output": "stable within-user fractional ranks",
        }
        if args.confirmation
        else {
            "core_training": "2022-04-08..2022-04-11",
            "epoch_selection": "2022-04-12..2022-04-14",
            "refit": "2022-04-08..2022-04-14",
            "evaluation": "2022-04-15..2022-04-21",
            "confirmation_labels_accessed": False,
            "hidden_test_accessed": False,
            "ranking_output": "stable within-user fractional ranks",
        }
    )
    report = {
        "experiment": "FinalMLP-style two-stream bilinear interactions",
        "status": (
            "confirmation_evaluated"
            if args.confirmation
            else (
                "screen_passed"
                if matched_candidate_improves_all_metrics else "screen_rejected"
            )
        ),
        "merits_confirmation": (
            None if args.confirmation else matched_candidate_improves_all_metrics
        ),
        "matched_candidate_improves_all_metrics": (
            matched_candidate_improves_all_metrics
        ),
        "protocol": protocol,
        "matched_configuration": {
            "fields": list(data_module.FIELDS),
            "streams": {
                "user_history": ["user_id"],
                "item_context": ["video_id", "author_id", "tab", "dur_bucket"],
            },
            "seed": args.seed,
            "embedding_dim": args.embedding_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "threads": torch.get_num_threads(),
        },
        "rows": {
            "core_training": len(core),
            "epoch_selection": len(selection),
            "evaluation": len(evaluation),
        },
        "control": concise(control),
        "candidate": concise(candidate),
        "deltas": deltas,
        "artifacts": {
            "raw_and_rank_scores": str(score_artifact.relative_to(ROOT)),
            "training_details": str(details_artifact.relative_to(ROOT)),
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
        },
        "resource_usage_total": usage(overall_wall, overall_cpu),
        "hardware": {
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count() or 1,
            "torch_version": torch.__version__,
        },
    }
    if args.confirmation:
        _, evaluation_y, evaluation_users = refit_encoded["valid"]
        audit = audit_champion_residual(
            users=evaluation_users,
            labels=evaluation_y,
            candidate_ranks=candidate_ranks,
            score_artifact=score_artifact,
            output_path=DEFAULT_AUDIT_REPORT,
        )
        report["residual_audit"] = {
            "artifact": str(DEFAULT_AUDIT_REPORT.relative_to(ROOT)),
            "fixed_weight": audit["fixed_weight"],
            "fixed_residual_metrics": audit["fixed_residual_metrics"],
            "fixed_residual_gains": audit["fixed_residual_gains"],
            "all_fixed_metrics_nonnegative": audit[
                "all_fixed_metrics_nonnegative"
            ],
            "all_four_folds_all_metrics_nonnegative": audit[
                "all_four_folds_all_metrics_nonnegative"
            ],
            "merits_promotion": audit["merits_promotion"],
        }
        report["resource_usage_total"] = usage(overall_wall, overall_cpu)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
