#!/usr/bin/env python3
"""Leak-free MaskNet-style screen and confirmation for KuaiRand-Pure.

Default fast-screen protocol (the second organizer-validation file is never opened):
  * fit/epoch selection: 2022-04-08..11 -> 2022-04-12..14
  * refit:               2022-04-08..14
  * evaluation:          2022-04-15..21

The candidate and control use the same starter-kit fields, embeddings, optimizer,
seed, batch size, and pointwise BCE objective.  They differ only in the
candidate's instance-guided multiplicative mask blocks.

The explicit ``--confirmation`` mode promotes the same fixed configuration to
Apr 22--28.  Later rows are skipped by date before their outcome is accessed.
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
DEFAULT_RUNTIME = ROOT / "runtime" / "parallel-masknet"
DEFAULT_REPORT = ROOT / "results" / "parallel-methods" / "masknet-screen.json"
DEFAULT_CONFIRMATION_REPORT = (
    ROOT / "results" / "parallel-methods" / "masknet-confirmation.json"
)
DEFAULT_AUDIT_REPORT = (
    ROOT / "results" / "parallel-methods" / "masknet-residual-audit.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_module = load_module("masknet_official_data", STARTER / "data.py")
evaluate_module = load_module("masknet_official_evaluate", STARTER / "evaluate.py")


def load_train_only_rows(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Read only the 04/08--04/21 file and make the three temporal windows."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_to_author[row["video_id"]] = row["author_id"]

    core: list[tuple] = []
    selection: list[tuple] = []
    evaluation: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            date = int(row["date"])
            item = (
                date,
                row["user_id"],
                row["video_id"],
                video_to_author.get(row["video_id"], "UNK"),
                row["tab"],
                float(row["duration_ms"]),
                1 if row["long_view"] != "0" else 0,
            )
            if date <= 20220411:
                core.append(item)
            elif date <= 20220414:
                selection.append(item)
            else:
                evaluation.append(item)
    total = len(core) + len(selection) + len(evaluation)
    if total != 1_141_112 or not core or not selection or not evaluation:
        raise RuntimeError(
            "Unexpected train-only windows: "
            f"core={len(core)}, selection={len(selection)}, evaluation={len(evaluation)}"
        )
    return core, selection, evaluation


def load_confirmation_rows(
    data_dir: Path,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Load Apr 8--28 without ever consuming an Apr 29+ CSV record."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_to_author[row["video_id"]] = row["author_id"]

    def item(row: dict[str, str]) -> tuple:
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
            parsed = item(row)
            (core if parsed[0] <= 20220414 else selection).append(parsed)

    # Rows are not stored in date order.  Inspect the date first and never
    # dereference/convert the outcome field for Apr 29+ rows.
    evaluation: list[tuple] = []
    with (data_dir / "log_standard_4_22_to_5_08_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            date = int(row["date"])
            if date > 20220428:
                continue
            parsed = item(row)
            if date < 20220422:
                raise RuntimeError(f"Unexpected confirmation date {date}")
            evaluation.append(parsed)
    if (
        len(core) + len(selection) != 1_141_112
        or len(evaluation) != 124_909
        or not core
        or not selection
    ):
        raise RuntimeError(
            "Unexpected confirmation windows: "
            f"core={len(core)}, selection={len(selection)}, evaluation={len(evaluation)}"
        )
    return core, selection, evaluation


def within_user_ranks(users: list[str], scores: np.ndarray) -> np.ndarray:
    """Stable fractional ranks in [0, 1], aligned to the input rows."""
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    output = np.empty(len(scores), dtype=np.float32)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        order = np.argsort(np.asarray(scores)[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float32)
        ranks[order] = np.arange(len(indices), dtype=np.float32)
        output[indices] = ranks / max(len(indices) - 1, 1)
    return output


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def usage(start_wall: float, start_cpu: float) -> dict:
    wall = time.monotonic() - start_wall
    cpu = time.process_time() - start_cpu
    return {
        "wall_seconds": round(wall, 3),
        "cpu_seconds": round(cpu, 3),
        "cpu_hours": round(cpu / 3600, 6),
        "cpu_utilization_percent": round(100 * cpu / max(wall, 1e-9), 2),
        "peak_rss_mb": round(peak_rss_mb(), 3),
        "gpu_count": 0,
        "gpu_hours": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "device": "cpu",
    }


def encode_selection(
    core_rows: list[tuple], selection_rows: list[tuple]
) -> tuple[dict, int]:
    return data_module.encode(
        {"train": core_rows, "valid": selection_rows, "test": []}
    )


def encode_refit(
    core_rows: list[tuple], selection_rows: list[tuple], evaluation_rows: list[tuple]
) -> tuple[dict, int]:
    return data_module.encode(
        {
            "train": core_rows + selection_rows,
            "valid": evaluation_rows,
            "test": [],
        }
    )


def make_model(
    *, dimension: int, fields: int, embedding_dim: int, hidden: int,
    dropout: float, masked: bool,
):
    import torch
    from torch import nn

    class InteractionModel(nn.Module):
        def __init__(self):
            super().__init__()
            width = fields * embedding_dim
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.masked = masked
            # A two-layer bottleneck creates a sample-specific gate.  Scaling
            # sigmoid by two initializes its semantics around the identity.
            if masked:
                bottleneck = max(width // 4, 8)
                self.mask1 = nn.Sequential(
                    nn.Linear(width, bottleneck),
                    nn.ReLU(),
                    nn.Linear(bottleneck, width),
                )
                self.mask2 = nn.Sequential(
                    nn.Linear(hidden, max(hidden // 4, 8)),
                    nn.ReLU(),
                    nn.Linear(max(hidden // 4, 8), hidden),
                )
            self.deep1 = nn.Linear(width, hidden)
            self.deep2 = nn.Linear(hidden, max(hidden // 2, 8))
            self.output = nn.Linear(max(hidden // 2, 8), 1)
            self.dropout = nn.Dropout(dropout)
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)
            if masked:
                nn.init.zeros_(self.mask1[-1].weight)
                nn.init.zeros_(self.mask1[-1].bias)
                nn.init.zeros_(self.mask2[-1].weight)
                nn.init.zeros_(self.mask2[-1].bias)

        def forward(self, x):
            flat = self.embedding(x).flatten(start_dim=1)
            if self.masked:
                flat = flat * (2.0 * torch.sigmoid(self.mask1(flat)))
            hidden_value = torch.relu(self.deep1(flat))
            if self.masked:
                hidden_value = hidden_value * (
                    2.0 * torch.sigmoid(self.mask2(hidden_value))
                )
            hidden_value = self.dropout(hidden_value)
            hidden_value = self.dropout(torch.relu(self.deep2(hidden_value)))
            return self.linear(x).sum(dim=1).squeeze(-1) + self.output(
                hidden_value
            ).squeeze(-1)

    return InteractionModel()


def train_selected_and_refit(
    *, name: str, masked: bool, selection_encoded: dict, selection_dimension: int,
    refit_encoded: dict, refit_dimension: int, output_dir: Path, seed: int,
    embedding_dim: int, hidden: int, dropout: float, learning_rate: float,
    weight_decay: float, batch_size: int, epochs: int, patience: int,
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

    core_x_t, core_y_t = tensors(core_x, core_y)
    dev_x_t = tensors(dev_x)

    def predict(model, x_tensor):
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), batch_size * 4):
                parts.append(model(x_tensor[start:start + batch_size * 4]).numpy())
        return np.concatenate(parts).astype(np.float32)

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

    torch.manual_seed(seed)
    model = make_model(
        dimension=selection_dimension,
        fields=core_x.shape[1],
        embedding_dim=embedding_dim,
        hidden=hidden,
        dropout=dropout,
        masked=masked,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    random = np.random.default_rng(seed)
    best_primary, best_epoch, bad_epochs = -1.0, 0, 0
    selection_history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model, optimizer, core_x_t, core_y_t, random.permutation(len(core_y))
        )
        dev_logits = predict(model, dev_x_t)
        dev_ranks = within_user_ranks(dev_users, dev_logits)
        metrics = evaluate_module.evaluate(dev_users, dev_y, dev_ranks)
        selection_history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "GAUC": float(metrics["GAUC"]),
                "nDCG@5": float(metrics["nDCG@5"]),
                "primary": float(metrics["primary"]),
            }
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = float(metrics["primary"])
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if not best_epoch:
        raise RuntimeError(f"{name} temporal selection produced no checkpoint")

    # Recreate the model because the refit vocabulary is built strictly from
    # Apr 8--14.  The same seed and selected epoch count make the refit exact.
    full_x_t, full_y_t = tensors(full_x, full_y)
    evaluation_x_t = tensors(evaluation_x)
    torch.manual_seed(seed)
    model = make_model(
        dimension=refit_dimension,
        fields=full_x.shape[1],
        embedding_dim=embedding_dim,
        hidden=hidden,
        dropout=dropout,
        masked=masked,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    refit_random = np.random.default_rng(seed + 8_129)
    refit_losses = []
    for epoch in range(1, best_epoch + 1):
        refit_losses.append(
            train_epoch(
                model,
                optimizer,
                full_x_t,
                full_y_t,
                refit_random.permutation(len(full_y)),
            )
        )

    evaluation_logits = predict(model, evaluation_x_t)
    evaluation_ranks = within_user_ranks(evaluation_users, evaluation_logits)
    evaluation_metrics = evaluate_module.evaluate(
        evaluation_users, evaluation_y, evaluation_ranks
    )
    checkpoint = output_dir / f"{name}-checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    return evaluation_logits, evaluation_ranks, {
        "name": name,
        "architecture": (
            "two_instance_guided_multiplicative_masks_plus_mlp"
            if masked else "plain_wide_mlp"
        ),
        "objective": "pointwise_binary_cross_entropy",
        "seed": seed,
        "selected_epoch": best_epoch,
        "selection_history": selection_history,
        "refit_losses": refit_losses,
        "evaluation_metrics": {
            "GAUC": float(evaluation_metrics["GAUC"]),
            "nDCG@5": float(evaluation_metrics["nDCG@5"]),
            "primary": float(evaluation_metrics["primary"]),
            "users": int(evaluation_metrics["users"]),
            "rows": int(evaluation_metrics["rows"]),
        },
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "resource_usage": usage(started_wall, started_cpu),
    }


def user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode("utf-8")).hexdigest()[:8], 16) % 4


def audit_champion_residual(
    *, users: list[str], labels: np.ndarray, candidate_scores: np.ndarray,
    artifact: Path, output_path: Path,
) -> dict:
    """Cross-fit one fixed MaskNet residual weight against the frozen champion."""
    started_wall, started_cpu = time.monotonic(), time.process_time()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.kuailab.champion import load_champion_scores

    champion_scores, manifest = load_champion_scores(expected_rows=len(labels))
    champion = within_user_ranks(users, champion_scores)
    candidate = within_user_ranks(users, candidate_scores)
    users_array = np.asarray(users, dtype=object)
    folds = np.asarray([user_fold(str(user)) for user in users], dtype=np.int8)
    grid = np.round(np.linspace(-0.25, 0.25, 101), 6)

    def metrics(mask: np.ndarray, weight: float) -> dict:
        scores = champion[mask] + weight * (candidate[mask] - champion[mask])
        evaluated = evaluate_module.evaluate(
            users_array[mask].tolist(), labels[mask], scores,
        )
        return {
            "primary": float(evaluated["primary"]),
            "gauc": float(evaluated["GAUC"]),
            "ndcg5": float(evaluated["nDCG@5"]),
        }

    selected_weights: list[float] = []
    fold_results = []
    for fold in range(4):
        selection_mask = folds != fold
        holdout_mask = folds == fold
        baseline_selection = metrics(selection_mask, 0.0)
        scans = [(float(weight), metrics(selection_mask, float(weight))) for weight in grid]
        weight, selected = max(
            scans,
            key=lambda item: (
                round(item[1]["primary"] - baseline_selection["primary"], 12),
                -abs(item[0]),
            ),
        )
        if selected["primary"] <= baseline_selection["primary"] + 1e-12:
            weight = 0.0
            selected = baseline_selection
        selected_weights.append(weight)
        baseline_holdout = metrics(holdout_mask, 0.0)
        selected_holdout = metrics(holdout_mask, weight)
        fold_results.append(
            {
                "fold": fold,
                "selected_weight": weight,
                "selection_primary_gain": (
                    selected["primary"] - baseline_selection["primary"]
                ),
                "holdout_primary_gain": (
                    selected_holdout["primary"] - baseline_holdout["primary"]
                ),
                "holdout_gauc_gain": selected_holdout["gauc"] - baseline_holdout["gauc"],
                "holdout_ndcg5_gain": (
                    selected_holdout["ndcg5"] - baseline_holdout["ndcg5"]
                ),
            }
        )

    fixed_weight = float(np.mean(selected_weights))
    all_mask = np.ones(len(labels), dtype=bool)
    baseline = metrics(all_mask, 0.0)
    fixed = metrics(all_mask, fixed_weight)
    fixed_gains = {key: fixed[key] - baseline[key] for key in baseline}
    fixed_improves_both = bool(
        fixed_gains["gauc"] > 0.0 and fixed_gains["ndcg5"] > 0.0
    )
    all_folds_nonnegative = all(
        item["holdout_primary_gain"] >= -1e-12 for item in fold_results
    )
    result = {
        "experiment": "MaskNet residual against frozen champion",
        "artifact": str(artifact.relative_to(ROOT)),
        "protocol": (
            "four actual-user-ID folds; choose weight on three folds from a "
            "zero-preferring [-0.25,0.25] grid; report held-out fourth"
        ),
        "grid": {"minimum": -0.25, "maximum": 0.25, "step": 0.005},
        "champion_manifest_metrics": manifest["validation_metrics"],
        "baseline_metrics_recomputed": baseline,
        "candidate_standalone_metrics": metrics(all_mask, 1.0),
        "selected_weights": selected_weights,
        "folds": fold_results,
        "fixed_weight": fixed_weight,
        "fixed_metrics": fixed,
        "fixed_gains": fixed_gains,
        "fixed_improves_both_metrics": fixed_improves_both,
        "all_four_holdout_folds_nonnegative": all_folds_nonnegative,
        "merits_promotion": bool(fixed_improves_both and all_folds_nonnegative),
        "hidden_test_accessed": False,
        "resource_usage": usage(started_wall, started_cpu),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--confirmation", action="store_true",
        help="Select on Apr 15--21 and evaluate the protected Apr 22--28 window.",
    )
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

    import torch

    if args.report is None:
        args.report = DEFAULT_CONFIRMATION_REPORT if args.confirmation else DEFAULT_REPORT
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
    control_logits, control_ranks, control = train_selected_and_refit(
        name="plain-mlp-control", masked=False, **common
    )
    masknet_logits, masknet_ranks, candidate = train_selected_and_refit(
        name="masknet-interactions", masked=True, **common
    )
    rank_artifact = args.output_dir / (
        "confirmation-scores.npz" if args.confirmation else "screen-ranks.npz"
    )
    np.savez_compressed(
        rank_artifact,
        control_logits=control_logits,
        control_ranks=control_ranks,
        masknet_logits=masknet_logits,
        masknet_ranks=masknet_ranks,
    )

    details_artifact = args.output_dir / (
        "confirmation-training-details.json"
        if args.confirmation else "training-details.json"
    )
    details_artifact.write_text(
        json.dumps({"control": control, "candidate": candidate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    def concise_result(result: dict) -> dict:
        selected = next(
            item for item in result["selection_history"]
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
    # A screen must improve the actual objective and avoid merely trading away
    # either component.  Very small numerical changes do not justify touching
    # the protected confirmation labels.
    merits_confirmation = bool(
        deltas["primary"] >= 1e-4
        and deltas["GAUC"] >= -1e-5
        and deltas["nDCG@5"] >= -1e-5
    )
    protocol = (
        {
            "core_training": "2022-04-08..2022-04-14",
            "epoch_selection": "2022-04-15..2022-04-21",
            "refit": "2022-04-08..2022-04-21",
            "evaluation": "2022-04-22..2022-04-28",
            "confirmation_labels_accessed": True,
            "hidden_test_accessed": False,
            "ranking_output": "stable within-user fractional ranks",
        }
        if args.confirmation else
        {
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
        "experiment": "MaskNet-style multiplicative feature interactions",
        "status": (
            "confirmation_evaluated" if args.confirmation
            else ("screen_passed" if merits_confirmation else "screen_rejected")
        ),
        "merits_confirmation": merits_confirmation if not args.confirmation else None,
        "protocol": protocol,
        "matched_configuration": {
            "fields": list(data_module.FIELDS),
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
        "control": concise_result(control),
        "candidate": concise_result(candidate),
        "deltas": deltas,
        "artifacts": {
            "within_user_ranks": str(rank_artifact.relative_to(ROOT)),
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
            candidate_scores=masknet_logits,
            artifact=rank_artifact,
            output_path=DEFAULT_AUDIT_REPORT,
        )
        report["residual_audit"] = {
            "artifact": str(DEFAULT_AUDIT_REPORT.relative_to(ROOT)),
            "fixed_weight": audit["fixed_weight"],
            "fixed_gains": audit["fixed_gains"],
            "fixed_improves_both_metrics": audit["fixed_improves_both_metrics"],
            "all_four_holdout_folds_nonnegative": audit[
                "all_four_holdout_folds_nonnegative"
            ],
            "merits_promotion": audit["merits_promotion"],
            "resource_usage": audit["resource_usage"],
        }
        report["status"] = (
            "confirmation_promoted"
            if audit["merits_promotion"] else "confirmation_rejected"
        )
    # Refresh after the optional audit so the experiment total does not omit
    # the CPU spent on cross-fold metric scans.
    report["resource_usage_total"] = usage(overall_wall, overall_cpu)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
