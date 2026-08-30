#!/usr/bin/env python3
"""Regression-Compatible Ranking (RCR) loss ablation for KuaiRand.

Implements Bai et al., CIKM 2023, Definition 4 for binary relevance:
  (1-alpha) * SigmoidCE + alpha * ListCE(sigmoid)

The default screen selects the epoch count using Apr 8--11 -> Apr 12--14,
then trains the predeclared alpha grid on Apr 8--14 and evaluates Apr 15--21.
No Apr 22+ file is opened by the screen.  Confirmation is an explicit locked
mode and safely parses outcomes only for Apr 22--28.
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
RUNTIME = ROOT / "runtime" / "parallel-calibrated-ranking" / "rcr"
SCREEN_REPORT = ROOT / "results" / "calibrated-ranking" / "rcr-screen.json"
CONFIRM_REPORT = ROOT / "results" / "calibrated-ranking" / "rcr-confirmation.json"
AUDIT_REPORT = ROOT / "results" / "calibrated-ranking" / "rcr-residual-audit.json"
PAPER = "https://arxiv.org/abs/2211.01494"
ALPHA_GRID = (0.0, 0.0005, 0.001, 0.002)
FIXED_RESIDUAL_WEIGHT = 0.05


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_module = load_module("rcr_official_data", STARTER / "data.py")
evaluate_module = load_module("rcr_official_evaluate", STARTER / "evaluate.py")


def parsed_row(row: dict[str, str], authors: dict[str, str]) -> tuple:
    return (
        int(row["date"]),
        row["user_id"],
        row["video_id"],
        authors.get(row["video_id"], "UNK"),
        row["tab"],
        float(row["duration_ms"]),
        1 if row["long_view"] != "0" else 0,
    )


def load_authors(data_dir: Path) -> dict[str, str]:
    authors: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            authors[row["video_id"]] = row["author_id"]
    return authors


def load_screen_rows(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Only opens the Apr 8--21 log."""
    authors = load_authors(data_dir)
    core: list[tuple] = []
    dev: list[tuple] = []
    evaluation: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            item = parsed_row(row, authors)
            if item[0] <= 20220411:
                core.append(item)
            elif item[0] <= 20220414:
                dev.append(item)
            elif item[0] <= 20220421:
                evaluation.append(item)
            else:
                raise RuntimeError(f"Unexpected date in first log: {item[0]}")
    if len(core) + len(dev) + len(evaluation) != 1_141_112:
        raise RuntimeError(
            f"Unexpected screen rows: {len(core)}, {len(dev)}, {len(evaluation)}"
        )
    return core, dev, evaluation


def load_confirmation_rows(data_dir: Path) -> tuple[list[tuple], list[tuple]]:
    """Load Apr 8--21 and Apr 22--28 without parsing Apr 29+ outcomes."""
    authors = load_authors(data_dir)
    training: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            training.append(parsed_row(row, authors))

    evaluation: list[tuple] = []
    later = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    with later.open(encoding="utf-8") as stream:
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
                raise RuntimeError(f"Unexpected earlier later-log date: {date}")
            values = next(csv.reader([line]))
            evaluation.append(parsed_row(dict(zip(fieldnames, values, strict=True)), authors))
    if len(training) != 1_141_112 or len(evaluation) != 124_909:
        raise RuntimeError(
            f"Unexpected confirmation rows: {len(training)}, {len(evaluation)}"
        )
    return training, evaluation


def encode(train: list[tuple], valid: list[tuple]) -> tuple[dict, int]:
    return data_module.encode({"train": train, "valid": valid, "test": []})


def within_user_ranks(users: list[str], scores: np.ndarray) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    output = np.empty(len(scores), dtype=np.float32)
    for row_list in groups.values():
        rows = np.asarray(row_list, dtype=np.int64)
        order = np.argsort(scores[rows], kind="stable")
        ranks = np.empty(len(rows), dtype=np.float32)
        ranks[order] = np.arange(len(rows), dtype=np.float32)
        output[rows] = ranks / max(len(rows) - 1, 1)
    return output


def metrics(users: list[str], labels: np.ndarray, raw: np.ndarray) -> tuple[dict, np.ndarray]:
    ranks = within_user_ranks(users, raw)
    result = evaluate_module.evaluate(users, labels, ranks)
    return {
        "primary": float(result["primary"]),
        "GAUC": float(result["GAUC"]),
        "nDCG@5": float(result["nDCG@5"]),
        "rows": int(result["rows"]),
        "users": int(result["users"]),
    }, ranks


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
        "device": "cpu",
        "gpu_count": 0,
        "gpu_hours": 0.0,
        "peak_gpu_memory_mb": 0.0,
    }


def user_groups(users: list[str]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        grouped.setdefault(str(user), []).append(index)
    return [np.asarray(rows, dtype=np.int64) for rows in grouped.values()]


def group_batches(groups: list[np.ndarray], batch_size: int, random) -> list[tuple]:
    """Pack complete user groups into batches; never split a query."""
    batches: list[tuple] = []
    packed: list[np.ndarray] = []
    rows = 0
    for group_index in random.permutation(len(groups)):
        group = groups[int(group_index)]
        if packed and rows + len(group) > batch_size:
            indices = np.concatenate(packed)
            local_groups = np.concatenate(
                [np.full(len(value), i, dtype=np.int64) for i, value in enumerate(packed)]
            )
            batches.append((indices, local_groups, len(packed)))
            packed, rows = [], 0
        packed.append(group)
        rows += len(group)
    if packed:
        indices = np.concatenate(packed)
        local_groups = np.concatenate(
            [np.full(len(value), i, dtype=np.int64) for i, value in enumerate(packed)]
        )
        batches.append((indices, local_groups, len(packed)))
    return batches


def make_model(*, dimension: int, fields: int, embedding_dim: int, hidden: int, dropout: float):
    import torch
    from torch import nn

    class MaskNetBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            width = fields * embedding_dim
            bottleneck = max(width // 4, 8)
            second = max(hidden // 2, 8)
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.mask1 = nn.Sequential(
                nn.Linear(width, bottleneck), nn.ReLU(), nn.Linear(bottleneck, width)
            )
            self.deep1 = nn.Linear(width, hidden)
            self.mask2 = nn.Sequential(
                nn.Linear(hidden, max(hidden // 4, 8)),
                nn.ReLU(),
                nn.Linear(max(hidden // 4, 8), hidden),
            )
            self.deep2 = nn.Linear(hidden, second)
            self.output = nn.Linear(second, 1)
            self.dropout = nn.Dropout(dropout)
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)
            for mask in (self.mask1, self.mask2):
                nn.init.zeros_(mask[-1].weight)
                nn.init.zeros_(mask[-1].bias)

        def forward(self, x):
            flat = self.embedding(x).flatten(start_dim=1)
            flat = flat * (2.0 * torch.sigmoid(self.mask1(flat)))
            hidden_value = torch.relu(self.deep1(flat))
            hidden_value = hidden_value * (2.0 * torch.sigmoid(self.mask2(hidden_value)))
            hidden_value = self.dropout(hidden_value)
            hidden_value = self.dropout(torch.relu(self.deep2(hidden_value)))
            return self.linear(x).sum(dim=1).squeeze(-1) + self.output(hidden_value).squeeze(-1)

    return MaskNetBackbone()


def sigmoid_listce(logits, labels, group_ids, group_count: int):
    """Mean sigmoid-normalized ListCE over complete positive-containing users."""
    import torch

    probabilities = torch.sigmoid(logits).clamp_min(1e-7)
    probability_sum = logits.new_zeros(group_count)
    positive_sum = logits.new_zeros(group_count)
    probability_sum.scatter_add_(0, group_ids, probabilities)
    positive_sum.scatter_add_(0, group_ids, labels)
    valid = positive_sum > 0
    if not bool(valid.any()):
        return logits.new_zeros(())
    log_distribution = probabilities.log() - probability_sum[group_ids].clamp_min(1e-7).log()
    target_distribution = labels / positive_sum[group_ids].clamp_min(1.0)
    item_loss = -target_distribution * log_distribution
    query_loss = logits.new_zeros(group_count)
    query_loss.scatter_add_(0, group_ids, item_loss)
    return query_loss[valid].mean()


def train_fixed(
    *, alpha: float, train_x: np.ndarray, train_y: np.ndarray, train_users: list[str],
    dimension: int, epochs: int, seed: int, embedding_dim: int, hidden: int,
    dropout: float, learning_rate: float, weight_decay: float, batch_size: int,
    valid_x: np.ndarray | None = None, valid_y: np.ndarray | None = None,
    valid_users: list[str] | None = None, patience: int | None = None,
) -> tuple[object, list[dict], int, dict]:
    import torch
    from torch import nn

    started_wall, started_cpu = time.monotonic(), time.process_time()
    train_x_t = torch.from_numpy(train_x.astype(np.int64, copy=False))
    train_y_t = torch.from_numpy(train_y.astype(np.float32, copy=False))
    valid_x_t = None if valid_x is None else torch.from_numpy(valid_x.astype(np.int64, copy=False))
    groups = user_groups(train_users)
    torch.manual_seed(seed)
    model = make_model(
        dimension=dimension, fields=train_x.shape[1], embedding_dim=embedding_dim,
        hidden=hidden, dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    random = np.random.default_rng(seed + 17_071)
    history: list[dict] = []
    best_primary, best_epoch, stale = -1.0, 0, 0

    def predict(x_tensor):
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), batch_size * 4):
                chunks.append(model(x_tensor[start:start + batch_size * 4]).numpy())
        return np.concatenate(chunks).astype(np.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        total_losses, bce_losses, list_losses = [], [], []
        for indices_np, local_groups_np, group_count in group_batches(groups, batch_size, random):
            indices = torch.from_numpy(indices_np)
            local_groups = torch.from_numpy(local_groups_np)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x_t[indices])
            labels = train_y_t[indices]
            bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            listce = sigmoid_listce(logits, labels, local_groups, group_count)
            loss = (1.0 - alpha) * bce + alpha * listce
            loss.backward()
            optimizer.step()
            total_losses.append(float(loss.detach()))
            bce_losses.append(float(bce.detach()))
            list_losses.append(float(listce.detach()))
        record = {
            "epoch": epoch,
            "loss": float(np.mean(total_losses)),
            "BCE": float(np.mean(bce_losses)),
            "ListCE(sigmoid)": float(np.mean(list_losses)),
        }
        if valid_x_t is not None and valid_y is not None and valid_users is not None:
            measured, _ = metrics(valid_users, valid_y, predict(valid_x_t))
            record.update({key: measured[key] for key in ("primary", "GAUC", "nDCG@5")})
            if measured["primary"] > best_primary + 1e-5:
                best_primary, best_epoch, stale = measured["primary"], epoch, 0
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    history.append(record)
                    break
        history.append(record)
    if valid_x_t is None:
        best_epoch = epochs
    return model, history, best_epoch, usage(started_wall, started_cpu)


def predict_model(model, x: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    x_tensor = torch.from_numpy(x.astype(np.int64, copy=False))
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size * 4):
            chunks.append(model(x_tensor[start:start + batch_size * 4]).numpy())
    return np.concatenate(chunks).astype(np.float32)


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def actual_user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) % 4


def fixed_residual_audit(users: list[str], labels: np.ndarray, candidate_ranks: np.ndarray) -> dict:
    started_wall, started_cpu = time.monotonic(), time.process_time()
    champion_path = ROOT / "results" / "final-model" / "validation-scores.npz"
    with np.load(champion_path) as stored:
        champion_raw = np.asarray(stored["scores"], dtype=np.float32)
    if len(champion_raw) != len(labels):
        raise RuntimeError("Champion/candidate alignment mismatch")
    champion = within_user_ranks(users, champion_raw)
    blended = champion + FIXED_RESIDUAL_WEIGHT * (candidate_ranks - champion)
    users_array = np.asarray(users, dtype=object)
    folds = np.asarray([actual_user_fold(str(user)) for user in users], dtype=np.int8)

    def measured(mask, values):
        result = evaluate_module.evaluate(users_array[mask].tolist(), labels[mask], values[mask])
        return {"primary": float(result["primary"]), "GAUC": float(result["GAUC"]), "nDCG@5": float(result["nDCG@5"])}

    fold_results = []
    for fold in range(4):
        mask = folds == fold
        baseline = measured(mask, champion)
        residual = measured(mask, blended)
        gains = {key: residual[key] - baseline[key] for key in baseline}
        fold_results.append({
            "fold": fold, "rows": int(mask.sum()),
            "users": len(set(users_array[mask].tolist())),
            "baseline_metrics": baseline, "residual_metrics": residual, "gains": gains,
            "all_metrics_nonnegative": bool(all(value >= -1e-12 for value in gains.values())),
        })
    mask = np.ones(len(labels), dtype=bool)
    baseline = measured(mask, champion)
    residual = measured(mask, blended)
    gains = {key: residual[key] - baseline[key] for key in baseline}
    all_folds = bool(all(item["all_metrics_nonnegative"] for item in fold_results))
    all_global = bool(all(value >= -1e-12 for value in gains.values()))
    report = {
        "experiment": "fixed conservative RCR residual against frozen champion",
        "fixed_weight": FIXED_RESIDUAL_WEIGHT,
        "champion_artifact": str(champion_path.relative_to(ROOT)),
        "baseline_metrics": baseline, "fixed_residual_metrics": residual,
        "fixed_residual_gains": gains, "folds": fold_results,
        "all_four_folds_all_metrics_nonnegative": all_folds,
        "all_global_metrics_nonnegative": all_global,
        "merits_promotion": bool(all_folds and all_global and gains["primary"] > 0),
        "resource_usage": usage(started_wall, started_cpu),
        "integrity_incidents": [], "hidden_test_outcomes_parsed": False,
    }
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def common_config(args) -> dict:
    return {
        "seed": args.seed, "embedding_dim": args.embedding_dim, "hidden": args.hidden,
        "dropout": args.dropout, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
    }


def run_screen(args) -> dict:
    overall_wall, overall_cpu = time.monotonic(), time.process_time()
    core, dev, evaluation = load_screen_rows(args.data_dir.resolve())
    selection_encoded, selection_dimension = encode(core, dev)
    core_x, core_y, core_users = selection_encoded["train"]
    dev_x, dev_y, dev_users = selection_encoded["valid"]
    control_selection, control_history, selected_epoch, selection_usage = train_fixed(
        alpha=0.0, train_x=core_x, train_y=core_y, train_users=core_users,
        dimension=selection_dimension, epochs=args.epochs, patience=args.patience,
        valid_x=dev_x, valid_y=dev_y, valid_users=dev_users, **common_config(args),
    )
    del control_selection
    if selected_epoch <= 0:
        raise RuntimeError("BCE control selected no epoch")

    refit_encoded, refit_dimension = encode(core + dev, evaluation)
    full_x, full_y, full_users = refit_encoded["train"]
    eval_x, eval_y, eval_users = refit_encoded["valid"]
    runs, arrays = [], {}
    for alpha in ALPHA_GRID:
        model, history, _, run_usage = train_fixed(
            alpha=alpha, train_x=full_x, train_y=full_y, train_users=full_users,
            dimension=refit_dimension, epochs=selected_epoch, patience=None,
            **common_config(args),
        )
        raw = predict_model(model, eval_x, args.batch_size)
        measured, ranks = metrics(eval_users, eval_y, raw)
        tag = alpha_tag(alpha)
        arrays[f"alpha_{tag}_raw"] = raw
        arrays[f"alpha_{tag}_ranks"] = ranks
        checkpoint = args.output_dir / f"screen-alpha-{tag}.pt"
        import torch
        torch.save(model.state_dict(), checkpoint)
        runs.append({
            "alpha": alpha, "metrics": measured, "training_history": history,
            "checkpoint": str(checkpoint.relative_to(ROOT)), "resource_usage": run_usage,
        })
    score_path = args.output_dir / "screen-scores.npz"
    np.savez_compressed(score_path, **arrays)
    baseline = next(item for item in runs if item["alpha"] == 0.0)
    eligible = []
    for item in runs:
        if item["alpha"] == 0.0:
            continue
        gains = {key: item["metrics"][key] - baseline["metrics"][key] for key in ("primary", "GAUC", "nDCG@5")}
        item["deltas_vs_control"] = gains
        if gains["primary"] >= 0.0001 and gains["GAUC"] > 0 and gains["nDCG@5"] > 0:
            eligible.append(item)
    selected = max(eligible, key=lambda item: (item["metrics"]["primary"], -item["alpha"])) if eligible else baseline
    gate = bool(eligible)
    report = {
        "experiment": "Bai et al. Regression-Compatible Ranking",
        "paper": PAPER, "status": "screen_passed" if gate else "screen_rejected",
        "merits_confirmation": gate, "selected_alpha": selected["alpha"],
        "locked_epoch_count": selected_epoch,
        "zero_preferring_selection": "alpha=0 unless a nonzero alpha gains >=0.0001 primary and strictly improves GAUC and nDCG@5",
        "alpha_grid_predeclared": list(ALPHA_GRID),
        "protocol": {
            "epoch_selection": "Apr8-11 -> Apr12-14 using alpha=0 control",
            "refit": "Apr8-14", "evaluation": "Apr15-21",
            "apr22_plus_file_opened": False, "hidden_test_outcomes_parsed": False,
            "integrity_incidents": [], "complete_user_groups": True,
        },
        "matched_configuration": {
            **common_config(args), "epochs_max": args.epochs, "patience": args.patience,
            "fields": list(data_module.FIELDS), "backbone": "MaskNet-style pointwise neural model",
            "objective": "(1-alpha)*BCE + alpha*ListCE(sigmoid)",
            "same_initialization_and_group_order": True,
        },
        "rows": {"epoch_train": len(core), "epoch_dev": len(dev), "screen_evaluation": len(evaluation)},
        "control_epoch_selection_history": control_history,
        "control_epoch_selection_resource_usage": selection_usage,
        "runs": runs,
        "artifacts": {"scores": str(score_path.relative_to(ROOT)), "script": str(Path(__file__).resolve().relative_to(ROOT))},
        "resource_usage_total": usage(overall_wall, overall_cpu),
        "hardware": {"architecture": platform.machine(), "logical_cpu_count": os.cpu_count() or 1},
        "recommendation": "run locked confirmation" if gate else "reject RCR; do not access Apr22+",
    }
    SCREEN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SCREEN_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_confirmation(args) -> dict:
    if args.alpha not in ALPHA_GRID or args.alpha == 0.0:
        raise RuntimeError("Confirmation alpha must be a locked nonzero member of the grid")
    if args.locked_epochs <= 0:
        raise RuntimeError("--locked-epochs must come from the passed screen")
    overall_wall, overall_cpu = time.monotonic(), time.process_time()
    training, evaluation = load_confirmation_rows(args.data_dir.resolve())
    encoded, dimension = encode(training, evaluation)
    train_x, train_y, train_users = encoded["train"]
    eval_x, eval_y, eval_users = encoded["valid"]
    runs, arrays = [], {}
    for alpha in (0.0, args.alpha):
        model, history, _, run_usage = train_fixed(
            alpha=alpha, train_x=train_x, train_y=train_y, train_users=train_users,
            dimension=dimension, epochs=args.locked_epochs, patience=None,
            **common_config(args),
        )
        raw = predict_model(model, eval_x, args.batch_size)
        measured, ranks = metrics(eval_users, eval_y, raw)
        tag = alpha_tag(alpha)
        arrays[f"alpha_{tag}_raw"] = raw
        arrays[f"alpha_{tag}_ranks"] = ranks
        checkpoint = args.output_dir / f"confirmation-alpha-{tag}.pt"
        import torch
        torch.save(model.state_dict(), checkpoint)
        runs.append({"alpha": alpha, "metrics": measured, "training_history": history, "checkpoint": str(checkpoint.relative_to(ROOT)), "resource_usage": run_usage})
    score_path = args.output_dir / "confirmation-scores.npz"
    np.savez_compressed(score_path, **arrays)
    control, candidate = runs
    gains = {key: candidate["metrics"][key] - control["metrics"][key] for key in ("primary", "GAUC", "nDCG@5")}
    candidate_ranks = arrays[f"alpha_{alpha_tag(args.alpha)}_ranks"]
    audit = fixed_residual_audit(eval_users, eval_y, candidate_ranks)
    report = {
        "experiment": "locked RCR confirmation", "paper": PAPER,
        "status": "confirmation_evaluated", "selected_alpha_locked": args.alpha,
        "locked_epoch_count": args.locked_epochs, "control": control,
        "candidate": candidate, "deltas_vs_control": gains,
        "matched_candidate_improves_all_metrics": bool(all(value > 0 for value in gains.values())),
        "protocol": {
            "train": "Apr8-21", "evaluation": "Apr22-28", "retuning": False,
            "outcome_fields_parsed_only_for": "Apr22-28",
            "hidden_test_outcomes_parsed": False, "integrity_incidents": [],
        },
        "residual_audit": {
            "artifact": str(AUDIT_REPORT.relative_to(ROOT)), "fixed_weight": FIXED_RESIDUAL_WEIGHT,
            "fixed_residual_metrics": audit["fixed_residual_metrics"],
            "fixed_residual_gains": audit["fixed_residual_gains"],
            "all_four_folds_all_metrics_nonnegative": audit["all_four_folds_all_metrics_nonnegative"],
            "all_global_metrics_nonnegative": audit["all_global_metrics_nonnegative"],
            "merits_promotion": audit["merits_promotion"],
        },
        "artifacts": {"scores": str(score_path.relative_to(ROOT)), "script": str(Path(__file__).resolve().relative_to(ROOT))},
        "resource_usage_total": usage(overall_wall, overall_cpu),
        "recommendation": "eligible for integrity review; do not auto-promote" if audit["merits_promotion"] else "reject; retain frozen champion",
    }
    CONFIRM_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--locked-epochs", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=RUNTIME)
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
    torch.set_num_threads(max(1, min(args.threads, 12)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return 0 if (run_confirmation(args) if args.confirmation else run_screen(args)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
