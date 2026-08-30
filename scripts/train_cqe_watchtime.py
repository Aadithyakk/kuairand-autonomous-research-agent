#!/usr/bin/env python3
"""Conditional Quantile Estimation watch-time ranking for KuaiRand.

The train-only screen uses Apr 8--11 for fit/config selection, Apr 12--14 for
selection, then refits Apr 8--14 and evaluates once on Apr 15--21.  The Apr 22+
file is unopened unless explicit locked confirmation is requested.
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
RUNTIME = ROOT / "runtime" / "parallel-calibrated-ranking" / "cqe"
SCREEN_REPORT = ROOT / "results" / "calibrated-ranking" / "cqe-screen.json"
CONFIRM_REPORT = ROOT / "results" / "calibrated-ranking" / "cqe-confirmation.json"
AUDIT_REPORT = ROOT / "results" / "calibrated-ranking" / "cqe-residual-audit.json"
PAPER = "https://arxiv.org/abs/2407.12223"
CONTROL_AUX_WEIGHT = 0.02
FIXED_RESIDUAL_WEIGHT = 0.05
MAX_WATCH_MS = 600_000.0
TEMPERATURE = 0.15
QUANTILE_CONFIGS = (
    {"levels": (0.10, 0.25, 0.50, 0.75, 0.90), "aux_weight": 0.02, "blend": 0.10},
    {"levels": (0.10, 0.25, 0.50, 0.75, 0.90), "aux_weight": 0.05, "blend": 0.10},
    {"levels": (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95), "aux_weight": 0.02, "blend": 0.10},
    {"levels": (0.10, 0.25, 0.50, 0.75, 0.90), "aux_weight": 0.02, "blend": 0.25},
)
MAX_QUANTILES = 7


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_module = load_module("cqe_official_data", STARTER / "data.py")
evaluate_module = load_module("cqe_official_evaluate", STARTER / "evaluate.py")


def load_authors(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[row["video_id"]] = row["author_id"]
    return result


def parse_row(row: dict[str, str], authors: dict[str, str]) -> tuple:
    return (
        int(row["date"]), row["user_id"], row["video_id"],
        authors.get(row["video_id"], "UNK"), row["tab"],
        float(row["duration_ms"]), 1 if row["long_view"] != "0" else 0,
        float(row["play_time_ms"]),
    )


def load_screen_rows(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    authors = load_authors(data_dir)
    core: list[tuple] = []
    dev: list[tuple] = []
    screen: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            item = parse_row(row, authors)
            if item[0] <= 20220411:
                core.append(item)
            elif item[0] <= 20220414:
                dev.append(item)
            elif item[0] <= 20220421:
                screen.append(item)
            else:
                raise RuntimeError(f"Unexpected first-log date {item[0]}")
    if len(core) + len(dev) + len(screen) != 1_141_112:
        raise RuntimeError(f"Unexpected screen rows {len(core)}, {len(dev)}, {len(screen)}")
    return core, dev, screen


def load_confirmation_rows(data_dir: Path) -> tuple[list[tuple], list[tuple]]:
    """Parse outcome columns only after a raw-line Apr 22--28 date check."""
    authors = load_authors(data_dir)
    train: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            train.append(parse_row(row, authors))
    valid: list[tuple] = []
    with (data_dir / "log_standard_4_22_to_5_08_pure.csv").open(encoding="utf-8") as stream:
        header = next(stream)
        fieldnames = next(csv.reader([header]))
        if fieldnames[:3] != ["user_id", "video_id", "date"]:
            raise RuntimeError(f"Unexpected later-log header {fieldnames[:3]}")
        for line in stream:
            prefix = line.split(",", 3)
            if len(prefix) != 4:
                raise RuntimeError("Malformed later-log row")
            date = int(prefix[2])
            if date > 20220428:
                continue
            if date < 20220422:
                raise RuntimeError(f"Unexpected later-log date {date}")
            values = next(csv.reader([line]))
            valid.append(parse_row(dict(zip(fieldnames, values, strict=True)), authors))
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError(f"Unexpected confirmation rows {len(train)}, {len(valid)}")
    return train, valid


def encode(train: list[tuple], valid: list[tuple]) -> tuple[dict, int]:
    return data_module.encode({"train": train, "valid": valid, "test": []})


def watch_targets(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    watch_ms = np.asarray([row[7] for row in rows], dtype=np.float32)
    duration_ms = np.asarray([row[5] for row in rows], dtype=np.float32)
    watch_log = np.log1p(np.clip(watch_ms, 0.0, MAX_WATCH_MS) / 1000.0).astype(np.float32)
    threshold_log = np.log1p(np.minimum(np.maximum(duration_ms, 1.0), 18_000.0) / 1000.0).astype(np.float32)
    return watch_log, threshold_log


def within_user_ranks(users: list[str], scores: np.ndarray) -> np.ndarray:
    grouped: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        grouped.setdefault(str(user), []).append(index)
    output = np.empty(len(scores), dtype=np.float32)
    for row_list in grouped.values():
        rows = np.asarray(row_list, dtype=np.int64)
        order = np.argsort(scores[rows], kind="stable")
        ranks = np.empty(len(rows), dtype=np.float32)
        ranks[order] = np.arange(len(rows), dtype=np.float32)
        output[rows] = ranks / max(len(rows) - 1, 1)
    return output


def evaluate(users: list[str], labels: np.ndarray, scores: np.ndarray) -> tuple[dict, np.ndarray]:
    ranks = within_user_ranks(users, scores)
    result = evaluate_module.evaluate(users, labels, ranks)
    return {
        "primary": float(result["primary"]), "GAUC": float(result["GAUC"]),
        "nDCG@5": float(result["nDCG@5"]), "rows": int(result["rows"]),
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
        "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3),
        "cpu_hours": round(cpu / 3600, 6),
        "cpu_utilization_percent": round(100 * cpu / max(wall, 1e-9), 2),
        "peak_rss_mb": round(peak_rss_mb(), 3), "device": "cpu",
        "gpu_count": 0, "gpu_hours": 0.0, "peak_gpu_memory_mb": 0.0,
    }


def make_model(dimension: int, fields: int, embedding_dim: int, hidden: int, dropout: float):
    import torch
    from torch import nn

    class CQEBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            shared_width = max(hidden // 2, 8)
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.shared = nn.Sequential(
                nn.Linear(fields * embedding_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, shared_width), nn.ReLU(), nn.Dropout(dropout),
            )
            self.long_head = nn.Linear(shared_width, 1)
            self.mean_watch_head = nn.Linear(shared_width, 1)
            self.quantile_head = nn.Linear(shared_width, MAX_QUANTILES)
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)
            nn.init.constant_(self.quantile_head.bias[1:], -2.0)

        def forward(self, x):
            embedded = self.embedding(x)
            summed = embedded.sum(dim=1)
            fm = 0.5 * (summed.square() - embedded.square().sum(dim=1)).sum(dim=1)
            linear = self.linear(x).sum(dim=1).squeeze(-1)
            shared = self.shared(embedded.flatten(start_dim=1))
            long_logit = linear + fm + self.long_head(shared).squeeze(-1)
            mean_watch = torch.nn.functional.softplus(self.mean_watch_head(shared).squeeze(-1))
            raw = self.quantile_head(shared)
            first = torch.nn.functional.softplus(raw[:, :1])
            increments = torch.nn.functional.softplus(raw[:, 1:])
            quantiles = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
            return long_logit, mean_watch, quantiles

    return CQEBackbone()


def pinball_loss(prediction, target, levels):
    error = target[:, None] - prediction
    return np_torch_max(levels * error, (levels - 1.0) * error).mean()


def np_torch_max(left, right):
    import torch
    return torch.maximum(left, right)


def quantile_weights(levels: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(levels, dtype=np.float32)
    boundaries = np.concatenate(([0.0], (values[:-1] + values[1:]) / 2.0, [1.0]))
    return np.diff(boundaries).astype(np.float32)


def cqe_probability(quantiles: np.ndarray, threshold_log: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    weights = quantile_weights(levels)
    difference = np.clip((quantiles - threshold_log[:, None]) / TEMPERATURE, -30.0, 30.0)
    survival_votes = 1.0 / (1.0 + np.exp(-difference))
    return (survival_votes * weights[None, :]).sum(axis=1).astype(np.float32)


def train_model(
    *, mode: str, config: dict | None, train_x: np.ndarray, train_y: np.ndarray,
    train_watch: np.ndarray, dimension: int, epochs: int, seed: int,
    embedding_dim: int, hidden: int, dropout: float, learning_rate: float,
    weight_decay: float, batch_size: int, valid_x: np.ndarray | None = None,
    valid_y: np.ndarray | None = None, valid_users: list[str] | None = None,
    valid_threshold: np.ndarray | None = None, patience: int | None = None,
) -> tuple[object, list[dict], int, dict]:
    import torch
    from torch import nn

    started_wall, started_cpu = time.monotonic(), time.process_time()
    x_t = torch.from_numpy(train_x.astype(np.int64, copy=False))
    y_t = torch.from_numpy(train_y.astype(np.float32, copy=False))
    watch_t = torch.from_numpy(train_watch.astype(np.float32, copy=False))
    valid_x_t = None if valid_x is None else torch.from_numpy(valid_x.astype(np.int64, copy=False))
    torch.manual_seed(seed)
    model = make_model(dimension, train_x.shape[1], embedding_dim, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    random = np.random.default_rng(seed + 31_337)
    levels_t = None
    quantile_count = 0
    if mode == "cqe":
        assert config is not None
        quantile_count = len(config["levels"])
        levels_t = torch.tensor(config["levels"], dtype=torch.float32)
    best_primary, best_epoch, stale = -1.0, 0, 0
    history: list[dict] = []

    def predict(x_tensor):
        model.eval()
        long_parts, quantile_parts = [], []
        with torch.no_grad():
            for start in range(0, len(x_tensor), batch_size * 4):
                long_logit, _, quantiles = model(x_tensor[start:start + batch_size * 4])
                long_parts.append(torch.sigmoid(long_logit).numpy())
                quantile_parts.append(quantiles[:, :quantile_count].numpy())
        long_probability = np.concatenate(long_parts).astype(np.float32)
        quantile_values = np.concatenate(quantile_parts).astype(np.float32) if quantile_count else None
        return long_probability, quantile_values

    for epoch in range(1, epochs + 1):
        model.train()
        losses, bces, auxiliaries = [], [], []
        order = random.permutation(len(train_y))
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            long_logit, mean_watch, quantiles = model(x_t[indices])
            bce = nn.functional.binary_cross_entropy_with_logits(long_logit, y_t[indices])
            if mode == "control":
                auxiliary = nn.functional.smooth_l1_loss(mean_watch, watch_t[indices])
                loss = bce + CONTROL_AUX_WEIGHT * auxiliary
            else:
                auxiliary = pinball_loss(quantiles[:, :quantile_count], watch_t[indices], levels_t)
                loss = bce + float(config["aux_weight"]) * auxiliary
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach())); bces.append(float(bce.detach())); auxiliaries.append(float(auxiliary.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "BCE": float(np.mean(bces)), "auxiliary_loss": float(np.mean(auxiliaries))}
        if valid_x_t is not None and valid_y is not None and valid_users is not None:
            long_probability, quantiles = predict(valid_x_t)
            if mode == "cqe":
                assert valid_threshold is not None and quantiles is not None and config is not None
                probability = (1.0 - float(config["blend"])) * long_probability + float(config["blend"]) * cqe_probability(quantiles, valid_threshold, config["levels"])
            else:
                probability = long_probability
            measured, _ = evaluate(valid_users, valid_y, probability)
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


def score_model(model, x: np.ndarray, threshold: np.ndarray, mode: str, config: dict | None, batch_size: int) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    x_t = torch.from_numpy(x.astype(np.int64, copy=False))
    long_parts, quantile_parts = [], []
    count = 0 if config is None else len(config["levels"])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_t), batch_size * 4):
            long_logit, _, quantiles = model(x_t[start:start + batch_size * 4])
            long_parts.append(torch.sigmoid(long_logit).numpy())
            if count:
                quantile_parts.append(quantiles[:, :count].numpy())
    long_probability = np.concatenate(long_parts).astype(np.float32)
    quantile_values = np.concatenate(quantile_parts).astype(np.float32) if quantile_parts else None
    if mode == "control":
        return long_probability, None
    assert config is not None and quantile_values is not None
    quantile_probability = cqe_probability(quantile_values, threshold, config["levels"])
    final = (1.0 - float(config["blend"])) * long_probability + float(config["blend"]) * quantile_probability
    return final.astype(np.float32), quantile_values


def common(args) -> dict:
    return {"seed": args.seed, "embedding_dim": args.embedding_dim, "hidden": args.hidden, "dropout": args.dropout, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size}


def actual_user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) % 4


def residual_audit(users: list[str], labels: np.ndarray, candidate_ranks: np.ndarray) -> dict:
    started_wall, started_cpu = time.monotonic(), time.process_time()
    champion_path = ROOT / "results" / "final-model" / "validation-scores.npz"
    with np.load(champion_path) as stored:
        champion = within_user_ranks(users, np.asarray(stored["scores"], dtype=np.float32))
    blended = champion + FIXED_RESIDUAL_WEIGHT * (candidate_ranks - champion)
    users_array = np.asarray(users, dtype=object)
    folds = np.asarray([actual_user_fold(str(user)) for user in users], dtype=np.int8)
    def measured(mask, scores):
        result = evaluate_module.evaluate(users_array[mask].tolist(), labels[mask], scores[mask])
        return {"primary": float(result["primary"]), "GAUC": float(result["GAUC"]), "nDCG@5": float(result["nDCG@5"])}
    fold_results = []
    for fold in range(4):
        mask = folds == fold
        baseline, candidate = measured(mask, champion), measured(mask, blended)
        gains = {key: candidate[key] - baseline[key] for key in baseline}
        fold_results.append({"fold": fold, "rows": int(mask.sum()), "users": len(set(users_array[mask].tolist())), "baseline_metrics": baseline, "residual_metrics": candidate, "gains": gains, "all_metrics_nonnegative": bool(all(value >= -1e-12 for value in gains.values()))})
    mask = np.ones(len(labels), dtype=bool)
    baseline, candidate = measured(mask, champion), measured(mask, blended)
    gains = {key: candidate[key] - baseline[key] for key in baseline}
    all_folds = bool(all(item["all_metrics_nonnegative"] for item in fold_results))
    all_global = bool(all(value >= -1e-12 for value in gains.values()))
    report = {"experiment": "fixed 5% CQE residual against frozen champion", "fixed_weight": FIXED_RESIDUAL_WEIGHT, "baseline_metrics": baseline, "fixed_residual_metrics": candidate, "fixed_residual_gains": gains, "folds": fold_results, "all_four_folds_all_metrics_nonnegative": all_folds, "all_global_metrics_nonnegative": all_global, "merits_promotion": bool(all_folds and all_global and gains["primary"] > 0), "resource_usage": usage(started_wall, started_cpu), "integrity_incidents": [], "hidden_test_outcomes_parsed": False}
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_screen(args) -> dict:
    overall_wall, overall_cpu = time.monotonic(), time.process_time()
    core, dev, screen = load_screen_rows(args.data_dir.resolve())
    selection_encoded, selection_dimension = encode(core, dev)
    core_x, core_y, _ = selection_encoded["train"]
    dev_x, dev_y, dev_users = selection_encoded["valid"]
    core_watch, _ = watch_targets(core)
    _, dev_threshold = watch_targets(dev)
    control_model, control_history, selected_epoch, control_usage = train_model(mode="control", config=None, train_x=core_x, train_y=core_y, train_watch=core_watch, dimension=selection_dimension, epochs=args.epochs, valid_x=dev_x, valid_y=dev_y, valid_users=dev_users, valid_threshold=dev_threshold, patience=args.patience, **common(args))
    control_scores, _ = score_model(control_model, dev_x, dev_threshold, "control", None, args.batch_size)
    control_selection_metrics, _ = evaluate(dev_users, dev_y, control_scores)
    config_runs = []
    for index, config in enumerate(QUANTILE_CONFIGS):
        model, history, _, run_usage = train_model(mode="cqe", config=config, train_x=core_x, train_y=core_y, train_watch=core_watch, dimension=selection_dimension, epochs=selected_epoch, patience=None, **common(args))
        scores, quantiles = score_model(model, dev_x, dev_threshold, "cqe", config, args.batch_size)
        measured, _ = evaluate(dev_users, dev_y, scores)
        config_runs.append({"config_index": index, "config": {**config, "levels": list(config["levels"])}, "selection_metrics": measured, "deltas_vs_control": {key: measured[key] - control_selection_metrics[key] for key in ("primary", "GAUC", "nDCG@5")}, "training_history": history, "resource_usage": run_usage, "quantile_crossing_rate": float(np.mean(np.diff(quantiles, axis=1) < -1e-7))})
    selected = max(config_runs, key=lambda item: (item["selection_metrics"]["primary"], -item["config_index"]))
    selected_config = QUANTILE_CONFIGS[selected["config_index"]]

    refit_encoded, refit_dimension = encode(core + dev, screen)
    full_x, full_y, _ = refit_encoded["train"]
    screen_x, screen_y, screen_users = refit_encoded["valid"]
    full_watch, _ = watch_targets(core + dev)
    _, screen_threshold = watch_targets(screen)
    artifacts = {}
    final_runs = []
    for mode, config in (("control", None), ("cqe", selected_config)):
        model, history, _, run_usage = train_model(mode=mode, config=config, train_x=full_x, train_y=full_y, train_watch=full_watch, dimension=refit_dimension, epochs=selected_epoch, patience=None, **common(args))
        scores, quantiles = score_model(model, screen_x, screen_threshold, mode, config, args.batch_size)
        measured, ranks = evaluate(screen_users, screen_y, scores)
        artifacts[f"{mode}_scores"] = scores; artifacts[f"{mode}_ranks"] = ranks
        if quantiles is not None: artifacts["cqe_quantiles"] = quantiles
        import torch
        checkpoint = args.output_dir / f"screen-{mode}.pt"; torch.save(model.state_dict(), checkpoint)
        final_runs.append({"mode": mode, "metrics": measured, "training_history": history, "resource_usage": run_usage, "checkpoint": str(checkpoint.relative_to(ROOT))})
    score_path = args.output_dir / "screen-scores.npz"; np.savez_compressed(score_path, **artifacts)
    control, candidate = final_runs
    gains = {key: candidate["metrics"][key] - control["metrics"][key] for key in ("primary", "GAUC", "nDCG@5")}
    gate = bool(gains["primary"] >= 0.0001 and gains["GAUC"] > 0 and gains["nDCG@5"] > 0)
    report = {"experiment": "Conditional Quantile Estimation watch-time ranking", "paper": PAPER, "status": "screen_passed" if gate else "screen_rejected", "merits_confirmation": gate, "selected_config_index": selected["config_index"], "selected_config": {**selected_config, "levels": list(selected_config["levels"])}, "locked_epoch_count": selected_epoch, "predeclared_config_grid": [{**config, "levels": list(config["levels"])} for config in QUANTILE_CONFIGS], "control_selection_metrics": control_selection_metrics, "config_selection_runs": config_runs, "control": control, "candidate": candidate, "deltas_vs_control": gains, "protocol": {"selection_fit": "Apr8-11", "config_selection": "Apr12-14", "refit": "Apr8-14", "screen": "Apr15-21", "apr22_plus_file_opened": False, "hidden_test_outcomes_parsed": False, "integrity_incidents": []}, "matched_configuration": {**common(args), "control_auxiliary": f"{CONTROL_AUX_WEIGHT}*SmoothL1(log_watch)", "candidate_auxiliary": "pinball loss over monotonic quantile heads", "threshold": "min(duration_ms,18000)", "watch_transform": f"log1p(clip(play_time_ms,0,{int(MAX_WATCH_MS)})/1000)", "same_initialization_and_row_order": True}, "resources": {"control_epoch_selection": control_usage, "total": usage(overall_wall, overall_cpu)}, "artifacts": {"scores": str(score_path.relative_to(ROOT)), "script": str(Path(__file__).resolve().relative_to(ROOT))}, "recommendation": "run locked confirmation" if gate else "reject CQE; do not access Apr22+"}
    SCREEN_REPORT.parent.mkdir(parents=True, exist_ok=True); SCREEN_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"); print(json.dumps(report, indent=2, sort_keys=True)); return report


def run_confirmation(args) -> dict:
    if not 0 <= args.config_index < len(QUANTILE_CONFIGS) or args.locked_epochs <= 0:
        raise RuntimeError("Confirmation requires locked screen config and epoch count")
    overall_wall, overall_cpu = time.monotonic(), time.process_time()
    config = QUANTILE_CONFIGS[args.config_index]
    train, valid = load_confirmation_rows(args.data_dir.resolve())
    encoded, dimension = encode(train, valid)
    train_x, train_y, _ = encoded["train"]; valid_x, valid_y, valid_users = encoded["valid"]
    train_watch, _ = watch_targets(train); _, valid_threshold = watch_targets(valid)
    arrays = {}; runs = []
    for mode, run_config in (("control", None), ("cqe", config)):
        model, history, _, run_usage = train_model(mode=mode, config=run_config, train_x=train_x, train_y=train_y, train_watch=train_watch, dimension=dimension, epochs=args.locked_epochs, patience=None, **common(args))
        scores, quantiles = score_model(model, valid_x, valid_threshold, mode, run_config, args.batch_size)
        measured, ranks = evaluate(valid_users, valid_y, scores)
        arrays[f"{mode}_scores"] = scores; arrays[f"{mode}_ranks"] = ranks
        if quantiles is not None: arrays["cqe_quantiles"] = quantiles
        import torch
        checkpoint = args.output_dir / f"confirmation-{mode}.pt"; torch.save(model.state_dict(), checkpoint)
        runs.append({"mode": mode, "metrics": measured, "training_history": history, "resource_usage": run_usage, "checkpoint": str(checkpoint.relative_to(ROOT))})
    score_path = args.output_dir / "confirmation-scores.npz"; np.savez_compressed(score_path, **arrays)
    control, candidate = runs
    gains = {key: candidate["metrics"][key] - control["metrics"][key] for key in ("primary", "GAUC", "nDCG@5")}
    audit = residual_audit(valid_users, valid_y, arrays["cqe_ranks"])
    report = {"experiment": "locked CQE watch-time confirmation", "paper": PAPER, "status": "confirmation_evaluated", "selected_config_index_locked": args.config_index, "selected_config": {**config, "levels": list(config["levels"])}, "locked_epoch_count": args.locked_epochs, "control": control, "candidate": candidate, "deltas_vs_control": gains, "matched_candidate_improves_all_metrics": bool(all(value > 0 for value in gains.values())), "protocol": {"train": "Apr8-21", "evaluation": "Apr22-28", "retuning": False, "outcome_fields_parsed_only_for": "Apr22-28", "hidden_test_outcomes_parsed": False, "integrity_incidents": []}, "residual_audit": {"artifact": str(AUDIT_REPORT.relative_to(ROOT)), "fixed_weight": FIXED_RESIDUAL_WEIGHT, "fixed_residual_metrics": audit["fixed_residual_metrics"], "fixed_residual_gains": audit["fixed_residual_gains"], "all_four_folds_all_metrics_nonnegative": audit["all_four_folds_all_metrics_nonnegative"], "all_global_metrics_nonnegative": audit["all_global_metrics_nonnegative"], "merits_promotion": audit["merits_promotion"]}, "resources": {"total": usage(overall_wall, overall_cpu)}, "artifacts": {"scores": str(score_path.relative_to(ROOT)), "script": str(Path(__file__).resolve().relative_to(ROOT))}, "recommendation": "eligible for integrity review; do not auto-promote" if audit["merits_promotion"] else "reject; retain frozen champion"}
    CONFIRM_REPORT.parent.mkdir(parents=True, exist_ok=True); CONFIRM_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"); print(json.dumps(report, indent=2, sort_keys=True)); return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--confirmation", action="store_true"); parser.add_argument("--config-index", type=int, default=-1); parser.add_argument("--locked-epochs", type=int, default=0); parser.add_argument("--data-dir", type=Path, default=DATA); parser.add_argument("--output-dir", type=Path, default=RUNTIME); parser.add_argument("--seed", type=int, default=260830); parser.add_argument("--threads", type=int, default=6); parser.add_argument("--embedding-dim", type=int, default=12); parser.add_argument("--hidden", type=int, default=64); parser.add_argument("--dropout", type=float, default=0.05); parser.add_argument("--learning-rate", type=float, default=0.001); parser.add_argument("--weight-decay", type=float, default=1e-6); parser.add_argument("--batch-size", type=int, default=8192); parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--patience", type=int, default=4); args = parser.parse_args()
    import torch
    torch.set_num_threads(max(1, min(args.threads, 12))); args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_confirmation(args) if args.confirmation else run_screen(args)
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
