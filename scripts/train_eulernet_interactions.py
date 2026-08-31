#!/usr/bin/env python3
"""Compact EulerNet-style feature-interaction ablation for KuaiRand."""
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
RUNTIME = ROOT / "runtime" / "parallel-calibrated-ranking" / "eulernet"
SCREEN_REPORT = ROOT / "results" / "calibrated-ranking" / "eulernet-screen.json"
CONFIRM_REPORT = ROOT / "results" / "calibrated-ranking" / "eulernet-confirmation.json"
AUDIT_REPORT = ROOT / "results" / "calibrated-ranking" / "eulernet-residual-audit.json"
PAPER = "https://arxiv.org/abs/2304.10711"
FIXED_RESIDUAL_WEIGHT = 0.05
ARCHITECTURES = (
    {"order_fields": (6,), "layer_norm": True},
    {"order_fields": (6, 4), "layer_norm": True},
    {"order_fields": (8,), "layer_norm": False},
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_module = load_module("eulernet_official_data", STARTER / "data.py")
evaluate_module = load_module("eulernet_official_evaluate", STARTER / "evaluate.py")


def load_authors(data_dir: Path) -> dict[str, str]:
    authors: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            authors[row["video_id"]] = row["author_id"]
    return authors


def parsed(row: dict[str, str], authors: dict[str, str]) -> tuple:
    return (
        int(row["date"]), row["user_id"], row["video_id"],
        authors.get(row["video_id"], "UNK"), row["tab"],
        float(row["duration_ms"]), 1 if row["long_view"] != "0" else 0,
    )


def load_screen_rows(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    authors = load_authors(data_dir)
    core: list[tuple] = []
    dev: list[tuple] = []
    screen: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            item = parsed(row, authors)
            if item[0] <= 20220411:
                core.append(item)
            elif item[0] <= 20220414:
                dev.append(item)
            elif item[0] <= 20220421:
                screen.append(item)
            else:
                raise RuntimeError(f"Unexpected first-log date {item[0]}")
    if len(core) + len(dev) + len(screen) != 1_141_112:
        raise RuntimeError(f"Unexpected screen row counts {len(core)}, {len(dev)}, {len(screen)}")
    return core, dev, screen


def load_confirmation_rows(data_dir: Path) -> tuple[list[tuple], list[tuple]]:
    """Raw date-prefix filter prevents parsing any Apr 29+ outcomes."""
    authors = load_authors(data_dir)
    train: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            train.append(parsed(row, authors))
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
            valid.append(parsed(dict(zip(fieldnames, values, strict=True)), authors))
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError(f"Unexpected confirmation rows {len(train)}, {len(valid)}")
    return train, valid


def encode(train: list[tuple], valid: list[tuple]) -> tuple[dict, int]:
    return data_module.encode({"train": train, "valid": valid, "test": []})


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
        "peak_rss_mb": round(peak_rss_mb(), 3), "device": "cpu", "gpu_count": 0,
        "gpu_hours": 0.0, "peak_gpu_memory_mb": 0.0,
    }


def make_model(
    *, dimension: int, fields: int, embedding_dim: int, hidden: int,
    dropout: float, architecture: dict,
):
    import torch
    from torch import nn

    class EulerLayer(nn.Module):
        def __init__(self, input_fields: int, output_fields: int, normalize: bool):
            super().__init__()
            if input_fields == output_fields:
                orders = torch.eye(input_fields)
            else:
                orders = torch.softmax(torch.randn(input_fields, output_fields) / 0.1, dim=0)
            self.orders = nn.Parameter(orders)
            self.real_projection = nn.Linear(input_fields * embedding_dim, output_fields * embedding_dim)
            self.imag_projection = nn.Linear(input_fields * embedding_dim, output_fields * embedding_dim)
            self.modulus_bias = nn.Parameter(torch.zeros(1, embedding_dim, output_fields))
            self.phase_bias = nn.Parameter(torch.zeros(1, embedding_dim, output_fields))
            self.real_norm = nn.LayerNorm(embedding_dim) if normalize else nn.Identity()
            self.imag_norm = nn.LayerNorm(embedding_dim) if normalize else nn.Identity()
            self.output_fields = output_fields
            nn.init.normal_(self.real_projection.weight, std=0.01)
            nn.init.normal_(self.imag_projection.weight, std=0.01)

        def forward(self, values):
            real, imag = values
            modulus_log = 0.5 * torch.log(real.square() + imag.square() + 1e-8)
            phase = torch.atan2(imag, real)
            modulus_log = modulus_log.transpose(1, 2) @ self.orders + self.modulus_bias
            phase = phase.transpose(1, 2) @ self.orders + self.phase_bias
            # The clamp is the CPU-safety bound: explicit magnitude cannot overflow.
            modulus = torch.exp(torch.clamp(modulus_log, -6.0, 6.0)).transpose(1, 2)
            phase = phase.transpose(1, 2)
            batch = real.shape[0]
            implicit_real = torch.relu(self.real_projection(real.flatten(start_dim=1))).reshape(batch, self.output_fields, embedding_dim)
            implicit_imag = torch.relu(self.imag_projection(imag.flatten(start_dim=1))).reshape(batch, self.output_fields, embedding_dim)
            out_real = self.real_norm(implicit_real + modulus * torch.cos(phase))
            out_imag = self.imag_norm(implicit_imag + modulus * torch.sin(phase))
            return out_real, out_imag

    class CompactEulerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.control_mlp = nn.Sequential(
                nn.Linear(fields * embedding_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, max(hidden // 2, 8)), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(max(hidden // 2, 8), 1),
            )
            layers = []
            input_fields = fields
            for output_fields in architecture["order_fields"]:
                layers.append(EulerLayer(input_fields, output_fields, architecture["layer_norm"]))
                input_fields = output_fields
            self.euler_layers = nn.ModuleList(layers)
            self.modulus = nn.Parameter(torch.ones(1, fields, 1))
            self.euler_head = nn.Linear(2 * input_fields * embedding_dim, 1)
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)
            nn.init.xavier_normal_(self.euler_head.weight)

        def forward(self, x, mode: str):
            embedded = self.embedding(x)
            wide = self.linear(x).sum(dim=1).squeeze(-1)
            if mode == "control":
                return wide + self.control_mlp(embedded.flatten(start_dim=1)).squeeze(-1)
            real = self.modulus * torch.cos(embedded)
            imag = self.modulus * torch.sin(embedded)
            for layer in self.euler_layers:
                real, imag = layer((real, imag))
            interaction = torch.cat([real.flatten(start_dim=1), imag.flatten(start_dim=1)], dim=1)
            return wide + self.euler_head(interaction).squeeze(-1)

    return CompactEulerModel()


def active_parameter_count(model, mode: str) -> int:
    prefixes = ("linear", "embedding", "control_mlp") if mode == "control" else (
        "linear", "embedding", "euler_layers", "modulus", "euler_head"
    )
    return sum(value.numel() for name, value in model.named_parameters() if name.startswith(prefixes))


def train_model(
    *, mode: str, architecture: dict, train_x: np.ndarray, train_y: np.ndarray,
    dimension: int, epochs: int, seed: int, embedding_dim: int, hidden: int,
    dropout: float, learning_rate: float, weight_decay: float, batch_size: int,
    valid_x: np.ndarray | None = None, valid_y: np.ndarray | None = None,
    valid_users: list[str] | None = None, patience: int | None = None,
) -> tuple[object, list[dict], int, dict, int]:
    import torch
    from torch import nn

    started_wall, started_cpu = time.monotonic(), time.process_time()
    x_t = torch.from_numpy(train_x.astype(np.int64, copy=False))
    y_t = torch.from_numpy(train_y.astype(np.float32, copy=False))
    valid_x_t = None if valid_x is None else torch.from_numpy(valid_x.astype(np.int64, copy=False))
    torch.manual_seed(seed)
    model = make_model(dimension=dimension, fields=train_x.shape[1], embedding_dim=embedding_dim, hidden=hidden, dropout=dropout, architecture=architecture)
    # Reinitialize shared categorical parameters from an architecture-independent
    # RNG point so depth cannot change the matched embedding initialization.
    torch.manual_seed(seed + 101)
    nn.init.normal_(model.embedding.weight, std=0.01)
    nn.init.zeros_(model.linear.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    random = np.random.default_rng(seed + 51_091)
    best_primary, best_epoch, stale = -1.0, 0, 0
    history: list[dict] = []
    nonfinite_batches = 0

    def predict(x_tensor):
        model.eval(); parts = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), batch_size * 4):
                parts.append(model(x_tensor[start:start + batch_size * 4], mode).numpy())
        return np.concatenate(parts).astype(np.float32)

    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        order = random.permutation(len(train_y))
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_t[indices], mode)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t[indices])
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                nonfinite_batches += 1
                raise FloatingPointError(f"Non-finite {mode} batch at epoch {epoch}")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "nonfinite_batches": nonfinite_batches}
        if valid_x_t is not None and valid_y is not None and valid_users is not None:
            scores = predict(valid_x_t)
            if not bool(np.isfinite(scores).all()):
                raise FloatingPointError(f"Non-finite {mode} validation scores")
            measured, _ = evaluate(valid_users, valid_y, scores)
            record.update({key: measured[key] for key in ("primary", "GAUC", "nDCG@5")})
            if measured["primary"] > best_primary + 1e-5:
                best_primary, best_epoch, stale = measured["primary"], epoch, 0
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    history.append(record); break
        history.append(record)
    if valid_x_t is None:
        best_epoch = epochs
    details = usage(started_wall, started_cpu)
    details["nonfinite_batches"] = nonfinite_batches
    return model, history, best_epoch, details, active_parameter_count(model, mode)


def predict_model(model, x: np.ndarray, mode: str, batch_size: int) -> np.ndarray:
    import torch
    x_t = torch.from_numpy(x.astype(np.int64, copy=False)); parts = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(x_t), batch_size * 4):
            parts.append(model(x_t[start:start + batch_size * 4], mode).numpy())
    result = np.concatenate(parts).astype(np.float32)
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError(f"Non-finite {mode} predictions")
    return result


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
        value = evaluate_module.evaluate(users_array[mask].tolist(), labels[mask], scores[mask])
        return {"primary": float(value["primary"]), "GAUC": float(value["GAUC"]), "nDCG@5": float(value["nDCG@5"])}
    fold_results = []
    for fold in range(4):
        mask = folds == fold; baseline = measured(mask, champion); candidate = measured(mask, blended)
        gains = {key: candidate[key] - baseline[key] for key in baseline}
        fold_results.append({"fold": fold, "rows": int(mask.sum()), "users": len(set(users_array[mask].tolist())), "baseline_metrics": baseline, "residual_metrics": candidate, "gains": gains, "all_metrics_nonnegative": bool(all(value >= -1e-12 for value in gains.values()))})
    mask = np.ones(len(labels), dtype=bool); baseline = measured(mask, champion); candidate = measured(mask, blended)
    gains = {key: candidate[key] - baseline[key] for key in baseline}
    all_folds = bool(all(item["all_metrics_nonnegative"] for item in fold_results)); all_global = bool(all(value >= -1e-12 for value in gains.values()))
    report = {"experiment": "fixed 5% EulerNet residual against frozen champion", "fixed_weight": FIXED_RESIDUAL_WEIGHT, "baseline_metrics": baseline, "fixed_residual_metrics": candidate, "fixed_residual_gains": gains, "folds": fold_results, "all_four_folds_all_metrics_nonnegative": all_folds, "all_global_metrics_nonnegative": all_global, "merits_promotion": bool(all_folds and all_global and gains["primary"] > 0), "resource_usage": usage(started_wall, started_cpu), "nonfinite_incidents": [], "integrity_incidents": [], "hidden_test_outcomes_parsed": False}
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True); AUDIT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"); return report


def run_screen(args) -> dict:
    overall_wall, overall_cpu = time.monotonic(), time.process_time()
    core, dev, screen = load_screen_rows(args.data_dir.resolve())
    selection_encoded, selection_dimension = encode(core, dev)
    core_x, core_y, _ = selection_encoded["train"]; dev_x, dev_y, dev_users = selection_encoded["valid"]
    # The first architecture is used only to instantiate inactive Euler modules in the matched control.
    control_model, control_history, selected_epoch, control_usage, control_parameters = train_model(mode="control", architecture=ARCHITECTURES[0], train_x=core_x, train_y=core_y, dimension=selection_dimension, epochs=args.epochs, valid_x=dev_x, valid_y=dev_y, valid_users=dev_users, patience=args.patience, **common(args))
    control_scores = predict_model(control_model, dev_x, "control", args.batch_size); control_metrics, _ = evaluate(dev_users, dev_y, control_scores)
    architecture_runs = []
    for index, architecture in enumerate(ARCHITECTURES):
        model, history, _, run_usage, parameters = train_model(mode="euler", architecture=architecture, train_x=core_x, train_y=core_y, dimension=selection_dimension, epochs=selected_epoch, patience=None, **common(args))
        scores = predict_model(model, dev_x, "euler", args.batch_size); measured, _ = evaluate(dev_users, dev_y, scores)
        architecture_runs.append({"architecture_index": index, "architecture": {"order_fields": list(architecture["order_fields"]), "layer_norm": architecture["layer_norm"]}, "selection_metrics": measured, "deltas_vs_control": {key: measured[key] - control_metrics[key] for key in ("primary", "GAUC", "nDCG@5")}, "active_parameters": parameters, "parameter_ratio_vs_control": parameters / control_parameters, "training_history": history, "resource_usage": run_usage})
    selected = max(architecture_runs, key=lambda item: (item["selection_metrics"]["primary"], -item["architecture_index"])); selected_architecture = ARCHITECTURES[selected["architecture_index"]]
    refit_encoded, refit_dimension = encode(core + dev, screen)
    full_x, full_y, _ = refit_encoded["train"]; screen_x, screen_y, screen_users = refit_encoded["valid"]
    final_runs = []; arrays = {}
    for mode in ("control", "euler"):
        model, history, _, run_usage, parameters = train_model(mode=mode, architecture=selected_architecture, train_x=full_x, train_y=full_y, dimension=refit_dimension, epochs=selected_epoch, patience=None, **common(args))
        scores = predict_model(model, screen_x, mode, args.batch_size); measured, ranks = evaluate(screen_users, screen_y, scores)
        arrays[f"{mode}_scores"] = scores; arrays[f"{mode}_ranks"] = ranks
        import torch
        checkpoint = args.output_dir / f"screen-{mode}.pt"; torch.save(model.state_dict(), checkpoint)
        final_runs.append({"mode": mode, "metrics": measured, "active_parameters": parameters, "training_history": history, "resource_usage": run_usage, "checkpoint": str(checkpoint.relative_to(ROOT))})
    score_path = args.output_dir / "screen-scores.npz"; np.savez_compressed(score_path, **arrays)
    control, candidate = final_runs; gains = {key: candidate["metrics"][key] - control["metrics"][key] for key in ("primary", "GAUC", "nDCG@5")}
    gate = bool(gains["primary"] >= 0.0001 and gains["GAUC"] > 0 and gains["nDCG@5"] > 0)
    report = {"experiment": "compact EulerNet complex-valued feature interactions", "paper": PAPER, "status": "screen_passed" if gate else "screen_rejected", "merits_confirmation": gate, "selected_architecture_index": selected["architecture_index"], "selected_architecture": {"order_fields": list(selected_architecture["order_fields"]), "layer_norm": selected_architecture["layer_norm"]}, "locked_epoch_count": selected_epoch, "predeclared_architecture_grid": [{"order_fields": list(value["order_fields"]), "layer_norm": value["layer_norm"]} for value in ARCHITECTURES], "control_selection_metrics": control_metrics, "architecture_selection_runs": architecture_runs, "control": control, "candidate": candidate, "deltas_vs_control": gains, "protocol": {"selection_fit": "Apr8-11", "architecture_selection": "Apr12-14", "refit": "Apr8-14", "screen": "Apr15-21", "apr22_plus_file_opened": False, "hidden_test_outcomes_parsed": False, "integrity_incidents": [], "nonfinite_incidents": []}, "matched_configuration": {**common(args), "fields": list(data_module.FIELDS), "objective": "pointwise BCE", "same_embedding_initialization_and_row_order": True, "bounded_log_modulus": [-6.0, 6.0], "control": "embedding plus MLP"}, "resources": {"control_epoch_selection": control_usage, "total": usage(overall_wall, overall_cpu)}, "artifacts": {"scores": str(score_path.relative_to(ROOT)), "script": str(Path(__file__).resolve().relative_to(ROOT))}, "recommendation": "run locked confirmation" if gate else "reject EulerNet; do not access Apr22+"}
    SCREEN_REPORT.parent.mkdir(parents=True, exist_ok=True); SCREEN_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"); print(json.dumps(report, indent=2, sort_keys=True)); return report


def run_confirmation(args) -> dict:
    if not 0 <= args.architecture_index < len(ARCHITECTURES) or args.locked_epochs <= 0:
        raise RuntimeError("Confirmation requires locked architecture and epoch count")
    overall_wall, overall_cpu = time.monotonic(), time.process_time(); architecture = ARCHITECTURES[args.architecture_index]
    train, valid = load_confirmation_rows(args.data_dir.resolve()); encoded, dimension = encode(train, valid)
    train_x, train_y, _ = encoded["train"]; valid_x, valid_y, valid_users = encoded["valid"]
    runs = []; arrays = {}
    for mode in ("control", "euler"):
        model, history, _, run_usage, parameters = train_model(mode=mode, architecture=architecture, train_x=train_x, train_y=train_y, dimension=dimension, epochs=args.locked_epochs, patience=None, **common(args))
        scores = predict_model(model, valid_x, mode, args.batch_size); measured, ranks = evaluate(valid_users, valid_y, scores)
        arrays[f"{mode}_scores"] = scores; arrays[f"{mode}_ranks"] = ranks
        import torch
        checkpoint = args.output_dir / f"confirmation-{mode}.pt"; torch.save(model.state_dict(), checkpoint)
        runs.append({"mode": mode, "metrics": measured, "active_parameters": parameters, "training_history": history, "resource_usage": run_usage, "checkpoint": str(checkpoint.relative_to(ROOT))})
    score_path = args.output_dir / "confirmation-scores.npz"; np.savez_compressed(score_path, **arrays)
    control, candidate = runs; gains = {key: candidate["metrics"][key] - control["metrics"][key] for key in ("primary", "GAUC", "nDCG@5")}; audit = residual_audit(valid_users, valid_y, arrays["euler_ranks"])
    report = {"experiment": "locked compact EulerNet confirmation", "paper": PAPER, "status": "confirmation_evaluated", "selected_architecture_index_locked": args.architecture_index, "selected_architecture": {"order_fields": list(architecture["order_fields"]), "layer_norm": architecture["layer_norm"]}, "locked_epoch_count": args.locked_epochs, "control": control, "candidate": candidate, "deltas_vs_control": gains, "matched_candidate_improves_all_metrics": bool(all(value > 0 for value in gains.values())), "protocol": {"train": "Apr8-21", "evaluation": "Apr22-28", "retuning": False, "outcome_fields_parsed_only_for": "Apr22-28", "hidden_test_outcomes_parsed": False, "integrity_incidents": [], "nonfinite_incidents": []}, "residual_audit": {"artifact": str(AUDIT_REPORT.relative_to(ROOT)), "fixed_weight": FIXED_RESIDUAL_WEIGHT, "fixed_residual_metrics": audit["fixed_residual_metrics"], "fixed_residual_gains": audit["fixed_residual_gains"], "all_four_folds_all_metrics_nonnegative": audit["all_four_folds_all_metrics_nonnegative"], "all_global_metrics_nonnegative": audit["all_global_metrics_nonnegative"], "merits_promotion": audit["merits_promotion"]}, "resources": {"total": usage(overall_wall, overall_cpu)}, "artifacts": {"scores": str(score_path.relative_to(ROOT)), "script": str(Path(__file__).resolve().relative_to(ROOT))}, "recommendation": "eligible for integrity review; do not auto-promote" if audit["merits_promotion"] else "reject; retain frozen champion"}
    CONFIRM_REPORT.parent.mkdir(parents=True, exist_ok=True); CONFIRM_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"); print(json.dumps(report, indent=2, sort_keys=True)); return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--confirmation", action="store_true"); parser.add_argument("--architecture-index", type=int, default=-1); parser.add_argument("--locked-epochs", type=int, default=0); parser.add_argument("--data-dir", type=Path, default=DATA); parser.add_argument("--output-dir", type=Path, default=RUNTIME); parser.add_argument("--seed", type=int, default=260830); parser.add_argument("--threads", type=int, default=6); parser.add_argument("--embedding-dim", type=int, default=12); parser.add_argument("--hidden", type=int, default=64); parser.add_argument("--dropout", type=float, default=0.05); parser.add_argument("--learning-rate", type=float, default=0.001); parser.add_argument("--weight-decay", type=float, default=1e-6); parser.add_argument("--batch-size", type=int, default=8192); parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--patience", type=int, default=4); args = parser.parse_args()
    import torch
    torch.set_num_threads(max(1, min(args.threads, 12))); args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_confirmation(args) if args.confirmation else run_screen(args); return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
