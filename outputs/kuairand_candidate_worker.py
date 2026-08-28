"""Trusted Kaggle worker for one safety-approved, model-authored KuaiRand program."""

from __future__ import annotations

import ast
import collections
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = {"lightgbm": "lightgbm>=4.1", "pyarrow": "pyarrow>=17", "sklearn": "scikit-learn>=1.4"}
missing = [distribution for module, distribution in REQUIRED.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd


CANDIDATE_SOURCE = "__CANDIDATE_SOURCE_REPR__"
TRUSTED_COMPONENTS_SOURCE = "__TRUSTED_COMPONENTS_SOURCE_REPR__"
PROPOSAL = json.loads("__PROPOSAL_REPR__")
WORK = Path("/kaggle/working/openai-candidate")
PUBLIC = WORK / "public"
EXPERIMENT = WORK / "experiment"
REPORT = Path("/kaggle/working/candidate_result.json")
EVENTS = Path("/kaggle/working/candidate_events.jsonl")
BASELINE = {"GAUC": 0.6674002647399903, "nDCG@5": 0.5357441067695617, "primary": 0.601572185754776}
PROXY_MIN_DELTA = -0.003
SCREEN_RUNGS = [
    {"id": "smoke", "train_fraction": 0.08, "validation_fraction": 0.20, "timeout": 90, "seed": 2026},
    {"id": "screen", "train_fraction": 0.35, "validation_fraction": 0.50, "timeout": 240, "seed": 2027},
]
POST_EXPOSURE = {
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter",
}
SAFE_CONTEXT = {
    "user_id", "video_id", "author_id", "date", "hourmin", "time_ms", "duration_ms", "tab",
    "video_type", "upload_dt", "upload_type", "visible_status", "video_duration", "server_width",
    "server_height", "music_id", "music_type", "tag",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(kind: str, message: str, **payload) -> None:
    record = {"timestamp": now(), "kind": kind, "message": message, **payload}
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def inspect_source(source: str) -> list[str]:
    allowed = {"collections", "gc", "json", "lightgbm", "math", "numpy", "pandas", "pathlib", "sklearn", "time", "trusted_components"}
    forbidden_names = {"os", "socket", "subprocess", "requests", "urllib", "http", "shutil", "ctypes", "multiprocessing"}
    forbidden_calls = {"eval", "exec", "compile", "__import__", "input", "open"}
    findings = []
    lowered = source.lower()
    for fragment in ("validation_labels", "hidden_test", "test_labels", "../", "https://", "http://", "/kaggle/input"):
        if fragment in lowered:
            findings.append(f"forbidden fragment: {fragment}")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    has_main = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    findings.append(f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in allowed:
                findings.append(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            findings.append(f"forbidden capability: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr == "parent":
            findings.append("parent traversal forbidden")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                findings.append(f"forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"absolute", "cwd", "glob", "iterdir", "resolve", "rglob"}:
                findings.append(f"filesystem discovery forbidden: {node.func.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(("/", "~")) or ".." in Path(node.value).parts:
                findings.append("absolute or parent path")
        elif isinstance(node, ast.FunctionDef) and node.name == "main":
            has_main = True
    if not has_main:
        findings.append("main() is required")
    return sorted(set(findings))


def locate_data() -> Path:
    for candidate in Path("/kaggle/input").rglob("log_standard_4_08_to_4_21_pure.csv"):
        directory = candidate.parent
        if (directory / "log_standard_4_22_to_5_08_pure.csv").exists() and (directory / "video_features_basic_pure.csv").exists():
            return directory
    raise FileNotFoundError("Attached Kaggle source does not contain KuaiRand-Pure")


def prepare() -> pd.DataFrame:
    data = locate_data()
    early = pd.read_csv(data / "log_standard_4_08_to_4_21_pure.csv", low_memory=False)
    late = pd.read_csv(data / "log_standard_4_22_to_5_08_pure.csv", low_memory=False)
    video = pd.read_csv(data / "video_features_basic_pure.csv", low_memory=False)
    early["date"] = pd.to_numeric(early["date"], errors="raise").astype(np.int32)
    late["date"] = pd.to_numeric(late["date"], errors="raise").astype(np.int32)
    train = early[early["date"].between(20220408, 20220421)].copy()
    private_validation = late[late["date"].between(20220422, 20220428)].copy()
    train["long_view"] = (pd.to_numeric(train["long_view"], errors="coerce").fillna(0) != 0).astype(np.int8)
    labels = (pd.to_numeric(private_validation["long_view"], errors="coerce").fillna(0) != 0).to_numpy(np.int8)
    metadata = [column for column in SAFE_CONTEXT if column in video.columns]
    if "video_id" not in metadata:
        metadata.append("video_id")
    train = train.merge(video[metadata], on="video_id", how="left", validate="many_to_one")
    private_validation = private_validation.merge(video[metadata], on="video_id", how="left", validate="many_to_one")
    public_train = train[[column for column in train.columns if column in SAFE_CONTEXT or column == "long_view"]].copy()
    public_validation = private_validation[[column for column in private_validation.columns if column in SAFE_CONTEXT]].copy()
    public_validation.insert(0, "row_id", np.arange(len(public_validation), dtype=np.int64))
    evaluator = pd.DataFrame({
        "row_id": public_validation["row_id"].to_numpy(),
        "user_id": private_validation["user_id"].to_numpy(),
        "label": labels,
    })
    PUBLIC.mkdir(parents=True, exist_ok=True)
    public_train.to_parquet(PUBLIC / "train.parquet", index=False)
    public_validation.to_parquet(PUBLIC / "validation.parquet", index=False)
    del early, late, video, train, private_validation, public_train, public_validation, labels
    gc.collect()
    return evaluator


def auc(labels: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    cursor = 0
    while cursor < len(pairs):
        end = cursor
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[cursor][0]:
            end += 1
        average = (cursor + end) / 2.0 + 1.0
        for index in range(cursor, end + 1):
            ranks[index] = average
        cursor = end + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return 0.5
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def evaluate(frame: pd.DataFrame, scores: np.ndarray, k: int = 5) -> dict[str, float]:
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Predictions are missing, misaligned, NaN, or infinite")
    grouped = collections.defaultdict(list)
    for user_id, label, score, row_id in zip(frame.user_id, frame.label, scores, frame.row_id):
        grouped[user_id].append((float(score), int(label), int(row_id)))
    gnum = gden = 0.0
    ndcgs = []
    for rows in grouped.values():
        labels = [row[1] for row in rows]
        positives = sum(labels)
        if 0 < positives < len(rows):
            gnum += positives * auc(labels, [row[0] for row in rows])
            gden += positives
        ranked = sorted(rows, key=lambda row: (-row[0], row[2]))
        top = [row[1] for row in ranked[:k]]
        dcg = sum(label / math.log2(index + 2) for index, label in enumerate(top))
        ideal = sorted(labels, reverse=True)[:k]
        idcg = sum(label / math.log2(index + 2) for index, label in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    gauc = gnum / gden if gden else 0.5
    ndcg = float(np.mean(ndcgs))
    return {"GAUC": gauc, "nDCG@5": ndcg, "primary": (gauc + ndcg) / 2.0}


def execute_program(directory: Path, timeout: int) -> tuple[subprocess.CompletedProcess | None, str | None, float]:
    (directory / "trusted_components.py").write_text(TRUSTED_COMPONENTS_SOURCE, encoding="utf-8")
    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "-P", "experiment.py"], cwd=directory, capture_output=True, text=True,
            timeout=timeout, check=False,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(directory), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return completed, None, time.time() - started
    except subprocess.TimeoutExpired:
        return None, f"{timeout}-second timeout", time.time() - started


def smoke_test() -> dict:
    smoke = WORK / "smoke"
    smoke_data = smoke / "data"
    smoke_data.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(PUBLIC / "train.parquet")
    validation = pd.read_parquet(PUBLIC / "validation.parquet")
    train_parts = []
    for _, group in train.groupby("date", sort=True):
        train_parts.append(group.sample(n=min(5000, len(group)), random_state=2026))
    train_small = pd.concat(train_parts, ignore_index=True)
    validation_small = validation.sample(n=min(15000, len(validation)), random_state=2026).sort_values("row_id").reset_index(drop=True)
    train_small.to_parquet(smoke_data / "train.parquet", index=False)
    validation_small.to_parquet(smoke_data / "validation.parquet", index=False)
    (smoke / "experiment.py").write_text(CANDIDATE_SOURCE, encoding="utf-8")
    completed, timeout_error, elapsed = execute_program(smoke, 120)
    if timeout_error:
        return {"passed": False, "error": timeout_error, "elapsed_seconds": elapsed}
    if completed is None or completed.returncode:
        return {
            "passed": False, "error": f"smoke exit {completed.returncode if completed else 'unknown'}",
            "stderr_tail": completed.stderr[-2000:] if completed else "", "elapsed_seconds": elapsed,
        }
    output = smoke / "predictions.npy"
    if not output.exists():
        return {"passed": False, "error": "smoke predictions.npy missing", "elapsed_seconds": elapsed}
    scores = np.load(output)
    if len(scores) != len(validation_small) or not np.isfinite(scores).all():
        return {"passed": False, "error": "smoke predictions are misaligned or non-finite", "elapsed_seconds": elapsed}
    return {"passed": True, "rows": len(validation_small), "elapsed_seconds": elapsed}


def trusted_module(directory: Path):
    module_path = directory / "trusted_components.py"
    module_path.write_text(TRUSTED_COMPONENTS_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("screen_trusted_components", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load trusted screening components")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def temporal_rung(frame: pd.DataFrame, rung: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.sort(frame["date"].unique())
    if len(dates) < 4:
        raise ValueError("Temporal screen requires at least four dates")
    boundary = dates[-2]
    fit_pool = frame[frame["date"] < boundary].copy()
    holdout_pool = frame[frame["date"] >= boundary].copy()
    users = np.sort(holdout_pool["user_id"].astype(str).unique())
    rng = np.random.default_rng(int(rung["seed"]))
    count = max(2, int(math.ceil(len(users) * float(rung["validation_fraction"]))))
    selected_users = set(rng.choice(users, size=min(count, len(users)), replace=False))
    holdout = holdout_pool[holdout_pool["user_id"].astype(str).isin(selected_users)].copy()
    fit = fit_pool[fit_pool["user_id"].astype(str).isin(selected_users)].copy()
    keep_ratio = min(1.0, float(rung["train_fraction"]) / float(rung["validation_fraction"]))
    if keep_ratio < 1.0:
        fit = fit.sample(frac=keep_ratio, random_state=int(rung["seed"]))
    fit = fit.sort_values(["date", "time_ms"], kind="stable").reset_index(drop=True)
    holdout = holdout.sort_values(["user_id", "time_ms"], kind="stable").reset_index(drop=True)
    labels = holdout["long_view"].to_numpy(np.int8)
    public_holdout = holdout.drop(columns=["long_view"]).copy()
    public_holdout.insert(0, "row_id", np.arange(len(public_holdout), dtype=np.int64))
    evaluator = pd.DataFrame({
        "row_id": public_holdout["row_id"].to_numpy(),
        "user_id": public_holdout["user_id"].to_numpy(),
        "label": labels,
    })
    return fit, public_holdout, evaluator


def official_fm_proxy(fit: pd.DataFrame, holdout: pd.DataFrame, directory: Path, seed: int) -> np.ndarray:
    trusted = trusted_module(directory)
    quantiles = np.unique(np.quantile(pd.to_numeric(fit["duration_ms"], errors="coerce").fillna(0), np.linspace(0, 1, 11)[1:-1]))

    def features(frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in ("user_id", "video_id", "author_id", "tab"):
            result[column] = frame[column].fillna("UNK").astype(str)
        result["dur_bucket"] = np.searchsorted(quantiles, pd.to_numeric(frame["duration_ms"], errors="coerce").fillna(0)).astype(str)
        return result

    fit_features, holdout_features = features(fit), features(holdout)
    columns = list(fit_features.columns)
    fit_matrix, holdout_matrix, dimension = trusted.encode_fm(fit_features, holdout_features, columns)
    labels = fit["long_view"].to_numpy(np.float32)
    model = trusted.TrustedFM(dimension=dimension, factors=16, learning_rate=0.001, l2=1e-6, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(12):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), 8192):
            batch = order[start:start + 8192]
            model.step(fit_matrix[batch], labels[batch])
    return model.predict(holdout_matrix).astype(np.float64)


def quality_screen(public_train: pd.DataFrame) -> dict:
    records = []
    for rung in SCREEN_RUNGS:
        directory = WORK / f"quality-{rung['id']}"
        data = directory / "data"
        data.mkdir(parents=True, exist_ok=True)
        fit, holdout, evaluator = temporal_rung(public_train, rung)
        fit.to_parquet(data / "train.parquet", index=False)
        holdout.to_parquet(data / "validation.parquet", index=False)
        (directory / "experiment.py").write_text(CANDIDATE_SOURCE, encoding="utf-8")
        completed, timeout_error, elapsed = execute_program(directory, int(rung["timeout"]))
        if timeout_error or completed is None or completed.returncode:
            record = {
                "rung": rung["id"], "passed": False, "reason": timeout_error or f"exit {completed.returncode if completed else 'unknown'}",
                "elapsed_seconds": elapsed, "train_rows": len(fit), "validation_rows": len(holdout),
                "stderr_tail": completed.stderr[-1200:] if completed else "",
            }
            records.append(record)
            return {"passed": False, "records": records, "rejected_at": rung["id"]}
        output = directory / "predictions.npy"
        if not output.exists():
            records.append({"rung": rung["id"], "passed": False, "reason": "predictions.npy missing"})
            return {"passed": False, "records": records, "rejected_at": rung["id"]}
        candidate_scores = np.load(output).astype(np.float64)
        candidate_metrics = evaluate(evaluator, candidate_scores)
        anchor_scores = official_fm_proxy(fit, holdout, directory, int(rung["seed"]))
        anchor_metrics = evaluate(evaluator, anchor_scores)
        delta = candidate_metrics["primary"] - anchor_metrics["primary"]
        passed = bool(delta >= PROXY_MIN_DELTA)
        record = {
            "rung": rung["id"], "passed": passed, "candidate_metrics": candidate_metrics,
            "anchor_metrics": anchor_metrics, "delta_vs_anchor": delta, "minimum_delta": PROXY_MIN_DELTA,
            "elapsed_seconds": elapsed, "train_rows": len(fit), "validation_rows": len(holdout),
        }
        records.append(record)
        event("screen", f"Training-only {rung['id']} tournament finished", screen=record)
        if not passed:
            return {"passed": False, "records": records, "rejected_at": rung["id"]}
    return {"passed": True, "records": records}


def run_candidate(evaluator: pd.DataFrame) -> dict:
    findings = inspect_source(CANDIDATE_SOURCE)
    if findings:
        return {"status": "failed", "stage": "safety", "failure_type": "implementation_failed", "counts_as_experiment": False, "error": "; ".join(findings)}
    smoke = smoke_test()
    event("smoke", "Candidate smoke test finished", smoke=smoke)
    if not smoke["passed"]:
        return {"status": "failed", "stage": "smoke", "failure_type": "implementation_failed", "counts_as_experiment": False, "metrics": {}, "error": smoke["error"], "smoke": smoke}
    public_train = pd.read_parquet(PUBLIC / "train.parquet")
    screening = quality_screen(public_train)
    if not screening["passed"]:
        last = screening["records"][-1]
        return {
            "status": "screen_rejected", "stage": "quality_screen", "failure_type": "model_quality",
            "counts_as_experiment": True, "external_validated": False,
            "metrics": last.get("candidate_metrics", {}), "proxy_anchor": last.get("anchor_metrics", {}),
            "delta_vs_anchor": last.get("delta_vs_anchor"), "error": last.get("reason", "Candidate fell below the temporal FM anchor"),
            "fidelity": {
                "id": last.get("rung", "screen"), "label": f"Temporal {last.get('rung', 'screen')}",
                "train_fraction": next((item["train_fraction"] for item in SCREEN_RUNGS if item["id"] == last.get("rung")), 0.0),
                "validation_fraction": next((item["validation_fraction"] for item in SCREEN_RUNGS if item["id"] == last.get("rung")), 0.0),
                "seeds": 1, "max_seconds": next((item["timeout"] for item in SCREEN_RUNGS if item["id"] == last.get("rung")), 0),
                "can_promote_champion": False,
            },
            "smoke": smoke, "screening": screening,
        }
    del public_train
    gc.collect()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    data = EXPERIMENT / "data"
    data.mkdir(exist_ok=True)
    for name in ("train.parquet", "validation.parquet"):
        os.link(PUBLIC / name, data / name)
    program = EXPERIMENT / "experiment.py"
    program.write_text(CANDIDATE_SOURCE, encoding="utf-8")
    completed, timeout_error, elapsed = execute_program(EXPERIMENT, 600)
    if timeout_error:
        return {"status": "failed", "stage": "execution", "failure_type": "implementation_failed", "counts_as_experiment": False, "error": timeout_error, "elapsed_seconds": elapsed, "smoke": smoke}
    Path("/kaggle/working/candidate_stdout.log").write_text(completed.stdout, encoding="utf-8")
    Path("/kaggle/working/candidate_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed is None or completed.returncode:
        return {
            "status": "failed", "stage": "execution", "failure_type": "implementation_failed", "counts_as_experiment": False, "error": f"exit {completed.returncode if completed else 'unknown'}",
            "stderr_tail": completed.stderr[-2000:] if completed else "", "elapsed_seconds": elapsed, "smoke": smoke,
        }
    predictions = EXPERIMENT / "predictions.npy"
    if not predictions.exists():
        return {"status": "failed", "stage": "output", "failure_type": "implementation_failed", "counts_as_experiment": False, "error": "predictions.npy missing", "smoke": smoke}
    metrics = evaluate(evaluator, np.load(predictions).astype(np.float64))
    return {
        "status": "completed", "stage": "evaluation", "metrics": metrics,
        "delta_vs_official_fm": metrics["primary"] - BASELINE["primary"],
        "improved": metrics["primary"] > BASELINE["primary"],
        "counts_as_experiment": True, "external_validated": True,
        "fidelity": {"id": "confirm", "label": "Full confirmation", "train_fraction": 1.0, "validation_fraction": 1.0, "seeds": 1, "max_seconds": 600, "can_promote_champion": True},
        "smoke": smoke, "screening": screening, "elapsed_seconds": elapsed,
    }


def main() -> None:
    event("start", "Trusted worker started", proposal=PROPOSAL, source_sha256=hashlib.sha256(CANDIDATE_SOURCE.encode()).hexdigest())
    evaluator = prepare()
    event("data", "Sanitized KuaiRand benchmark prepared", train_rows=1_141_112, validation_rows=len(evaluator))
    result = run_candidate(evaluator)
    report = {
        "timestamp": now(), "model": "gpt-5.6-sol", "proposal": PROPOSAL,
        "official_fm": BASELINE, "source_sha256": hashlib.sha256(CANDIDATE_SOURCE.encode()).hexdigest(),
        "manual_interventions": 0, "result": result,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    event("result", "Candidate evaluation finished", result=result)


if __name__ == "__main__":
    main()
