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
PROPOSAL = json.loads("__PROPOSAL_REPR__")
WORK = Path("/kaggle/working/openai-candidate")
PUBLIC = WORK / "public"
EXPERIMENT = WORK / "experiment"
REPORT = Path("/kaggle/working/candidate_result.json")
EVENTS = Path("/kaggle/working/candidate_events.jsonl")
BASELINE = {"GAUC": 0.6674002647399903, "nDCG@5": 0.5357441067695617, "primary": 0.601572185754776}
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
    allowed = {"collections", "gc", "json", "lightgbm", "math", "numpy", "pandas", "pathlib", "sklearn"}
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


def run_candidate(evaluator: pd.DataFrame) -> dict:
    findings = inspect_source(CANDIDATE_SOURCE)
    if findings:
        return {"status": "failed", "stage": "safety", "error": "; ".join(findings)}
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    data = EXPERIMENT / "data"
    data.mkdir(exist_ok=True)
    for name in ("train.parquet", "validation.parquet"):
        os.link(PUBLIC / name, data / name)
    program = EXPERIMENT / "experiment.py"
    program.write_text(CANDIDATE_SOURCE, encoding="utf-8")
    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "experiment.py"],
            cwd=EXPERIMENT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "stage": "execution", "error": "ten-minute timeout", "elapsed_seconds": time.time() - started}
    Path("/kaggle/working/candidate_stdout.log").write_text(completed.stdout, encoding="utf-8")
    Path("/kaggle/working/candidate_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        return {
            "status": "failed", "stage": "execution", "error": f"exit {completed.returncode}",
            "stderr_tail": completed.stderr[-2000:], "elapsed_seconds": time.time() - started,
        }
    predictions = EXPERIMENT / "predictions.npy"
    if not predictions.exists():
        return {"status": "failed", "stage": "output", "error": "predictions.npy missing"}
    metrics = evaluate(evaluator, np.load(predictions).astype(np.float64))
    return {
        "status": "completed", "stage": "evaluation", "metrics": metrics,
        "delta_vs_official_fm": metrics["primary"] - BASELINE["primary"],
        "improved": metrics["primary"] > BASELINE["primary"],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    event("start", "Trusted worker started", proposal=PROPOSAL, source_sha256=hashlib.sha256(CANDIDATE_SOURCE.encode()).hexdigest())
    evaluator = prepare()
    event("data", "Sanitized KuaiRand benchmark prepared", train_rows=1_141_112, validation_rows=len(evaluator))
    result = run_candidate(evaluator)
    report = {
        "timestamp": now(), "model": "gpt-5.6-luna", "proposal": PROPOSAL,
        "official_fm": BASELINE, "source_sha256": hashlib.sha256(CANDIDATE_SOURCE.encode()).hexdigest(),
        "manual_interventions": 0, "result": result,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    event("result", "Candidate evaluation finished", result=result)


if __name__ == "__main__":
    main()
