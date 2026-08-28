"""Private Kaggle pilot: an LLM proposes and implements real KuaiRand experiments.

This is intentionally self-contained so the Kaggle execution record is an
auditable artifact. Validation labels remain in the controller process and are
never written into an experiment workspace.
"""

from __future__ import annotations

import ast
import collections
import copy
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = {
    "accelerate": "accelerate>=1.1",
    "lightgbm": "lightgbm>=4.1",
    "json_repair": "json-repair>=0.50",
    "pyarrow": "pyarrow>=17",
    "transformers": "transformers>=4.46,<5",
}
missing = [distribution for module, distribution in REQUIRED.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import torch
from json_repair import loads as repair_json_loads
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_EXPERIMENTS = 2
WORK = Path("/kaggle/working/real-agent")
PUBLIC = WORK / "public"
EXPERIMENTS = WORK / "experiments"
REPORT = Path("/kaggle/working/real_agent_report.json")
EVENTS = Path("/kaggle/working/real_agent_events.jsonl")
BASELINE = {"GAUC": 0.6674002647399903, "nDCG@5": 0.5357441067695617, "primary": 0.6015721678733825}
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


def locate_data() -> Path:
    root = Path("/kaggle/input")
    for candidate in root.rglob("log_standard_4_08_to_4_21_pure.csv"):
        directory = candidate.parent
        if (directory / "log_standard_4_22_to_5_08_pure.csv").exists() and (directory / "video_features_basic_pure.csv").exists():
            return directory
    raise FileNotFoundError("The attached Kaggle source does not contain KuaiRand-Pure")


def prepare_public_benchmark() -> tuple[pd.DataFrame, np.ndarray]:
    data = locate_data()
    early = pd.read_csv(data / "log_standard_4_08_to_4_21_pure.csv", low_memory=False)
    late = pd.read_csv(data / "log_standard_4_22_to_5_08_pure.csv", low_memory=False)
    video = pd.read_csv(data / "video_features_basic_pure.csv", low_memory=False)
    early["date"] = pd.to_numeric(early["date"], errors="raise").astype(np.int32)
    late["date"] = pd.to_numeric(late["date"], errors="raise").astype(np.int32)
    train = early[early["date"].between(20220408, 20220421)].copy()
    validation_private = late[late["date"].between(20220422, 20220428)].copy()
    train["long_view"] = (pd.to_numeric(train["long_view"], errors="coerce").fillna(0) != 0).astype(np.int8)
    validation_labels = (pd.to_numeric(validation_private["long_view"], errors="coerce").fillna(0) != 0).to_numpy(np.int8)

    metadata = [column for column in SAFE_CONTEXT if column in video.columns]
    if "video_id" not in metadata:
        metadata.append("video_id")
    train = train.merge(video[metadata], on="video_id", how="left", validate="many_to_one")
    validation_private = validation_private.merge(video[metadata], on="video_id", how="left", validate="many_to_one")

    public_train_columns = [column for column in train.columns if column in SAFE_CONTEXT or column == "long_view"]
    public_validation_columns = [column for column in validation_private.columns if column in SAFE_CONTEXT]
    public_train = train[public_train_columns].copy()
    public_validation = validation_private[public_validation_columns].copy()
    public_validation.insert(0, "row_id", np.arange(len(public_validation), dtype=np.int64))
    evaluator_frame = pd.DataFrame({
        "row_id": public_validation["row_id"].to_numpy(),
        "user_id": validation_private["user_id"].to_numpy(),
        "label": validation_labels,
    })

    PUBLIC.mkdir(parents=True, exist_ok=True)
    public_train.to_parquet(PUBLIC / "train.parquet", index=False)
    public_validation.to_parquet(PUBLIC / "validation.parquet", index=False)
    del early, late, video, train, validation_private, public_train, public_validation
    return evaluator_frame, validation_labels


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
    if positives == 0 or negatives == 0:
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


class LocalResearchModel:
    def __init__(self) -> None:
        event("model", "Loading local research model", model=MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
        self.last_raw = ""

    def ask(self, system: str, payload: dict, max_new_tokens: int) -> dict:
        generated = self._generate(system, payload, max_new_tokens)
        match = re.search(r"\{.*\}", generated, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Research model did not return a JSON object: {generated[-500:]}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            repaired = repair_json_loads(match.group(0))
            if not isinstance(repaired, dict):
                raise ValueError("Structured-output repair did not produce an object")
            return repaired

    def ask_code(self, system: str, payload: dict, max_new_tokens: int) -> str:
        generated = self._generate(system, payload, max_new_tokens)
        fenced = re.search(r"```(?:python)?\s*(.*?)```", generated, flags=re.DOTALL | re.IGNORECASE)
        source = fenced.group(1).strip() if fenced else generated.strip()
        if source.lower().startswith("python\n"):
            source = source.split("\n", 1)[1]
        if not source:
            raise ValueError("Research coder returned an empty program")
        return source

    def _generate(self, system: str, payload: dict, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.04,
            )
        generated = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        self.last_raw = generated
        return generated


class SafetyGate:
    allowed_imports = {"collections", "gc", "json", "lightgbm", "math", "numpy", "pandas", "pathlib", "sklearn"}
    forbidden_names = {"os", "socket", "subprocess", "requests", "urllib", "http", "shutil", "ctypes", "multiprocessing"}
    forbidden_calls = {"eval", "exec", "compile", "__import__", "input", "open"}

    def inspect(self, source: str) -> list[str]:
        findings = []
        if len(source.encode()) > 40_000:
            findings.append("program exceeds 40 KB")
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
                    if alias.name.split(".")[0] not in self.allowed_imports:
                        findings.append(f"import not allowed: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in self.allowed_imports:
                    findings.append(f"import not allowed: {node.module}")
            elif isinstance(node, ast.Name) and node.id in self.forbidden_names:
                findings.append(f"forbidden capability: {node.id}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.forbidden_calls:
                findings.append(f"forbidden call: {node.func.id}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith(("/", "~")) or ".." in Path(node.value).parts:
                    findings.append("absolute or parent path")
            elif isinstance(node, ast.FunctionDef) and node.name == "main":
                has_main = True
        if not has_main:
            findings.append("main() is required")
        return sorted(set(findings))


def proposal_prompt() -> str:
    return """You are the planner of an autonomous recommender-systems research agent. Return strict JSON with diagnostic, research_query, and candidates. Create 2-3 distinct new experiments; do not choose from a supplied model list. Each candidate requires id, title, hypothesis, change_kind, research_basis, expected_gain, risk, estimated_minutes, acceptance_rule, and abort_condition. expected_gain is a realistic absolute change in the 0..0.03 primary metric range; risk is 0..1; estimated_minutes must be 0.1..10. Favor hypotheses that can run on 1.14M rows within ten minutes. Never use validation outcomes as features. A prior LambdaRank-history experiment underperformed the official FM, so diagnose rather than repeating it."""


def code_prompt() -> str:
    return """You are the coding member of an autonomous recommender research agent. Return only one complete Python program inside a python markdown code fence; do not wrap the program in JSON. The program must define and call main(), read data/train.parquet and data/validation.parquet, learn only from the training long_view label, and write predictions.npy in exact validation row order. Validation has row_id but no outcomes. Use only pandas, numpy, lightgbm, sklearn, math, collections, gc, json, or pathlib. Do not use network, subprocess, os, open(), absolute/parent paths, validation labels, test data, or evaluator internals. Use an internal temporal training holdout if tuning is needed. Keep runtime below ten minutes and memory below 24 GB."""


def research_context(memory: list[dict]) -> dict:
    return {
        "benchmark": "KuaiRand-Pure within-user ranking",
        "train": {"dates": "20220408-20220421", "rows": 1_141_112, "label": "long_view"},
        "validation": {"dates": "20220422-20220428", "rows": 124_909, "labels": "external evaluator only"},
        "metrics": ["GAUC", "nDCG@5", "primary = mean"],
        "official_fm": BASELINE,
        "observed_prior_experiment": {
            "title": "causal history + LambdaRank",
            "metrics": {"GAUC": 0.6636374904605207, "nDCG@5": 0.5331166190352395, "primary": 0.5983770547478802},
            "lesson": "Ranking-loss alignment alone did not compensate for weaker generalization than the five-field FM.",
        },
        "calibration_ensemble": {
            "components": ["official FM", "pointwise history LightGBM", "history LambdaRank", "BPR"],
            "metrics": {"GAUC": 0.6693836797580355, "nDCG@5": 0.5365475685771528, "primary": 0.6029656241675941},
            "caution": "Weights were selected on validation and the gain is below the 0.002 convergence threshold; treat as directional evidence only.",
        },
        "columns": sorted(SAFE_CONTEXT | {"long_view"}),
        "literature_lessons": [
            "Wide-and-deep or residual combinations can preserve memorization while adding transferable history features.",
            "Multi-behavior signals may help representations, but same-impression outcomes cannot be inference features.",
            "Pairwise objectives can improve top-k ranking but may trade off calibrated discrimination.",
            "Rank averaging can combine structurally diverse models without relying on score calibration.",
        ],
        "memory": memory[-4:],
    }


def validate_candidates(value) -> list[dict]:
    required = {
        "id", "title", "hypothesis", "change_kind", "research_basis", "expected_gain", "risk",
        "estimated_minutes", "acceptance_rule", "abort_condition",
    }
    if not isinstance(value, list) or not value:
        raise ValueError("planner returned no candidates")
    result = []
    for raw in value[:4]:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError(f"candidate contract violation: {raw}")
        item = copy.deepcopy(raw)
        item["expected_gain"] = min(0.03, max(0.0, float(item["expected_gain"])))
        item["risk"] = min(1.0, max(0.0, float(item["risk"])))
        item["estimated_minutes"] = min(10.0, max(0.1, float(item["estimated_minutes"])))
        item["acquisition"] = item["expected_gain"] - 0.35 * item["risk"] - 0.002 * item["estimated_minutes"]
        result.append(item)
    return sorted(result, key=lambda item: (-item["acquisition"], item["estimated_minutes"]))


def run_experiment(source: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "data"
    data.mkdir(exist_ok=True)
    for name in ("train.parquet", "validation.parquet"):
        target = data / name
        if not target.exists():
            os.link(PUBLIC / name, target)
    program = directory / "experiment.py"
    program.write_text(source, encoding="utf-8")
    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "experiment.py"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "stage": "execution", "error": "ten-minute timeout", "elapsed_seconds": time.time() - started}
    (directory / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        return {
            "status": "failed", "stage": "execution", "error": f"exit {completed.returncode}",
            "stderr_tail": completed.stderr[-2000:], "elapsed_seconds": time.time() - started,
        }
    output = directory / "predictions.npy"
    if not output.exists():
        return {"status": "failed", "stage": "output", "error": "predictions.npy missing", "elapsed_seconds": time.time() - started}
    return {"status": "completed", "stage": "evaluation", "scores": np.load(output), "elapsed_seconds": time.time() - started}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    event("baseline", "Official five-seed FM reference loaded", metrics=BASELINE)
    evaluator_frame, _ = prepare_public_benchmark()
    event("data", "Sanitized benchmark created", train_rows=1_141_112, validation_rows=len(evaluator_frame))
    model = LocalResearchModel()
    gate = SafetyGate()
    memory = []
    experiments = []
    champion = {"experiment_id": "iteration-000", "title": "Official FM", "metrics": BASELINE}

    for iteration in range(1, MAX_EXPERIMENTS + 1):
        experiment_id = f"iteration-{iteration:03d}"
        context = research_context(memory)
        try:
            bundle = model.ask(proposal_prompt(), context, max_new_tokens=1300)
            candidates = validate_candidates(bundle.get("candidates"))
            selected = candidates[0]
            event("decision", "Agent selected a generated hypothesis", experiment_id=experiment_id, selected=selected, diagnostic=bundle.get("diagnostic"))
            source = model.ask_code(code_prompt(), {"proposal": selected, "benchmark": context, "prior_failures": memory[-3:]}, max_new_tokens=2800)
            findings = gate.inspect(source)
            if findings:
                result = {"status": "failed", "stage": "safety", "error": "; ".join(findings)}
            else:
                result = run_experiment(source, EXPERIMENTS / experiment_id)
                if result["status"] == "completed":
                    scores = np.asarray(result.pop("scores"), dtype=np.float64)
                    result["metrics"] = evaluate(evaluator_frame, scores)
                    result["improved"] = result["metrics"]["primary"] > champion["metrics"]["primary"]
                    result["delta_vs_champion"] = result["metrics"]["primary"] - champion["metrics"]["primary"]
                    if result["improved"]:
                        champion = {"experiment_id": experiment_id, "title": selected["title"], "metrics": result["metrics"]}
            record = {"experiment_id": experiment_id, "proposal": selected, **result}
        except Exception as exc:
            record = {
                "experiment_id": experiment_id,
                "status": "failed",
                "stage": "controller",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
                "model_output_tail": model.last_raw[-2000:],
            }
        experiments.append(record)
        lesson = {
            "experiment_id": experiment_id,
            "status": record["status"],
            "stage": record.get("stage"),
            "hypothesis": record.get("proposal", {}).get("hypothesis"),
            "metrics": record.get("metrics", {}),
            "error": record.get("error"),
            "lesson": (
                "Retain the measured gain and test whether it is robust."
                if record.get("improved")
                else "The attempt did not beat the champion; change the causal mechanism or repair the failed implementation."
            ),
        }
        memory.append(lesson)
        event("result", "Autonomous experiment finished", record=record)

    report = {
        "status": "completed",
        "timestamp": now(),
        "benchmark": "KuaiRand-Pure 20220408-20220428",
        "research_model": MODEL_ID,
        "baseline": BASELINE,
        "champion": champion,
        "experiments": experiments,
        "memory": memory,
        "manual_interventions": 0,
        "validation_labels_written_to_experiment_workspace": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    event("complete", "Real autonomous KuaiRand pilot completed", champion=champion)


if __name__ == "__main__":
    main()
