from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .provider import Proposal
from .resources import child_usage_delta, child_usage_snapshot, normalize_resource_usage


@dataclass
class Evaluation:
    primary: float
    gauc: float
    ndcg5: float
    runtime_seconds: float
    evidence: str
    resource_usage: dict

    def metrics(self) -> dict[str, float]:
        return {"primary": self.primary, "gauc": self.gauc, "ndcg5": self.ndcg5}


class BenchmarkRunError(RuntimeError):
    def __init__(self, message: str, resource_usage: dict):
        super().__init__(message)
        self.resource_usage = resource_usage


def validate_metrics(data: dict) -> None:
    required = ("primary", "gauc", "ndcg5")
    for key in required:
        value = data.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {key} metric: {value!r}")


class SyntheticBenchmark:
    """Deterministic smoke benchmark. Its values are clearly marked as demo evidence."""

    def baseline(self, workspace: Path) -> Evaluation:
        usage = normalize_resource_usage({"wall_seconds": 0, "train_seconds": 0, "device": "cpu"})
        return Evaluation(primary=0.6016, gauc=0.6612, ndcg5=0.5310, runtime_seconds=0, evidence="synthetic-demo", resource_usage=usage)

    def evaluate(self, proposal: Proposal, iteration: int, workspace: Path) -> Evaluation:
        if iteration == 4:
            usage = normalize_resource_usage({
                "wall_seconds": 1.7,
                "train_seconds": 1.7,
                "cpu_seconds": 1.35,
                "peak_rss_mb": 512,
                "device": "cpu",
            })
            (workspace / "resource-usage.json").write_text(json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8")
            raise BenchmarkRunError("Synthetic worker simulated an out-of-memory failure", usage)
        primary = min(0.625, 0.6016 + sum([0.0025, 0.0051, 0.0069, 0.0069, 0.0075, 0.0077][:iteration]))
        gauc = min(0.70, 0.6612 + (primary - 0.6016) * 1.9)
        ndcg = min(0.59, 0.5310 + (primary - 0.6016) * 1.55)
        metrics = {"primary": round(primary, 4), "gauc": round(gauc, 4), "ndcg5": round(ndcg, 4)}
        validate_metrics(metrics)
        runtime = round(2.2 + iteration * 0.6, 2)
        usage = normalize_resource_usage({
            "wall_seconds": runtime,
            "train_seconds": runtime,
            "cpu_seconds": round(runtime * 0.82, 3),
            "peak_rss_mb": 96 + iteration * 4,
            "device": "cpu",
        })
        return Evaluation(**metrics, runtime_seconds=runtime, evidence="synthetic-demo", resource_usage=usage)


class CommandBenchmark:
    """Runs a user-supplied organizer adapter; generated code is never executed directly."""

    def __init__(self, command: str, dataset_path: str, timeout_seconds: int):
        if not command:
            raise RuntimeError("KUAI_EXPERIMENT_COMMAND is required for KuaiRand mode")
        if not dataset_path or not Path(dataset_path).exists():
            raise RuntimeError("KUAIRAND_DATA_PATH must point to the local KuaiRand-Pure dataset")
        self.command = shlex.split(command)
        self.dataset_path = str(Path(dataset_path).resolve())
        self.timeout_seconds = timeout_seconds

    def _invoke(self, action: str, iteration: int, workspace: Path, proposal_path: str | None) -> Evaluation:
        request_path = workspace / "runner-request.json"
        metrics_path = workspace / "metrics.json"
        request_path.write_text(json.dumps({
            "action": action,
            "iteration": iteration,
            "dataset_path": self.dataset_path,
            "proposal_path": proposal_path,
            "metrics_path": str(metrics_path),
            "target": "long_view",
        }, indent=2), encoding="utf-8")
        environment = os.environ.copy()
        environment["KUAI_RUNNER_REQUEST"] = str(request_path)
        child_before = child_usage_snapshot()
        wall_started = time.monotonic()
        try:
            result = subprocess.run(self.command, cwd=workspace, env=environment, capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
        except subprocess.TimeoutExpired as error:
            elapsed = time.monotonic() - wall_started
            usage = child_usage_delta(child_before, wall_seconds=elapsed)
            (workspace / "resource-usage.json").write_text(json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8")
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            (workspace / "runner.stdout.log").write_text(stdout, encoding="utf-8")
            (workspace / "runner.stderr.log").write_text(stderr, encoding="utf-8")
            raise BenchmarkRunError(f"Organizer adapter timed out after {self.timeout_seconds} seconds", usage) from error
        elapsed = time.monotonic() - wall_started
        fallback_usage = child_usage_delta(child_before, wall_seconds=elapsed)
        (workspace / "runner.stdout.log").write_text(result.stdout, encoding="utf-8")
        (workspace / "runner.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            tail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr detail"
            (workspace / "resource-usage.json").write_text(json.dumps(fallback_usage, indent=2, sort_keys=True), encoding="utf-8")
            raise BenchmarkRunError(f"Organizer adapter exited with code {result.returncode}: {tail[:500]}", fallback_usage)
        if not metrics_path.exists():
            (workspace / "resource-usage.json").write_text(json.dumps(fallback_usage, indent=2, sort_keys=True), encoding="utf-8")
            raise BenchmarkRunError("Organizer adapter did not write metrics.json", fallback_usage)
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            validate_metrics(metrics)
        except (json.JSONDecodeError, ValueError) as error:
            (workspace / "resource-usage.json").write_text(json.dumps(fallback_usage, indent=2, sort_keys=True), encoding="utf-8")
            raise BenchmarkRunError(f"Organizer adapter returned invalid metrics: {error}", fallback_usage) from error
        raw_runtime = metrics.get("runtime_seconds", elapsed)
        runtime_seconds = float(raw_runtime) if isinstance(raw_runtime, (int, float)) and math.isfinite(raw_runtime) and raw_runtime >= 0 else elapsed
        usage = normalize_resource_usage(
            metrics.get("resource_usage") if isinstance(metrics.get("resource_usage"), dict) else None,
            wall_seconds=runtime_seconds,
            cpu_seconds=fallback_usage["cpu_seconds"],
            peak_rss_mb=fallback_usage["peak_rss_mb"],
        )
        (workspace / "resource-usage.json").write_text(json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8")
        return Evaluation(
            primary=float(metrics["primary"]), gauc=float(metrics["gauc"]), ndcg5=float(metrics["ndcg5"]),
            runtime_seconds=runtime_seconds, evidence="kuairand-pure-validation", resource_usage=usage,
        )

    def baseline(self, workspace: Path) -> Evaluation:
        return self._invoke("baseline", 0, workspace, None)

    def evaluate(self, proposal: Proposal, iteration: int, workspace: Path) -> Evaluation:
        return self._invoke("experiment", iteration, workspace, str(workspace / "proposal.json"))
