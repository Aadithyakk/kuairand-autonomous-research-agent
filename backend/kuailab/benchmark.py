from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .provider import Proposal


@dataclass
class Evaluation:
    primary: float
    gauc: float
    ndcg5: float
    runtime_seconds: float
    evidence: str

    def metrics(self) -> dict[str, float]:
        return {"primary": self.primary, "gauc": self.gauc, "ndcg5": self.ndcg5}


def validate_metrics(data: dict) -> None:
    required = ("primary", "gauc", "ndcg5")
    for key in required:
        value = data.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {key} metric: {value!r}")


class SyntheticBenchmark:
    """Deterministic smoke benchmark. Its values are clearly marked as demo evidence."""

    def baseline(self, workspace: Path) -> Evaluation:
        return Evaluation(primary=0.6016, gauc=0.6612, ndcg5=0.5310, runtime_seconds=0, evidence="synthetic-demo")

    def evaluate(self, proposal: Proposal, iteration: int, workspace: Path) -> Evaluation:
        if iteration == 4:
            raise RuntimeError("Synthetic worker simulated an out-of-memory failure")
        primary = min(0.625, 0.6016 + sum([0.0025, 0.0051, 0.0069, 0.0069, 0.0075, 0.0077][:iteration]))
        gauc = min(0.70, 0.6612 + (primary - 0.6016) * 1.9)
        ndcg = min(0.59, 0.5310 + (primary - 0.6016) * 1.55)
        metrics = {"primary": round(primary, 4), "gauc": round(gauc, 4), "ndcg5": round(ndcg, 4)}
        validate_metrics(metrics)
        return Evaluation(**metrics, runtime_seconds=round(2.2 + iteration * 0.6, 2), evidence="synthetic-demo")


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
        result = subprocess.run(self.command, cwd=workspace, env=environment, capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
        (workspace / "runner.stdout.log").write_text(result.stdout, encoding="utf-8")
        (workspace / "runner.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            tail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr detail"
            raise RuntimeError(f"Organizer adapter exited with code {result.returncode}: {tail[:500]}")
        if not metrics_path.exists():
            raise RuntimeError("Organizer adapter did not write metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validate_metrics(metrics)
        return Evaluation(
            primary=float(metrics["primary"]), gauc=float(metrics["gauc"]), ndcg5=float(metrics["ndcg5"]),
            runtime_seconds=float(metrics.get("runtime_seconds", 0)), evidence="kuairand-pure-validation",
        )

    def baseline(self, workspace: Path) -> Evaluation:
        return self._invoke("baseline", 0, workspace, None)

    def evaluate(self, proposal: Proposal, iteration: int, workspace: Path) -> Evaluation:
        return self._invoke("experiment", iteration, workspace, str(workspace / "proposal.json"))
