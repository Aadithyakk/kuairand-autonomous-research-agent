from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .core import read_json, utc_now, write_json_atomic
from .kaggle_packager import package
from .real_pilot import generate, reflect


StageCallback = Callable[[str, str, dict[str, Any] | None], None]
StopCallback = Callable[[], bool]


def requires_gpu(model_family: str, compute_preference: str) -> bool:
    preference = compute_preference.strip().lower()
    family_tokens = set(model_family.lower().replace("-", "_").split("_"))
    return preference in {"gpu", "t4", "cuda"} or bool(
        family_tokens.intersection({"sequence", "sequential", "neural", "transformer", "din", "dien", "sasrec"})
    )


class AutonomousKaggleCampaign:
    """Run the real propose/review/Kaggle/evaluate/reflect loop unattended.

    The process performs its own coarse Kaggle status checks. Those checks are
    ordinary API traffic and never require a Codex turn.
    """

    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.config = config
        campaign = config.get("campaign", {})
        self.generation_root = root / campaign.get("generation_root", "runtime/openai-campaign")
        self.kernel_root = root / campaign.get("kernel_root", "runtime/kaggle-campaign")
        self.memory_path = root / campaign.get("memory", "runtime/openai-pilot/memory.json")
        self.poll_seconds = max(20, int(campaign.get("kaggle_poll_seconds", 60)))
        self.dataset_source = campaign.get("dataset_source", "arashnic/an-unbiased-sequential-recommendation-dataset")
        self.kaggle_username = campaign.get("kaggle_username") or os.getenv("KAGGLE_USERNAME", "")

    def run_iteration(
        self,
        sequence: int,
        compute_profile_id: str,
        stage: StageCallback,
        should_stop: StopCallback,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not present in the controller process")
        if not os.getenv("KAGGLE_API_TOKEN"):
            raise RuntimeError("KAGGLE_API_TOKEN is not present in the controller process")
        if not self.kaggle_username:
            raise RuntimeError("Set campaign.kaggle_username or KAGGLE_USERNAME")

        run_name = f"iteration-{sequence:03d}"
        generation_dir = self.generation_root / run_name
        kernel_dir = self.kernel_root / run_name
        generation_dir.mkdir(parents=True, exist_ok=True)
        kernel_dir.mkdir(parents=True, exist_ok=True)

        stage("planning", "Planning and reviewing a bounded experiment.", None)
        report = generate(
            generation_dir,
            self.root / "configs/default.json",
            self.root / "knowledge/literature.json",
            self.memory_path,
            progress=lambda phase, message: stage(phase, message, None),
            should_stop=should_stop,
        )
        proposal = read_json(generation_dir / "proposal.json")
        proposal.update({"experiment_id": run_name, "status": "reviewed", "started_at": utc_now()})
        if not report.get("approved_for_execution"):
            stage("backtracking", "No proposed candidate passed deterministic and methodological preflight.", {"proposal": proposal})
            result = {
                "status": "failed",
                "stage": "preflight",
                "failure_type": "implementation_failed",
                "counts_as_experiment": False,
                "metrics": {},
                "error": "; ".join(report.get("safety", {}).get("findings", [])) or "All candidate implementations failed preflight.",
                "implementation_attempts": report.get("implementation_attempts", []),
            }
            return proposal, result
        proposal["implementation_attempts"] = report.get("implementation_attempts", [])
        stage("reviewed", "The proposal passed deterministic, methodology, leakage and code review.", {"proposal": proposal})

        worker_path = kernel_dir / "kuairand_candidate_worker.py"
        package(
            self.root / "outputs/kuairand_candidate_worker.py",
            generation_dir / "candidate.py",
            generation_dir / "proposal.json",
            worker_path,
        )
        # Kaggle derives the public slug from the title, even when metadata.id
        # requests another value. Keep both exactly aligned so status polling
        # and output retrieval address the kernel Kaggle actually creates.
        slug = f"kuairand-autonomous-agent-{sequence:03d}"
        kernel_ref = f"{self.kaggle_username}/{slug}"
        family = str(proposal.get("model_family", "")).lower()
        preference = str(proposal.get("compute_preference", "auto")).lower()
        gpu = requires_gpu(family, preference)
        selected_compute = "kaggle-t4" if gpu else "kaggle-cpu"
        proposal["compute_profile_id"] = selected_compute
        metadata = {
            "id": kernel_ref,
            "title": f"KuaiRand autonomous agent {sequence:03d}",
            "code_file": worker_path.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if gpu else "false",
            "enable_internet": "true",
            "dataset_sources": [self.dataset_source],
            "competition_sources": [],
            "kernel_sources": [],
        }
        write_json_atomic(kernel_dir / "kernel-metadata.json", metadata)
        stage("dispatching", f"Submitting the reviewed experiment to Kaggle on {selected_compute}.", {"kernel_ref": kernel_ref, "compute_profile_id": selected_compute})
        self._kaggle(["kernels", "push", "-p", str(kernel_dir)])

        stage("running", "Kaggle is training and evaluating the candidate.", {"kernel_ref": kernel_ref})
        while True:
            if should_stop():
                return proposal, {
                    "status": "stopped", "stage": "remote_execution", "metrics": {},
                    "error": "Operator requested stop; the remote Kaggle run may finish independently.",
                    "kernel_ref": kernel_ref,
                }
            status = self._kaggle(["kernels", "status", kernel_ref], check=False)
            normalized = status.upper()
            if "COMPLETE" in normalized:
                break
            if any(word in normalized for word in ("ERROR", "FAILED", "CANCEL", "DENIED", "CANNOT ACCESS")):
                return proposal, {
                    "status": "failed", "stage": "remote_execution", "metrics": {},
                    "error": status.strip()[-1200:], "kernel_ref": kernel_ref,
                }
            stage("running", "Kaggle is still training; the controller will check again automatically.", {"kernel_ref": kernel_ref})
            time.sleep(self.poll_seconds)

        output_dir = kernel_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        stage("downloading", "Collecting the trusted evaluator result.", {"kernel_ref": kernel_ref})
        self._kaggle(["kernels", "output", kernel_ref, "-p", str(output_dir), "--force"])
        result_path = output_dir / "candidate_result.json"
        if not result_path.exists():
            return proposal, {
                "status": "failed", "stage": "result", "metrics": {},
                "error": "Kaggle completed without candidate_result.json", "kernel_ref": kernel_ref,
            }
        trusted = read_json(result_path)
        result = trusted.get("result", trusted)
        result.setdefault("counts_as_experiment", result.get("status") == "completed")
        if result.get("status") != "completed":
            result.setdefault("failure_type", "implementation_failed")
        result["kernel_ref"] = kernel_ref
        result["kernel_url"] = f"https://www.kaggle.com/code/{kernel_ref}"
        stage("reflecting", "Recording the result and updating experiment memory.", {"kernel_ref": kernel_ref})
        reflect(result_path, generation_dir / "proposal.json", self.root / "configs/default.json", self.memory_path)
        return proposal, result

    def _kaggle(self, arguments: list[str], check: bool = True) -> str:
        env = os.environ.copy()
        extra_pythonpath = os.getenv("KAGGLE_PYTHONPATH") or self.config.get("campaign", {}).get("kaggle_pythonpath")
        if extra_pythonpath:
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(extra_pythonpath), env.get("PYTHONPATH")]))
        command = [sys.executable, "-m", "kaggle", *arguments]
        completed = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True, check=False)
        output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
        if check and completed.returncode != 0:
            raise RuntimeError(f"Kaggle command failed: {output[-1600:]}")
        return output
