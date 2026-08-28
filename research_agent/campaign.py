from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable

from .core import read_json, utc_now, write_json_atomic
from .kaggle_packager import package
from .real_pilot import generate, reflect, repair_runtime_candidate


StageCallback = Callable[[str, str, dict[str, Any] | None], None]
StopCallback = Callable[[], bool]


def requires_gpu(model_family: str, compute_preference: str) -> bool:
    preference = compute_preference.strip().lower()
    family_tokens = set(model_family.lower().replace("-", "_").split("_"))
    return preference in {"gpu", "t4", "cuda"} or bool(
        family_tokens.intersection({"sequence", "sequential", "neural", "transformer", "din", "dien", "sasrec"})
    )


def is_repairable_runtime_result(result: dict[str, Any], attempt: int, maximum: int, stopped: bool = False) -> bool:
    return bool(
        not stopped
        and attempt < maximum
        and result.get("failure_type") == "implementation_failed"
        and result.get("stage") in {"smoke", "execution", "output", "safety"}
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
        local_execution = compute_profile_id == "local-cpu"
        if not local_execution and not os.getenv("KAGGLE_API_TOKEN"):
            raise RuntimeError("KAGGLE_API_TOKEN is not present in the controller process")
        if not local_execution and not self.kaggle_username:
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
        # Kaggle derives the public slug from the title, even when metadata.id
        # requests another value. Keep both exactly aligned so status polling
        # and output retrieval address the kernel Kaggle actually creates.
        slug = f"kuairand-autonomous-agent-{sequence:03d}"
        kernel_ref = f"local-cpu/{run_name}" if local_execution else f"{self.kaggle_username}/{slug}"
        family = str(proposal.get("model_family", "")).lower()
        preference = str(proposal.get("compute_preference", "auto")).lower()
        gpu = requires_gpu(family, preference)
        selected_compute = "local-cpu" if local_execution else ("kaggle-t4" if gpu else "kaggle-cpu")
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
        output_dir = kernel_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime_repairs: list[dict[str, Any]] = []
        max_runtime_repairs = int(self.config.get("campaign", {}).get("runtime_repairs_per_candidate", 2))
        result_path: Path | None = None
        result: dict[str, Any] = {}
        for runtime_attempt in range(max_runtime_repairs + 1):
            package(
                self.root / "outputs/kuairand_candidate_worker.py",
                generation_dir / "candidate.py",
                generation_dir / "proposal.json",
                worker_path,
            )
            result, result_path = self._dispatch_and_collect(
                kernel_dir, output_dir, kernel_ref, selected_compute, stage, should_stop, runtime_attempt
            )
            result["kernel_ref"] = kernel_ref
            if not local_execution:
                result["kernel_url"] = f"https://www.kaggle.com/code/{kernel_ref}"
            repairable = is_repairable_runtime_result(result, runtime_attempt, max_runtime_repairs, should_stop())
            if not repairable:
                break
            trace = [str(result.get("error", "Runtime implementation failure"))]
            if result.get("stderr_tail"):
                trace.append(str(result["stderr_tail"]))
            smoke = result.get("smoke", {})
            if isinstance(smoke, dict) and smoke.get("stderr_tail"):
                trace.append(str(smoke["stderr_tail"]))
            stage(
                "runtime_repair",
                f"Trusted smoke failed; repairing the same hypothesis ({runtime_attempt + 1}/{max_runtime_repairs}).",
                {"proposal": proposal, "kernel_ref": kernel_ref},
            )
            repair_dir = generation_dir / f"runtime-repair-{runtime_attempt + 1:02d}"
            try:
                repaired = repair_runtime_candidate(
                    self.root / "configs/default.json",
                    self.memory_path,
                    proposal,
                    (generation_dir / "candidate.py").read_text(encoding="utf-8"),
                    trace,
                    repair_dir,
                    progress=lambda phase, message: stage(phase, message, {"proposal": proposal}),
                    should_stop=should_stop,
                )
            except RuntimeError as exc:
                runtime_repairs.append({"attempt": runtime_attempt + 1, "approved": False, "error": str(exc)})
                result["error"] = f"Runtime repair failed: {exc}"
                break
            runtime_repairs.append({
                "attempt": runtime_attempt + 1,
                "approved": repaired["approved"],
                "findings": repaired["findings"],
                "reviews": repaired["reviews"],
            })
            if not repaired["approved"]:
                result["error"] = "Runtime repair did not pass deterministic and methodological re-review"
                break
            (generation_dir / "candidate.py").write_text(repaired["source"], encoding="utf-8")
            stage("repaired", "The same candidate passed runtime repair and will be resubmitted.", {"proposal": proposal})
        result["runtime_repairs"] = runtime_repairs
        if result_path and result_path.exists():
            stage("reflecting", "Recording the final result and updating experiment memory.", {"kernel_ref": kernel_ref})
            reflect(result_path, generation_dir / "proposal.json", self.root / "configs/default.json", self.memory_path)
        return proposal, result

    def _dispatch_and_collect(
        self,
        kernel_dir: Path,
        output_dir: Path,
        kernel_ref: str,
        selected_compute: str,
        stage: StageCallback,
        should_stop: StopCallback,
        runtime_attempt: int,
    ) -> tuple[dict[str, Any], Path | None]:
        prior_result = output_dir / "candidate_result.json"
        if prior_result.exists():
            prior_result.replace(output_dir / f"candidate_result.before-runtime-repair-{runtime_attempt:02d}.json")
        if selected_compute == "local-cpu":
            return self._run_local(
                kernel_dir, output_dir, kernel_ref, stage, should_stop, runtime_attempt
            )
        stage(
            "dispatching",
            f"Submitting the reviewed experiment to Kaggle on {selected_compute}.",
            {"kernel_ref": kernel_ref, "compute_profile_id": selected_compute},
        )
        self._kaggle(["kernels", "push", "-p", str(kernel_dir)])
        stage("running", "Kaggle is training and evaluating the candidate.", {"kernel_ref": kernel_ref})
        while True:
            if should_stop():
                return {
                    "status": "stopped", "stage": "remote_execution", "metrics": {},
                    "counts_as_experiment": False,
                    "error": "Operator requested stop; the remote Kaggle run may finish independently.",
                }, None
            status = self._kaggle(["kernels", "status", kernel_ref], check=False)
            normalized = status.upper()
            if "COMPLETE" in normalized:
                break
            if any(word in normalized for word in ("ERROR", "FAILED", "CANCEL", "DENIED", "CANNOT ACCESS")):
                return {
                    "status": "failed", "stage": "remote_execution", "metrics": {},
                    "counts_as_experiment": False, "failure_type": "infrastructure_failed",
                    "error": status.strip()[-1200:],
                }, None
            stage("running", "Kaggle is still training; the controller will check again automatically.", {"kernel_ref": kernel_ref})
            time.sleep(self.poll_seconds)
        stage("downloading", "Collecting the trusted evaluator result.", {"kernel_ref": kernel_ref})
        self._kaggle(["kernels", "output", kernel_ref, "-p", str(output_dir), "--force"])
        result_path = output_dir / "candidate_result.json"
        if not result_path.exists():
            return {
                "status": "failed", "stage": "result", "metrics": {}, "counts_as_experiment": False,
                "failure_type": "infrastructure_failed", "error": "Kaggle completed without candidate_result.json",
            }, None
        trusted = read_json(result_path)
        result = trusted.get("result", trusted)
        result.setdefault("counts_as_experiment", result.get("status") in {"completed", "screen_rejected"})
        if result.get("status") != "completed":
            result.setdefault("failure_type", "implementation_failed")
        return result, result_path

    def _local_data_root(self) -> Path:
        """Extract the checked-in KuaiRand archive once for trusted local evaluation."""

        campaign = self.config.get("campaign", {})
        archive_path = self.root / campaign.get("local_dataset_archive", "work/kuairand/KuaiRand-Pure.tar.gz")
        data_root = self.root / campaign.get("local_data_root", "runtime/local-data")
        expected = data_root / "KuaiRand-Pure" / "data" / "log_standard_4_08_to_4_21_pure.csv"
        if expected.exists():
            return data_root
        if not archive_path.exists():
            raise RuntimeError(f"Local KuaiRand archive is missing: {archive_path}")
        data_root.mkdir(parents=True, exist_ok=True)
        destination = data_root.resolve()
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if target != destination and destination not in target.parents:
                    raise RuntimeError(f"Unsafe path in local dataset archive: {member.name}")
            archive.extractall(data_root)
        if not expected.exists():
            raise RuntimeError("Local KuaiRand archive did not contain the expected benchmark files")
        return data_root

    def _run_local(
        self,
        kernel_dir: Path,
        output_dir: Path,
        kernel_ref: str,
        stage: StageCallback,
        should_stop: StopCallback,
        runtime_attempt: int,
    ) -> tuple[dict[str, Any], Path | None]:
        """Execute the same packaged trusted worker on the selected local CPU."""

        data_root = self._local_data_root()
        worker_path = kernel_dir / "kuairand_candidate_worker.py"
        result_path = output_dir / "candidate_result.json"
        local_work = output_dir / f"local-work-{runtime_attempt:02d}-{time.time_ns()}"
        stdout_path = output_dir / f"local-worker-{runtime_attempt:02d}.stdout.log"
        stderr_path = output_dir / f"local-worker-{runtime_attempt:02d}.stderr.log"
        env = os.environ.copy()
        env.update({
            "KUAIRAND_INPUT_ROOT": str(data_root),
            "KUAIRAND_WORK_ROOT": str(local_work),
            "KUAIRAND_REPORT_PATH": str(result_path),
            "KUAIRAND_EVENTS_PATH": str(output_dir / f"local-worker-{runtime_attempt:02d}.events.jsonl"),
        })
        stage(
            "running",
            "The reviewed experiment is training and evaluating on local CPU.",
            {"kernel_ref": kernel_ref, "compute_profile_id": "local-cpu"},
        )
        timeout = int(self.config.get("campaign", {}).get("local_worker_timeout_seconds", 1200))
        started = time.monotonic()
        last_pulse = 0.0
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                [sys.executable, str(worker_path)], cwd=self.root, env=env,
                stdout=stdout_handle, stderr=stderr_handle, text=True,
            )
            while process.poll() is None:
                if should_stop():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return {
                        "status": "stopped", "stage": "local_execution", "metrics": {},
                        "counts_as_experiment": False, "error": "Operator requested stop during local execution.",
                    }, None
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return {
                        "status": "failed", "stage": "local_execution", "metrics": {},
                        "counts_as_experiment": False, "failure_type": "infrastructure_failed",
                        "error": f"Local trusted worker exceeded {timeout} seconds",
                    }, None
                if elapsed - last_pulse >= 15:
                    stage(
                        "running", "Local CPU training is active; the controller is supervising it.",
                        {"kernel_ref": kernel_ref, "compute_profile_id": "local-cpu"},
                    )
                    last_pulse = elapsed
                time.sleep(2)
        if process.returncode != 0 and not result_path.exists():
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            return {
                "status": "failed", "stage": "local_execution", "metrics": {},
                "counts_as_experiment": False, "failure_type": "infrastructure_failed",
                "error": f"Local trusted worker exited {process.returncode}", "stderr_tail": stderr_tail,
            }, None
        if not result_path.exists():
            return {
                "status": "failed", "stage": "result", "metrics": {},
                "counts_as_experiment": False, "failure_type": "infrastructure_failed",
                "error": "Local trusted worker completed without candidate_result.json",
            }, None
        stage("downloading", "Collecting the local trusted evaluator result.", {"kernel_ref": kernel_ref})
        trusted = read_json(result_path)
        result = trusted.get("result", trusted)
        result.setdefault("counts_as_experiment", result.get("status") in {"completed", "screen_rejected"})
        if result.get("status") != "completed":
            result.setdefault("failure_type", "implementation_failed")
        return result, result_path

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
