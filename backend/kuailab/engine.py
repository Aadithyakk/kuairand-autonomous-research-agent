from __future__ import annotations

import difflib
import json
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .benchmark import CommandBenchmark, SyntheticBenchmark
from .config import Settings
from .provider import DemoProvider, OpenAIProvider, Proposal
from .state import StateStore, utc_now


STAGES = ("inspect", "hypothesize", "implement", "train", "evaluate", "reflect")


class CampaignEngine:
    def __init__(self, settings: Settings, store: StateStore):
        self.settings = settings
        self.store = store
        self._thread: threading.Thread | None = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._guard = threading.Lock()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, mode: str, provider_name: str) -> None:
        if mode not in {"demo", "kuairand"}:
            raise ValueError("mode must be demo or kuairand")
        if provider_name not in {"demo", "gpt"}:
            raise ValueError("provider must be demo or gpt")
        if provider_name == "gpt" and not self.settings.api_key_available:
            raise RuntimeError("GPT mode needs a newly rotated OPENAI_API_KEY in the backend environment")
        with self._guard:
            if self.running:
                raise RuntimeError("A campaign is already running")
            if mode == "kuairand":
                CommandBenchmark(self.settings.experiment_command, self.settings.dataset_path, self.settings.run_timeout_seconds)
            self._pause.clear()
            self._stop.clear()
            campaign_id = f"run-{uuid.uuid4().hex[:10]}"

            def initialize(state: dict) -> None:
                baseline = {"primary": 0.6016, "gauc": 0.6612, "ndcg5": 0.5310}
                state["campaign"] = {
                    "id": campaign_id, "status": "running", "mode": mode, "provider": provider_name,
                    "started_at": utc_now(), "ended_at": None, "stop_reason": None, "steering": None,
                }
                state["current"] = None
                state["metrics"] = {"baseline": baseline, "champion": baseline, "delta": 0.0}
                state["usage"] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "wall_seconds": 0.0}
                state["events"] = []
                state["iterations"] = [{
                    "number": 0, "title": "Demo FM baseline" if mode == "demo" else "Measuring organizer baseline",
                    "status": "baseline", "stage": "complete" if mode == "demo" else "waiting", "metrics": baseline if mode == "demo" else None,
                    "delta": 0.0 if mode == "demo" else None, "accepted": True, "duration_seconds": 0.0,
                    "provider": "demo" if mode == "demo" else "organizer", "evidence": "synthetic-demo" if mode == "demo" else None,
                }]

            self.store.update(initialize)
            self.store.event("campaign", "Campaign started", f"{provider_name.upper()} researcher · {mode} benchmark")
            self._thread = threading.Thread(target=self._run, args=(mode, provider_name), daemon=True, name="kuailab-campaign")
            self._thread.start()

    def pause(self) -> None:
        if not self.running:
            raise RuntimeError("No campaign is running")
        self._pause.set()
        self.store.update(lambda state: state["campaign"].update(status="paused"))
        self.store.event("control", "Campaign paused", "The current safe stage boundary is being held.")

    def resume(self) -> None:
        if not self.running:
            raise RuntimeError("No campaign is running")
        self._pause.clear()
        self.store.update(lambda state: state["campaign"].update(status="running"))
        self.store.event("control", "Campaign resumed", "Autonomous iteration resumed.")

    def stop(self) -> None:
        if not self.running:
            raise RuntimeError("No campaign is running")
        self._stop.set()
        self._pause.clear()
        self.store.update(lambda state: state["campaign"].update(status="stopping"))
        self.store.event("control", "Stop requested", "The engine will stop at the next safe boundary.")

    def steer(self, instruction: str) -> None:
        clean = instruction.strip()[:1000]
        if not clean:
            raise ValueError("Steering instruction cannot be empty")
        self.store.update(lambda state: state["campaign"].update(steering=clean))
        self.store.event("control", "Operator guidance queued", clean)

    def reset(self) -> None:
        if self.running:
            raise RuntimeError("Stop the campaign before resetting")
        self.store.reset(self.settings.public_dict())

    def _wait_if_paused(self) -> bool:
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.15)
        return self._stop.is_set()

    def _set_stage(self, number: int, stage: str, note: str) -> None:
        def mutate(state: dict) -> None:
            current = state["current"]
            if not current or current["number"] != number:
                return
            current["stage"] = stage
            for item in current["stages"]:
                if STAGES.index(item["name"]) < STAGES.index(stage):
                    item["status"] = "done"
                elif item["name"] == stage:
                    item["status"] = "active"
                else:
                    item["status"] = "waiting"
            current["activity"] = note
        self.store.update(mutate)
        self.store.event("stage", stage.capitalize(), note, number, stage)
        if self.settings.stage_delay_seconds:
            time.sleep(self.settings.stage_delay_seconds)

    def _workspace(self, campaign_id: str, number: int) -> Path:
        path = self.settings.state_dir / "campaigns" / campaign_id / f"iteration-{number:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _save_proposal(workspace: Path, proposal: Proposal) -> None:
        payload = asdict(proposal)
        code = payload.pop("code")
        (workspace / "proposal.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        (workspace / "candidate.py").write_text(code, encoding="utf-8")
        diff = difflib.unified_diff([], code.splitlines(keepends=True), fromfile="champion/candidate.py", tofile="candidate/candidate.py")
        (workspace / "candidate.diff").write_text("".join(diff), encoding="utf-8")

    def _provider(self, name: str):
        return DemoProvider() if name == "demo" else OpenAIProvider(self.settings.model, self.settings.reasoning_effort)

    def _benchmark(self, mode: str):
        if mode == "demo":
            return SyntheticBenchmark()
        return CommandBenchmark(self.settings.experiment_command, self.settings.dataset_path, self.settings.run_timeout_seconds)

    def _run(self, mode: str, provider_name: str) -> None:
        started = time.monotonic()
        provider = self._provider(provider_name)
        benchmark = self._benchmark(mode)
        campaign = self.store.snapshot()["campaign"]
        campaign_id = campaign["id"]
        small_gain_streak = 0
        stop_reason = "iteration budget reached"
        try:
            if mode == "kuairand":
                baseline_workspace = self.settings.state_dir / "campaigns" / campaign_id / "baseline"
                baseline_workspace.mkdir(parents=True, exist_ok=True)
                self.store.event("stage", "Measuring organizer baseline", "Running the unmodified official starter-kit baseline on validation.", 0, "baseline")
                measured = benchmark.baseline(baseline_workspace)
                baseline_metrics = measured.metrics()
                def save_baseline(state: dict) -> None:
                    state["metrics"] = {"baseline": baseline_metrics, "champion": baseline_metrics, "delta": 0.0}
                    state["iterations"][0].update(title="Organizer baseline", stage="complete", metrics=baseline_metrics,
                                                  delta=0.0, duration_seconds=measured.runtime_seconds,
                                                  evidence=measured.evidence, artifact=str(baseline_workspace))
                self.store.update(save_baseline)
                self.store.event("result", "Baseline reproduced", f"Primary {baseline_metrics['primary']:.4f} · GAUC {baseline_metrics['gauc']:.4f} · nDCG@5 {baseline_metrics['ndcg5']:.4f}", 0, "baseline")
            for number in range(1, self.settings.max_iterations + 1):
                if self._stop.is_set():
                    stop_reason = "operator stopped"
                    break
                elapsed = time.monotonic() - started
                if elapsed >= self.settings.max_hours * 3600:
                    stop_reason = "wall-clock budget reached"
                    break
                iteration_started = time.monotonic()
                workspace = self._workspace(campaign_id, number)
                champion_before = self.store.snapshot()["metrics"]["champion"]

                def begin(state: dict) -> None:
                    state["current"] = {
                        "number": number, "title": "Designing next experiment", "hypothesis": "Inspecting evidence…",
                        "stage": "inspect", "status": "running", "activity": "Reading campaign evidence and resource limits.",
                        "stages": [{"name": name, "status": "active" if name == "inspect" else "waiting"} for name in STAGES],
                        "acceptance": f"Validation primary gain ≥ {self.settings.convergence_epsilon:.4f}",
                        "abort_condition": "Invalid output, timeout, or runner failure", "expected_gain": None,
                    }
                self.store.update(begin)
                try:
                    self._set_stage(number, "inspect", "Reviewing champion metrics, prior failures, and remaining budget.")
                    if self._wait_if_paused():
                        stop_reason = "operator stopped"
                        break
                    snapshot = self.store.snapshot()
                    context = {
                        "task": "KuaiRand-Pure long_view recommendation ranking",
                        "iteration": number,
                        "baseline": snapshot["metrics"]["baseline"],
                        "champion": champion_before,
                        "prior_iterations": snapshot["iterations"][-6:],
                        "epsilon": self.settings.convergence_epsilon,
                        "remaining_iterations": self.settings.max_iterations - number + 1,
                        "steering": snapshot["campaign"].get("steering"),
                        "constraints": ["GAUC and nDCG@5 validation", "no hidden-test access", "one isolated change", "must be reproducible"],
                        "executor_contract": {
                            "runtime": "NumPy-only official Factorization Machine; no Torch and no arbitrary package installation",
                            "experiment_types": {
                                "fm_config": "One FM with typed k/lr/epochs/batch_size/patience/seed parameters",
                                "fm_positive_weight": "FM logistic loss with the supplied positive_weight in [1,10]",
                                "fm_ensemble": "Mean validation logits from 1-3 independently trained FM seeds"
                            },
                            "defaults": {"k": 16, "lr": 0.001, "epochs": 40, "batch_size": 8192, "patience": 4, "seed": 0, "ensemble_seeds": [0, 1, 2], "positive_weight": 1.0},
                            "rule": "Select exactly one supported experiment_type and populate every typed parameter. Generated code is evidence; the trusted executor applies the typed change."
                        },
                    }
                    self._set_stage(number, "hypothesize", f"Asking {self.settings.model if provider_name == 'gpt' else 'deterministic demo planner'} for one falsifiable experiment.")
                    proposal = provider.propose(context)

                    def proposed(state: dict) -> None:
                        current = state["current"]
                        current.update(title=proposal.title, hypothesis=proposal.hypothesis, acceptance=proposal.acceptance,
                                       abort_condition=proposal.abort_condition, expected_gain=proposal.expected_gain)
                        for key, value in proposal.usage.items():
                            state["usage"][key] += value
                        state["campaign"]["steering"] = None
                    self.store.update(proposed)
                    self.store.event("decision", "Hypothesis selected", proposal.hypothesis, number, "hypothesize")
                    if self._wait_if_paused():
                        stop_reason = "operator stopped"
                        break

                    self._set_stage(number, "implement", "Writing an auditable proposal, candidate code, and unified diff.")
                    self._save_proposal(workspace, proposal)
                    if self._wait_if_paused():
                        stop_reason = "operator stopped"
                        break

                    self._set_stage(number, "train", "Running the sealed benchmark adapter; generated code is not executed on the host directly.")
                    evaluation = benchmark.evaluate(proposal, number, workspace)
                    self._set_stage(number, "evaluate", "Validating metric bounds and comparing against the retained champion.")
                    metrics = evaluation.metrics()
                    gain = round(metrics["primary"] - champion_before["primary"], 6)
                    accepted = gain > 0
                    baseline_delta = round(metrics["primary"] - snapshot["metrics"]["baseline"]["primary"], 6)

                    self._set_stage(number, "reflect", "Recording the result, promotion decision, resources, and next-step evidence.")
                    duration = round(time.monotonic() - iteration_started, 2)
                    record = {
                        "number": number, "title": proposal.title, "hypothesis": proposal.hypothesis,
                        "status": "accepted" if accepted else "rejected", "stage": "complete", "metrics": metrics,
                        "delta": baseline_delta, "gain": gain, "accepted": accepted, "duration_seconds": duration,
                        "provider": provider_name, "evidence": evaluation.evidence, "artifact": str(workspace),
                        "change_summary": proposal.change_summary, "response_id": proposal.response_id,
                    }

                    def finish_iteration(state: dict) -> None:
                        state["iterations"].append(record)
                        if accepted:
                            state["metrics"]["champion"] = metrics
                            state["metrics"]["delta"] = baseline_delta
                        state["current"]["status"] = "complete"
                        for item in state["current"]["stages"]:
                            item["status"] = "done"
                        state["usage"]["wall_seconds"] = round(time.monotonic() - started, 2)
                    self.store.update(finish_iteration)
                    action = "Champion promoted" if accepted else "Candidate rejected"
                    self.store.event("result", action, f"Primary {metrics['primary']:.4f} · gain {gain:+.4f} · {evaluation.evidence}", number, "reflect")
                    small_gain_streak = small_gain_streak + 1 if gain < self.settings.convergence_epsilon else 0
                    if small_gain_streak >= self.settings.convergence_patience:
                        stop_reason = f"converged: {small_gain_streak} consecutive gains below {self.settings.convergence_epsilon:.4f}"
                        break
                except Exception as error:
                    duration = round(time.monotonic() - iteration_started, 2)
                    failed = {
                        "number": number, "title": self.store.snapshot().get("current", {}).get("title", "Experiment"),
                        "status": "failed", "stage": "failed", "metrics": None, "delta": None, "accepted": False,
                        "duration_seconds": duration, "provider": provider_name, "error": str(error)[:800], "artifact": str(workspace),
                    }
                    def record_failure(state: dict) -> None:
                        state["iterations"].append(failed)
                        if state["current"]:
                            state["current"].update(status="failed", error=failed["error"])
                        state["usage"]["wall_seconds"] = round(time.monotonic() - started, 2)
                    self.store.update(record_failure)
                    self.store.event("error", "Iteration failed · recovered", failed["error"], number, "failed")
                    if provider_name == "gpt" and "OpenAI Responses API failed" in str(error):
                        stop_reason = "model provider error"
                        break
                    continue
        except Exception as error:
            stop_reason = f"campaign error: {str(error)[:300]}"
            self.store.event("error", "Campaign stopped unexpectedly", stop_reason)
        finally:
            final_status = "stopped" if self._stop.is_set() else "complete"
            def complete(state: dict) -> None:
                state["campaign"].update(status=final_status, ended_at=utc_now(), stop_reason=stop_reason)
                state["usage"]["wall_seconds"] = round(time.monotonic() - started, 2)
            self.store.update(complete)
            self.store.event("campaign", "Campaign finished", stop_reason)
