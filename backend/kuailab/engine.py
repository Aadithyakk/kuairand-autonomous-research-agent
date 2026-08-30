from __future__ import annotations

import difflib
import json
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .benchmark import BenchmarkRunError, CommandBenchmark, SyntheticBenchmark, validate_metrics
from .config import Settings
from .provider import DemoProvider, OpenAIProvider, Proposal
from .resources import add_resource_usage, combine_resource_usage, empty_campaign_usage
from .state import StateStore, utc_now


STAGES = ("inspect", "hypothesize", "implement", "train", "evaluate", "reflect")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_MAX_ITERATIONS = 50
OFFICIAL_MAX_HOURS = 6.0
OFFICIAL_CONVERGENCE_EPSILON = 0.002
OFFICIAL_CONVERGENCE_PATIENCE = 3


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

    def _limits(self, raw: dict | None, *, bootstrap_verified: bool = True) -> dict:
        values = raw if isinstance(raw, dict) else {}
        limits = {
            "max_iterations": int(values.get("max_iterations", self.settings.max_iterations)),
            "max_hours": float(values.get("max_hours", self.settings.max_hours)),
            "convergence_epsilon": float(values.get("convergence_epsilon", self.settings.convergence_epsilon)),
            "convergence_patience": int(values.get("convergence_patience", self.settings.convergence_patience)),
            "bootstrap_verified": bool(bootstrap_verified),
        }
        if not 1 <= limits["max_iterations"] <= OFFICIAL_MAX_ITERATIONS:
            raise ValueError(f"max_iterations must be between 1 and {OFFICIAL_MAX_ITERATIONS}")
        if not 0.1 <= limits["max_hours"] <= OFFICIAL_MAX_HOURS:
            raise ValueError(f"max_hours must be between 0.1 and {OFFICIAL_MAX_HOURS:g}")
        if not 0 <= limits["convergence_epsilon"] <= 0.01:
            raise ValueError("convergence_epsilon must be between 0 and 0.01")
        if not 0 <= limits["convergence_patience"] <= 50:
            raise ValueError("convergence_patience must be between 0 and 50")
        return limits

    def start(
        self,
        mode: str,
        provider_name: str,
        raw_limits: dict | None = None,
        *,
        bootstrap_verified: bool = True,
    ) -> None:
        if mode not in {"demo", "kuairand"}:
            raise ValueError("mode must be demo or kuairand")
        if provider_name not in {"demo", "gpt"}:
            raise ValueError("provider must be demo or gpt")
        if provider_name == "gpt" and not self.settings.api_key_available:
            raise RuntimeError("GPT mode needs a newly rotated OPENAI_API_KEY in the backend environment")
        limits = self._limits(raw_limits, bootstrap_verified=bootstrap_verified)
        if mode == "kuairand":
            limits["convergence_epsilon"] = OFFICIAL_CONVERGENCE_EPSILON
            limits["convergence_patience"] = OFFICIAL_CONVERGENCE_PATIENCE
        with self._guard:
            if self.running:
                raise RuntimeError("A campaign is already running")
            if mode == "kuairand":
                CommandBenchmark(self.settings.experiment_command, self.settings.dataset_path, self.settings.run_timeout_seconds)
            self._pause.clear()
            self._stop.clear()
            campaign_id = f"run-{uuid.uuid4().hex[:10]}"

            def initialize(state: dict) -> None:
                baseline = {"primary": 0.60155, "gauc": 0.6674, "ndcg5": 0.5357}
                state["campaign"] = {
                    "id": campaign_id, "status": "running", "mode": mode, "provider": provider_name,
                    "started_at": utc_now(), "ended_at": None, "stop_reason": None, "steering": None,
                    "continuations": 0, "session_started_at": utc_now(), "session_start_iteration": 1,
                    "session_start_wall_seconds": 0.0, "limits": limits,
                    "manual_interventions": 0, "failure_count": 0, "recovery_count": 0,
                    "consecutive_small_gains": 0,
                }
                state["current"] = None
                state["metrics"] = {"baseline": baseline, "champion": baseline, "delta": 0.0}
                state["usage"] = empty_campaign_usage()
                state["events"] = []
                state["iterations"] = [{
                    "number": 0, "title": "Demo FM baseline" if mode == "demo" else "Measuring organizer baseline",
                    "status": "baseline", "stage": "complete" if mode == "demo" else "waiting", "metrics": baseline if mode == "demo" else None,
                    "delta": 0.0 if mode == "demo" else None, "accepted": True, "duration_seconds": 0.0,
                    "provider": "demo" if mode == "demo" else "organizer", "evidence": "synthetic-demo" if mode == "demo" else None,
                    "resource_usage": None, "budget_counted": False,
                }]

            self.store.update(initialize)
            self.store.event("campaign", "Campaign started", f"{provider_name.upper()} researcher · {mode} benchmark")
            self._thread = threading.Thread(
                target=self._run,
                args=(mode, provider_name, 1, limits, False),
                daemon=True,
                name="kuailab-campaign",
            )
            self._thread.start()

    def continue_campaign(self, raw_limits: dict | None = None) -> None:
        snapshot = self.store.snapshot()
        campaign = snapshot.get("campaign", {})
        if not campaign.get("id") or not snapshot.get("iterations"):
            raise RuntimeError("There is no retained campaign to continue")
        mode = campaign.get("mode")
        provider_name = campaign.get("provider")
        if mode not in {"demo", "kuairand"} or provider_name not in {"demo", "gpt"}:
            raise RuntimeError("The retained campaign has an invalid mode or provider")
        if provider_name == "gpt" and not self.settings.api_key_available:
            raise RuntimeError("Continuing GPT research needs OPENAI_API_KEY in the backend environment")
        limits = self._limits(raw_limits, bootstrap_verified=False)
        if mode == "kuairand":
            limits["convergence_epsilon"] = OFFICIAL_CONVERGENCE_EPSILON
            limits["convergence_patience"] = OFFICIAL_CONVERGENCE_PATIENCE
        used = self._budget_iterations(snapshot)
        if used >= OFFICIAL_MAX_ITERATIONS:
            raise RuntimeError("The official 50-iteration campaign budget is exhausted")
        if float(snapshot.get("usage", {}).get("wall_seconds", 0.0)) >= OFFICIAL_MAX_HOURS * 3600:
            raise RuntimeError("The official six-hour campaign budget is exhausted")
        limits["max_iterations"] = min(limits["max_iterations"], OFFICIAL_MAX_ITERATIONS - used)
        with self._guard:
            if self.running:
                raise RuntimeError("A campaign is already running")
            if mode == "kuairand":
                CommandBenchmark(self.settings.experiment_command, self.settings.dataset_path, self.settings.run_timeout_seconds)
            self._pause.clear()
            self._stop.clear()
            start_number = max(int(item.get("number", 0)) for item in snapshot["iterations"]) + 1
            wall_before = float(snapshot.get("usage", {}).get("wall_seconds", 0.0))

            def resume_retained(state: dict) -> None:
                state["campaign"].update(
                    status="running",
                    ended_at=None,
                    stop_reason=None,
                    steering=None,
                    session_started_at=utc_now(),
                    session_start_iteration=start_number,
                    session_start_wall_seconds=wall_before,
                    limits=limits,
                    continuations=int(state["campaign"].get("continuations", 0)) + 1,
                )
                state["current"] = None

            self.store.update(resume_retained)
            self.store.event(
                "campaign",
                "Campaign continued",
                f"Retained champion {snapshot['metrics']['champion']['primary']:.6f} · next iteration {start_number:03d}",
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(mode, provider_name, start_number, limits, True),
                daemon=True,
                name="kuailab-campaign",
            )
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
        def queue(state: dict) -> None:
            state["campaign"].update(
                steering=clean,
                manual_interventions=int(state["campaign"].get("manual_interventions", 0)) + 1,
            )
        self.store.update(queue)
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

    @staticmethod
    def _budget_iterations(snapshot: dict) -> int:
        return sum(
            1 for item in snapshot.get("iterations", [])
            if item.get("budget_counted", item.get("number", 0) > 0 and not item.get("imported"))
        )

    @staticmethod
    def _small_gain_streak(snapshot: dict, epsilon: float) -> int:
        streak = 0
        for item in reversed(snapshot.get("iterations", [])):
            if not item.get("budget_counted", item.get("number", 0) > 0 and not item.get("imported")):
                continue
            gain = item.get("gain")
            if gain is not None and float(gain) > epsilon:
                break
            streak += 1
        return streak

    def _propose_with_retry(self, provider, context: dict, number: int) -> Proposal:
        try:
            return provider.propose(context)
        except Exception as first_error:
            def mark(state: dict) -> None:
                state["campaign"]["failure_count"] = int(state["campaign"].get("failure_count", 0)) + 1
                state["campaign"]["recovery_count"] = int(state["campaign"].get("recovery_count", 0)) + 1
            self.store.update(mark)
            self.store.event(
                "recovery", "Planner retry routed", f"First proposal request failed: {str(first_error)[:400]}",
                number, "hypothesize",
            )
            return provider.propose({**context, "recovery_instruction": "Retry once with the same sealed executor contract."})

    @staticmethod
    def _recovery_proposal(proposal: Proposal, error: Exception) -> Proposal:
        parameters = dict(proposal.parameters)
        changes: list[str] = []
        for key, minimum in (("batch_size", 2048), ("deep_threads", 1)):
            value = parameters.get(key)
            if isinstance(value, int) and value > minimum:
                reduced = max(minimum, value // 2)
                parameters[key] = reduced
                changes.append(f"{key} {value}->{reduced}")
        if not changes:
            value = parameters.get("epochs")
            if isinstance(value, int) and value > 5:
                reduced = max(5, int(value * 0.75))
                parameters["epochs"] = reduced
                changes.append(f"epochs {value}->{reduced}")
        recovery = ", ".join(changes) if changes else "repeat once after an isolated worker failure"
        code = proposal.code.rstrip() + f"\n# Automatic recovery route: {recovery}\n"
        return Proposal(
            title=f"{proposal.title} · recovery route",
            experiment_type=proposal.experiment_type,
            hypothesis=proposal.hypothesis,
            rationale=f"{proposal.rationale} Automatic recovery after: {str(error)[:240]}",
            change_summary=f"{proposal.change_summary}; recovery: {recovery}",
            code=code,
            parameters=parameters,
            acceptance=proposal.acceptance,
            abort_condition=proposal.abort_condition,
            expected_gain=proposal.expected_gain,
            usage=proposal.usage,
            response_id=proposal.response_id,
        )

    def _evaluate_with_recovery(self, benchmark, proposal: Proposal, number: int, workspace: Path):
        try:
            return proposal, benchmark.evaluate(proposal, number, workspace), []
        except BenchmarkRunError as first_error:
            retry_proposal = self._recovery_proposal(proposal, first_error)
            retry_workspace = workspace / "retry-1"
            retry_workspace.mkdir(parents=True, exist_ok=True)
            self._save_proposal(retry_workspace, retry_proposal)
            recovery = {
                "attempt": 1,
                "error": str(first_error)[:800],
                "route": retry_proposal.change_summary,
                "resource_usage": first_error.resource_usage,
            }
            def mark(state: dict) -> None:
                state["campaign"]["failure_count"] = int(state["campaign"].get("failure_count", 0)) + 1
                state["campaign"]["recovery_count"] = int(state["campaign"].get("recovery_count", 0)) + 1
            self.store.update(mark)
            self.store.event(
                "recovery", "Worker retry routed", f"{recovery['error']} · {recovery['route']}",
                number, "train",
            )
            try:
                evaluation = benchmark.evaluate(retry_proposal, number, retry_workspace)
            except BenchmarkRunError as retry_error:
                combined = combine_resource_usage(first_error.resource_usage, retry_error.resource_usage)
                combined_error = BenchmarkRunError(
                    f"Initial attempt failed ({first_error}); recovery attempt failed ({retry_error})",
                    combined,
                )
                recovery.update(
                    outcome="failed",
                    retry_error=str(retry_error)[:800],
                    retry_resource_usage=retry_error.resource_usage,
                )
                combined_error.recovery_events = [recovery]
                raise combined_error from retry_error
            evaluation.resource_usage = combine_resource_usage(first_error.resource_usage, evaluation.resource_usage)
            recovery["outcome"] = "recovered"
            return retry_proposal, evaluation, [recovery]

    @staticmethod
    def _write_iteration_log(workspace: Path, record: dict) -> None:
        (workspace / "iteration-log.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _verified_champion(self, baseline: dict) -> dict | None:
        evidence_dir = PROJECT_ROOT / "results" / "verified-slate-consensus"
        try:
            summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
            proposal = json.loads((evidence_dir / "proposal.json").read_text(encoding="utf-8"))
            champion = summary["champion"]
            metrics = {
                "primary": float(champion["primary"]),
                "gauc": float(champion["gauc"]),
                "ndcg5": float(champion["ndcg5"]),
            }
            validate_metrics(metrics)
            hidden_test_accessed = summary.get(
                "hidden_test_accessed", summary.get("split", {}).get("hidden_test_accessed")
            )
            if summary.get("target") != "long_view" or hidden_test_accessed is not False:
                return None
            if metrics["primary"] <= baseline["primary"]:
                return None
            return {
                "number": 1,
                "title": proposal.get("title", "Verified retained champion"),
                "hypothesis": proposal.get("hypothesis", "Restore the best cleanly verified validation recipe."),
                "status": "accepted",
                "stage": "complete",
                "metrics": metrics,
                "delta": round(metrics["primary"] - baseline["primary"], 6),
                "gain": round(metrics["primary"] - baseline["primary"], 6),
                "accepted": True,
                "duration_seconds": 0.0,
                "provider": "verified-evidence",
                "evidence": "kuairand-pure-validation-import",
                "artifact": str(evidence_dir),
                "change_summary": "Restored from the checked-in clean validation record; no hidden-test access.",
                "experiment_type": proposal.get("experiment_type"),
                "parameters": proposal.get("parameters", {}),
                "resource_usage": summary.get("resource_usage"),
                "imported": True,
                "budget_counted": False,
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _run(self, mode: str, provider_name: str, start_number: int, limits: dict, continuing: bool) -> None:
        started = time.monotonic()
        wall_before = float(self.store.snapshot().get("usage", {}).get("wall_seconds", 0.0)) if continuing else 0.0
        provider = self._provider(provider_name)
        benchmark = self._benchmark(mode)
        campaign = self.store.snapshot()["campaign"]
        campaign_id = campaign["id"]
        small_gain_streak = self._small_gain_streak(self.store.snapshot(), limits["convergence_epsilon"])
        stop_reason = "official iteration budget reached"
        try:
            if mode == "kuairand" and not continuing:
                baseline_workspace = self.settings.state_dir / "campaigns" / campaign_id / "baseline"
                baseline_workspace.mkdir(parents=True, exist_ok=True)
                self.store.event("stage", "Measuring organizer baseline", "Running the unmodified official starter-kit baseline on validation.", 0, "baseline")
                measured = benchmark.baseline(baseline_workspace)
                baseline_metrics = measured.metrics()
                def save_baseline(state: dict) -> None:
                    state["metrics"] = {"baseline": baseline_metrics, "champion": baseline_metrics, "delta": 0.0}
                    state["iterations"][0].update(title="Organizer baseline", stage="complete", metrics=baseline_metrics,
                                                  delta=0.0, duration_seconds=measured.runtime_seconds,
                                                  evidence=measured.evidence, artifact=str(baseline_workspace),
                                                  resource_usage=measured.resource_usage, budget_counted=False)
                    add_resource_usage(state["usage"], measured.resource_usage)
                self.store.update(save_baseline)
                self.store.event("result", "Baseline reproduced", f"Primary {baseline_metrics['primary']:.4f} · GAUC {baseline_metrics['gauc']:.4f} · nDCG@5 {baseline_metrics['ndcg5']:.4f}", 0, "baseline")
                if limits["bootstrap_verified"]:
                    retained = self._verified_champion(baseline_metrics)
                    if retained:
                        def restore_champion(state: dict) -> None:
                            state["iterations"].append(retained)
                            state["metrics"]["champion"] = retained["metrics"]
                            state["metrics"]["delta"] = retained["delta"]
                            state["campaign"]["session_start_iteration"] = 2
                        self.store.update(restore_champion)
                        self.store.event(
                            "result",
                            "Verified champion restored",
                            f"Primary {retained['metrics']['primary']:.6f} · continuing from checked-in validation-best evidence",
                            1,
                            "baseline",
                        )
                        start_number = 2
            snapshot = self.store.snapshot()
            used = self._budget_iterations(snapshot)
            remaining = min(limits["max_iterations"], OFFICIAL_MAX_ITERATIONS - used)
            if remaining <= 0:
                stop_reason = "official 50-iteration budget reached"
            end_number = start_number + max(0, remaining)
            for number in range(start_number, end_number):
                if self._stop.is_set():
                    stop_reason = "operator stopped"
                    break
                elapsed = time.monotonic() - started
                if wall_before + elapsed >= min(limits["max_hours"], OFFICIAL_MAX_HOURS) * 3600:
                    stop_reason = "official six-hour wall-clock budget reached"
                    break
                iteration_started = time.monotonic()
                workspace = self._workspace(campaign_id, number)
                champion_before = self.store.snapshot()["metrics"]["champion"]

                def begin(state: dict) -> None:
                    state["current"] = {
                        "number": number, "title": "Designing next experiment", "hypothesis": "Inspecting evidence…",
                        "stage": "inspect", "status": "running", "activity": "Reading campaign evidence and resource limits.",
                        "stages": [{"name": name, "status": "active" if name == "inspect" else "waiting"} for name in STAGES],
                        "acceptance": f"Any validation primary gain is promoted; convergence sensitivity {limits['convergence_epsilon']:.6f}",
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
                        "prior_iterations": snapshot["iterations"][-12:],
                        "epsilon": limits["convergence_epsilon"],
                        "remaining_iterations": end_number - number,
                        "official_iterations_used": self._budget_iterations(snapshot),
                        "official_iterations_limit": OFFICIAL_MAX_ITERATIONS,
                        "resource_usage": snapshot["usage"],
                        "steering": snapshot["campaign"].get("steering"),
                        "constraints": [
                            "GAUC and nDCG@5 validation",
                            "no hidden-test access",
                            "one isolated change",
                            "must be reproducible",
                            "do not repeat an exact experiment_type and parameter configuration already listed",
                            "prefer validation gain per CPU/GPU hour when expected gains are similar",
                        ],
                        "executor_contract": {
                            "runtime": "Trusted NumPy FM/BPR executors plus an installed PyTorch DeepFM executor; no package installation or arbitrary generated-code execution",
                            "experiment_types": {
                                "fm_config": "One FM with typed k/lr/epochs/batch_size/patience/seed parameters",
                                "fm_positive_weight": "FM logistic loss with the supplied positive_weight in [1,10]",
                                "fm_ensemble": "Mean validation logits from 1-3 independently trained FM seeds",
                                "fm_pairwise": "One FM trained with within-user BPR pairs sampled from logged impressions",
                                "fm_pairwise_blend": "Blend a 1-3 seed weighted-FM ensemble with one independently trained BPR FM",
                                "fm_deep_blend": "Blend weighted FM, BPR FM, and a nonlinear DeepFM trained with weighted BCE",
                                "fm_temporal_deep_blend": "Add a small globally standardized clock-context FM to the weighted FM, BPR, and DeepFM blend"
                            },
                            "defaults": {
                                "k": 16, "lr": 0.001, "epochs": 40, "batch_size": 8192, "patience": 4,
                                "seed": 0, "ensemble_seeds": [0, 1, 2], "positive_weight": 1.0,
                                "pairwise_lr": 0.002, "pairwise_epochs": 12, "pairwise_patience": 4,
                                "pairwise_seed": 0, "blend_weight": 0.455,
                                "deep_lr": 0.001, "deep_epochs": 15, "deep_patience": 4,
                                "deep_seed": 0, "deep_hidden": 64, "deep_dropout": 0.05,
                                "deep_threads": 6, "deep_blend_weight": 0.23, "temporal_blend_weight": 0.024,
                            },
                            "rule": "Select exactly one supported experiment_type and populate every typed parameter. Generated code is evidence; the trusted executor applies the typed change."
                        },
                    }
                    self._set_stage(number, "hypothesize", f"Asking {self.settings.model if provider_name == 'gpt' else 'deterministic demo planner'} for one falsifiable experiment.")
                    proposal = self._propose_with_retry(provider, context, number)

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
                    proposal, evaluation, recovery_events = self._evaluate_with_recovery(
                        benchmark, proposal, number, workspace
                    )
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
                        "experiment_type": proposal.experiment_type, "parameters": proposal.parameters,
                        "resource_usage": evaluation.resource_usage,
                        "recovery_events": recovery_events, "budget_counted": True,
                        "code_diff": "candidate.diff" if not recovery_events else "retry-1/candidate.diff",
                    }
                    self._write_iteration_log(workspace, record)

                    def finish_iteration(state: dict) -> None:
                        state["iterations"].append(record)
                        if accepted:
                            state["metrics"]["champion"] = metrics
                            state["metrics"]["delta"] = baseline_delta
                        add_resource_usage(state["usage"], evaluation.resource_usage)
                        state["current"]["status"] = "complete"
                        for item in state["current"]["stages"]:
                            item["status"] = "done"
                        state["usage"]["wall_seconds"] = round(wall_before + time.monotonic() - started, 2)
                    self.store.update(finish_iteration)
                    action = "Champion promoted" if accepted else "Candidate rejected"
                    usage = evaluation.resource_usage
                    self.store.event(
                        "result",
                        action,
                        f"Primary {metrics['primary']:.4f} · gain {gain:+.4f} · train {usage['train_seconds']:.1f}s · CPU {usage['cpu_hours']:.4f}h · GPU {usage['gpu_hours']:.4f}h",
                        number,
                        "reflect",
                    )
                    small_gain_streak = small_gain_streak + 1 if gain <= limits["convergence_epsilon"] else 0
                    self.store.update(
                        lambda state: state["campaign"].update(consecutive_small_gains=small_gain_streak)
                    )
                    if limits["convergence_patience"] and small_gain_streak >= limits["convergence_patience"]:
                        stop_reason = f"converged: {small_gain_streak} consecutive gains below {limits['convergence_epsilon']:.6f}"
                        break
                except Exception as error:
                    duration = round(time.monotonic() - iteration_started, 2)
                    resource_usage = getattr(error, "resource_usage", None)
                    failed = {
                        "number": number, "title": self.store.snapshot().get("current", {}).get("title", "Experiment"),
                        "status": "failed", "stage": "failed", "metrics": None, "delta": None, "accepted": False,
                        "duration_seconds": duration, "provider": provider_name, "error": str(error)[:800], "artifact": str(workspace),
                        "resource_usage": resource_usage,
                        "gain": None, "recovery_events": getattr(error, "recovery_events", []), "budget_counted": True,
                        "code_diff": "candidate.diff",
                    }
                    self._write_iteration_log(workspace, failed)
                    def record_failure(state: dict) -> None:
                        state["iterations"].append(failed)
                        state["campaign"]["failure_count"] = int(state["campaign"].get("failure_count", 0)) + 1
                        if state["current"]:
                            state["current"].update(status="failed", error=failed["error"])
                        add_resource_usage(state["usage"], resource_usage)
                        state["usage"]["wall_seconds"] = round(wall_before + time.monotonic() - started, 2)
                    self.store.update(record_failure)
                    self.store.event("error", "Iteration failed · recovered", failed["error"], number, "failed")
                    small_gain_streak += 1
                    self.store.update(
                        lambda state: state["campaign"].update(consecutive_small_gains=small_gain_streak)
                    )
                    if limits["convergence_patience"] and small_gain_streak >= limits["convergence_patience"]:
                        stop_reason = f"converged: {small_gain_streak} consecutive iterations without > {limits['convergence_epsilon']:.6f} gain"
                        break
                    if provider_name == "gpt" and "OpenAI Responses API failed" in str(error):
                        stop_reason = "model provider error"
                        break
                    continue
        except Exception as error:
            stop_reason = f"campaign error: {str(error)[:300]}"
            resource_usage = getattr(error, "resource_usage", None)
            if resource_usage:
                def record_campaign_failure(state: dict) -> None:
                    add_resource_usage(state["usage"], resource_usage)
                    if mode == "kuairand" and state["iterations"] and state["iterations"][0]["stage"] != "complete":
                        state["iterations"][0].update(status="failed", stage="failed", error=str(error)[:800], resource_usage=resource_usage)
                self.store.update(record_campaign_failure)
            self.store.event("error", "Campaign stopped unexpectedly", stop_reason)
        finally:
            final_status = "stopped" if self._stop.is_set() else "complete"
            def complete(state: dict) -> None:
                state["campaign"].update(status=final_status, ended_at=utc_now(), stop_reason=stop_reason)
                state["usage"]["wall_seconds"] = round(wall_before + time.monotonic() - started, 2)
            self.store.update(complete)
            self.store.event("campaign", "Campaign finished", stop_reason)
            final = self.store.snapshot()
            summary_path = self.settings.state_dir / "campaigns" / campaign_id / "resource-summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps({
                "campaign": final["campaign"],
                "usage": final["usage"],
                "official_budget": {
                    "iterations_used": self._budget_iterations(final),
                    "iterations_limit": OFFICIAL_MAX_ITERATIONS,
                    "wall_hours_used": round(final["usage"]["wall_seconds"] / 3600, 6),
                    "wall_hours_limit": OFFICIAL_MAX_HOURS,
                    "convergence_epsilon": limits["convergence_epsilon"],
                    "convergence_patience": limits["convergence_patience"],
                },
                "manual_interventions": int(final["campaign"].get("manual_interventions", 0)),
                "failure_count": int(final["campaign"].get("failure_count", 0)),
                "recovery_count": int(final["campaign"].get("recovery_count", 0)),
                "iterations": [
                    {
                        "number": item["number"],
                        "status": item["status"],
                        "primary": item.get("metrics", {}).get("primary") if item.get("metrics") else None,
                        "gain": item.get("gain"),
                        "resource_usage": item.get("resource_usage"),
                    }
                    for item in final["iterations"]
                ],
            }, indent=2, sort_keys=True), encoding="utf-8")
