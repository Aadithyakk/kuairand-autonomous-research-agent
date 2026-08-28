from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .fidelity import MultiFidelityPolicy


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokens(value: str) -> set[str]:
    return {part for part in re.findall(r"[a-z0-9_]+", value.lower()) if len(part) > 2}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def detect_compute_profiles(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return selectable compute without persisting credentials or running probes."""

    cpu_count = os.cpu_count() or 1
    machine = platform.machine() or "unknown"
    profiles = [{
        "id": "local-cpu",
        "label": "Local CPU",
        "provider": "local",
        "accelerator": f"{cpu_count} logical cores · {machine}",
        "available": True,
        "detected": True,
        "recommended": config.get("executor_mode") not in {"kaggle", "kaggle_autonomous"},
        "reason": "Available for baselines, feature generation, and tree models.",
    }]
    if shutil.which("nvidia-smi") is not None:
        profiles.append({
            "id": "local-cuda", "label": "Local NVIDIA GPU", "provider": "local",
            "accelerator": "CUDA GPU", "available": True, "detected": True,
            "recommended": True, "reason": "Detected through the NVIDIA runtime.",
        })
    elif sys.platform == "darwin" and machine == "arm64":
        torch_available = importlib.util.find_spec("torch") is not None
        profiles.append({
            "id": "local-metal", "label": "Apple Silicon GPU", "provider": "local",
            "accelerator": "Metal / MPS", "available": torch_available, "detected": True,
            "recommended": False,
            "reason": "MPS-capable runtime detected." if torch_available else "Install an MPS-enabled PyTorch environment to use this card.",
        })

    has_kaggle_token = bool(os.getenv("KAGGLE_API_TOKEN"))
    for configured in config.get("compute_profiles", []):
        profile = copy.deepcopy(configured)
        profile.setdefault("provider", "kaggle")
        profile.setdefault("detected", has_kaggle_token)
        profile["available"] = bool(profile.get("enabled", True) and has_kaggle_token and config.get("executor_mode") in {"kaggle", "kaggle_autonomous"})
        profile["recommended"] = bool(profile["available"] and profile.get("recommended", False))
        if not has_kaggle_token:
            profile["reason"] = "Kaggle token is not present in the agent process."
        elif config.get("executor_mode") not in {"kaggle", "kaggle_autonomous"}:
            profile["reason"] = "Kaggle is authenticated, but this server is not using the Kaggle dispatcher."
        profiles.append(profile)
    return profiles


class StateStore:
    """Thread-safe state plus append-only audit events."""

    def __init__(self, state_path: Path, events_path: Path):
        self.state_path = state_path
        self.events_path = events_path
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            return read_json(self.state_path) if self.state_path.exists() else {}

    def save(self, state: dict[str, Any]) -> None:
        with self._lock:
            state["updated_at"] = utc_now()
            write_json_atomic(self.state_path, state)

    def mutate(self, callback: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            callback(state)
            self.save(state)
            return copy.deepcopy(state)

    def event(self, kind: str, message: str, **payload: Any) -> dict[str, Any]:
        event = {"id": uuid.uuid4().hex[:12], "timestamp": utc_now(), "kind": kind, "message": message, **payload}
        with self._lock:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

            def append(state: dict[str, Any]) -> None:
                state.setdefault("events", []).append(event)
                state["events"] = state["events"][-250:]

            self.mutate(append)
        return event


class LiteratureIndex:
    def __init__(self, cards: list[dict[str, Any]]):
        self.cards = cards

    @classmethod
    def from_path(cls, path: Path) -> "LiteratureIndex":
        return cls(read_json(path))

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_tokens = tokens(query)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for card in self.cards:
            tags = set(card.get("tags", []))
            text = tokens(" ".join([card.get("title", ""), card.get("claim", ""), *tags]))
            overlap = len(query_tokens & text)
            tag_overlap = len(query_tokens & tags)
            score = overlap + 1.7 * tag_overlap
            if score > 0:
                ranked.append((score, card))
        ranked.sort(key=lambda item: (-item[0], -item[1].get("year", 0), item[1]["id"]))
        return [copy.deepcopy(card) for _, card in ranked[:limit]]


class LLMClient:
    """JSON research client for OpenAI Responses or a compatible local endpoint."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.get("mode") in {"openai_compatible", "openai_responses"}

    def complete_json(self, system: str, user: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self.last_error = None
        if self.config.get("mode") == "openai_responses":
            return self._complete_responses(system, user)
        return self._complete_compatible(system, user)

    def _complete_compatible(self, system: str, user: str) -> dict[str, Any] | None:
        base = self.config["base_url"].rstrip("/")
        key = os.getenv(self.config.get("api_key_env", "RESEARCH_AGENT_API_KEY"), "local")
        body = {
            "model": self.config["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.config.get("temperature", 0.3),
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.get("timeout_seconds", 90)) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            return json.loads(content)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def _complete_responses(self, system: str, user: str) -> dict[str, Any] | None:
        key_env = self.config.get("api_key_env", "OPENAI_API_KEY")
        key = os.getenv(key_env)
        if not key:
            self.last_error = f"Missing API key environment variable: {key_env}"
            return None
        base = self.config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        models = [self.config["model"]]
        fallback = self.config.get("fallback_model")
        if fallback and fallback not in models:
            models.append(fallback)
        for model in models:
            body = {
                "model": model,
                "instructions": system,
                "input": "Return JSON for this payload:\n" + user,
                "reasoning": {"effort": self.config.get("reasoning_effort", "medium")},
                "max_output_tokens": int(self.config.get("max_output_tokens", 8000)),
                "text": {"format": {"type": "json_object"}, "verbosity": "low"},
                "store": False,
            }
            request = urllib.request.Request(
                f"{base}/responses",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.get("timeout_seconds", 240)) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = "".join(
                    part.get("text", "")
                    for item in result.get("output", [])
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                ).strip()
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("Responses API output was not a JSON object")
                return parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[-1200:]
                self.last_error = f"{model}: HTTP {exc.code}: {detail}"
            except Exception as exc:
                self.last_error = f"{model}: {type(exc).__name__}: {exc}"
        return None


class ResearchPolicy:
    """Cost-aware experiment selection. The LLM advises; deterministic policy enforces constraints."""

    def __init__(self, actions: list[dict[str, Any]], literature: LiteratureIndex, llm: LLMClient, seed: int = 2026):
        self.actions = actions
        self.literature = literature
        self.llm = llm
        self.random = random.Random(seed)

    def candidates(self, state: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
        completed_ids = {item.get("action_id") for item in state.get("experiments", []) if item.get("status") in {"completed", "failed"}}
        successful_families = {item.get("family") for item in state.get("experiments", []) if item.get("improved")}
        remaining_seconds = max(0, state["run"]["budget_seconds"] - state["run"].get("elapsed_seconds", 0))
        latest_steer = " ".join(item.get("message", "") for item in state.get("steering", [])[-3:]).lower()
        scored: list[tuple[float, dict[str, Any]]] = []

        for action in self.actions:
            if action["id"] in completed_ids:
                continue
            if state["run"].get("executor_mode") == "command" and not action.get("command"):
                continue
            if action["id"] == "rank_blend_champions" and len(successful_families) < 2:
                continue
            cost_seconds = action["estimated_minutes"] * 60
            if cost_seconds > remaining_seconds:
                continue
            evidence = self.literature.search(" ".join(action.get("evidence_tags", [])), limit=3)
            novelty = 1.0 if action.get("family") not in {item.get("family") for item in state.get("experiments", [])} else 0.25
            cost_fit = 1.0 - min(1.0, cost_seconds / max(1, remaining_seconds))
            steering_boost = 0.0
            action_text = " ".join([action["title"], action["family"], *action.get("when", [])]).lower()
            if latest_steer and tokens(latest_steer) & tokens(action_text):
                steering_boost = 0.3
            score = 25 * action["prior_gain"] + 0.24 * novelty + 0.18 * cost_fit + 0.08 * len(evidence) - 0.28 * action["risk"] + steering_boost
            candidate = copy.deepcopy(action)
            candidate["policy_score"] = round(score, 5)
            candidate["evidence"] = evidence
            candidate["reason"] = self._fallback_reason(candidate, state)
            scored.append((score, candidate))

        scored.sort(key=lambda item: (-item[0], item[1]["estimated_minutes"], item[1]["id"]))
        shortlisted = [candidate for _, candidate in scored[:limit]]
        advice = self._llm_advice(state, shortlisted)
        if advice:
            preferred = advice.get("selected_action_id")
            for candidate in shortlisted:
                if candidate["id"] == preferred:
                    candidate["policy_score"] += 0.12
                    candidate["reason"] = advice.get("reason", candidate["reason"])
                    candidate["llm_advice"] = advice
            shortlisted.sort(key=lambda item: -item["policy_score"])
        return shortlisted

    def _fallback_reason(self, action: dict[str, Any], state: dict[str, Any]) -> str:
        best = state.get("best", {}).get("metrics", {}).get("primary", 0.0)
        evidence_titles = ", ".join(card["title"] for card in action.get("evidence", [])[:2])
        return (
            f"Test {action['hypothesis']} Current champion primary is {best:.4f}; "
            f"estimated cost is {action['estimated_minutes']} minutes. Evidence: {evidence_titles or 'organizer research directions'}."
        )

    def _llm_advice(self, state: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        safe_state = {
            "best": state.get("best"),
            "recent_experiments": state.get("experiments", [])[-4:],
            "steering": state.get("steering", [])[-3:],
            "remaining_seconds": state["run"]["budget_seconds"] - state["run"].get("elapsed_seconds", 0),
        }
        safe_candidates = [
            {key: item[key] for key in ["id", "title", "family", "hypothesis", "estimated_minutes", "risk", "prior_gain"]}
            for item in candidates
        ]
        return self.llm.complete_json(
            "You are the planner of an autonomous recommender research agent. Choose only from the supplied action IDs. Return JSON with selected_action_id, reason, expected_observation, and abort_condition. Never request hidden-test access.",
            json.dumps({"state": safe_state, "candidates": safe_candidates}),
        )


class ExperimentExecutor:
    """Runs configured commands or a deterministic simulator for product validation."""

    def __init__(self, mode: str, workspace: Path, seed: int = 2026):
        self.mode = mode
        self.workspace = workspace
        self.random = random.Random(seed)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, experiment: dict[str, Any], baseline: dict[str, float], timeout_seconds: int) -> dict[str, Any]:
        started = time.monotonic()
        if self.mode == "simulation":
            result = self._simulate(experiment, baseline)
        else:
            result = self._command(experiment, timeout_seconds)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return result

    def _simulate(self, experiment: dict[str, Any], baseline: dict[str, float]) -> dict[str, Any]:
        time.sleep(0.18)
        gain = max(-0.004, self.random.gauss(experiment["prior_gain"], 0.004 + experiment["risk"] * 0.004))
        ndcg_share = 0.58 if experiment["family"] in {"tree_ranker", "pairwise_cf", "sequence"} else 0.48
        primary = max(0.0, min(1.0, baseline["primary"] + gain))
        gauc = max(0.0, min(1.0, baseline["GAUC"] + gain * (2 - 2 * ndcg_share)))
        ndcg = max(0.0, min(1.0, baseline["nDCG@5"] + gain * (2 * ndcg_share)))
        return {
            "status": "completed",
            "metrics": {"GAUC": round(gauc, 6), "nDCG@5": round(ndcg, 6), "primary": round(primary, 6)},
            "artifacts": [],
            "notes": "Simulated result used to validate orchestration and dashboard wiring; not a competition result.",
        }

    def _command(self, experiment: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        command = experiment.get("command")
        if not command:
            return {"status": "failed", "error": "No command adapter is configured for this action.", "metrics": {}}
        exp_dir = self.workspace / experiment["experiment_id"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        spec_path = exp_dir / "spec.json"
        result_path = exp_dir / "result.json"
        write_json_atomic(spec_path, experiment)
        rendered = command.format(spec=str(spec_path), result=str(result_path), workspace=str(exp_dir))
        try:
            environment = os.environ.copy()
            project_root = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = project_root + os.pathsep + environment.get("PYTHONPATH", "")
            process = subprocess.run(
                shlex.split(rendered), cwd=exp_dir, env=environment, capture_output=True, text=True, timeout=timeout_seconds, check=False
            )
            (exp_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
            (exp_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
            if process.returncode != 0:
                return {"status": "failed", "error": f"Worker exited with code {process.returncode}", "metrics": {}, "artifacts": [str(exp_dir)]}
            if not result_path.exists():
                return {"status": "failed", "error": "Worker did not create result.json", "metrics": {}, "artifacts": [str(exp_dir)]}
            result = read_json(result_path)
            result.setdefault("artifacts", []).append(str(exp_dir))
            return result
        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": f"Experiment exceeded {timeout_seconds}s timeout", "metrics": {}, "artifacts": [str(exp_dir)]}


def baseline_from_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        match = next((name for name in names if name.endswith("iteration_000_baseline.json")), None)
        if not match:
            raise ValueError("Baseline ZIP does not contain iteration_000_baseline.json")
        return json.loads(archive.read(match).decode("utf-8"))


class ResearchController:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.config = config
        paths = config["paths"]
        self.store = StateStore(root / paths["state"], root / paths["events"])
        self.literature = LiteratureIndex.from_path(root / paths["literature"])
        self.actions = read_json(root / paths["actions"])
        self.llm = LLMClient(config["llm"])
        self.fidelity = MultiFidelityPolicy(config.get("search"))
        self.policy = ResearchPolicy(self.actions, self.literature, self.llm, config.get("random_seed", 2026))
        self.executor = ExperimentExecutor(config["executor_mode"], root / paths["workspace"], config.get("random_seed", 2026))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def compute_profiles(self) -> list[dict[str, Any]]:
        return detect_compute_profiles(self.config)

    def initialize(self, force: bool = False) -> dict[str, Any]:
        existing = self.store.load()
        if existing and not force:
            return existing
        configured_artifact = os.getenv("KUAI_BASELINE_ARTIFACT", self.config["paths"]["baseline_artifact"])
        artifact_path = Path(configured_artifact)
        if not artifact_path.is_absolute():
            artifact_path = self.root / artifact_path
        artifact = baseline_from_zip(artifact_path)
        validation = artifact["validation_mean"]
        baseline_metrics = {key: float(validation[key]) for key in ["GAUC", "nDCG@5", "primary"]}
        state = {
            "schema_version": 1,
            "run": {
                "id": f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "status": "ready",
                "benchmark": self.config["benchmark"],
                "label": self.config["label"],
                "metrics": self.config["metrics"],
                "budget_seconds": self.config["budget_seconds"],
                "elapsed_seconds": 0,
                "max_experiments": self.config["max_experiments"],
                "started_at": None,
                "manual_interventions": 0,
                "executor_mode": self.config["executor_mode"],
                "llm_mode": self.config["llm"]["mode"],
                "compute_profile_id": "local-cpu",
                "search": self.fidelity.public_summary(),
                "implementation_attempts": 0,
            },
            "baseline": {"metrics": baseline_metrics, "artifact": str(artifact_path), "status": artifact.get("status", "unknown")},
            "best": {"experiment_id": "iteration-000", "title": "Official FM baseline", "metrics": baseline_metrics},
            "current_experiment": None,
            "candidate_queue": [],
            "experiments": [],
            "implementation_attempts": [],
            "decisions": [],
            "steering": [],
            "literature_hits": [],
            "events": [],
            "created_at": utc_now(),
        }
        self.store.save(state)
        self.store.event("baseline", "Official FM baseline loaded and marked immutable.", metrics=baseline_metrics)
        self.store.event("system", "Research agent is ready.", mode=self.config["executor_mode"])
        return self.store.load()

    def start(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self.initialize()
        if self._thread and self._thread.is_alive():
            return state
        options = options or {}
        budget_minutes = float(options.get("budget_minutes", state["run"]["budget_seconds"] / 60))
        if not 1 <= budget_minutes <= 720:
            raise ValueError("Time budget must be between 1 minute and 12 hours")
        requested = str(options.get("compute_profile_id", state["run"].get("compute_profile_id", "local-cpu")))
        selected_profile = next((profile for profile in self.compute_profiles() if profile["id"] == requested), None)
        if selected_profile is None:
            raise ValueError(f"Unknown compute profile: {requested}")
        if not selected_profile["available"]:
            raise ValueError(selected_profile.get("reason") or f"Compute profile is unavailable: {requested}")

        def configure(current: dict[str, Any]) -> None:
            current["run"]["budget_seconds"] = int(budget_minutes * 60)
            current["run"]["compute_profile_id"] = requested
            current["run"]["compute"] = selected_profile

        self.store.mutate(configure)
        self.store.event("control", "Run configuration accepted.", budget_minutes=budget_minutes, compute_profile_id=requested)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="research-controller", daemon=True)
        self._thread.start()
        return self.store.load()

    def pause(self) -> dict[str, Any]:
        return self.store.mutate(lambda state: state["run"].update(status="paused"))

    def resume(self) -> dict[str, Any]:
        state = self.store.mutate(lambda current: current["run"].update(status="running"))
        if not self._thread or not self._thread.is_alive():
            self.start()
        return state

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self.store.event("control", "Run stopped by a human operator.")
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        return self.store.mutate(lambda current: current["run"].update(status="stopped"))

    def steer(self, message: str) -> dict[str, Any]:
        clean = " ".join(message.strip().split())[:1000]
        if not clean:
            raise ValueError("Steering message cannot be empty")
        item = {"id": uuid.uuid4().hex[:10], "timestamp": utc_now(), "message": clean, "status": "pending"}

        def add(state: dict[str, Any]) -> None:
            state.setdefault("steering", []).append(item)
            state["run"]["manual_interventions"] = state["run"].get("manual_interventions", 0) + 1

        state = self.store.mutate(add)
        self.store.event("steering", "Human guidance received.", steering_id=item["id"], message_text=clean)
        return state

    def _loop(self) -> None:
        started = time.monotonic()

        def mark_running(state: dict[str, Any]) -> None:
            state["run"]["status"] = "running"
            state["run"]["started_at"] = state["run"].get("started_at") or utc_now()

        self.store.mutate(mark_running)
        self.store.event("control", "Autonomous experiment loop started.")
        if self.config["executor_mode"] == "kaggle_autonomous":
            self._real_campaign_loop(started)
            return
        try:
            while not self._stop.is_set():
                state = self.store.load()
                status = state["run"]["status"]
                if status == "paused":
                    time.sleep(0.2)
                    continue
                if status in {"stopped", "completed", "converged", "budget_exhausted"}:
                    return
                elapsed = int(time.monotonic() - started)
                state["run"]["elapsed_seconds"] = elapsed
                self.store.save(state)
                budget_seconds = int(state["run"]["budget_seconds"])
                if elapsed >= budget_seconds:
                    self._finish("budget_exhausted", "Compute budget exhausted; champion retained.")
                    return
                if len(state["experiments"]) >= self.config["max_experiments"]:
                    self._finish("completed", "Maximum experiment count reached; champion designated final.")
                    return
                if self._converged(state):
                    self._finish("converged", "Convergence rule satisfied; champion designated final.")
                    return
                candidates = self.policy.candidates(state)
                if not candidates:
                    self._finish("completed", "No valid experiment fits the remaining constraints.")
                    return
                selected = candidates[0]
                experiment_id = f"iteration-{len(state['experiments']) + 1:03d}"
                selected.update({"experiment_id": experiment_id, "status": "running", "started_at": utc_now()})
                decision = {
                    "id": uuid.uuid4().hex[:10], "timestamp": utc_now(), "selected_action_id": selected["id"],
                    "reason": selected["reason"], "candidate_scores": [{"id": item["id"], "score": item["policy_score"]} for item in candidates],
                }

                def set_current(current: dict[str, Any]) -> None:
                    current["candidate_queue"] = candidates
                    current["current_experiment"] = selected
                    current["decisions"].append(decision)
                    current["literature_hits"] = selected.get("evidence", [])
                    for steer in current.get("steering", []):
                        if steer["status"] == "pending":
                            steer["status"] = "considered"

                self.store.mutate(set_current)
                self.store.event("decision", f"Selected {selected['title']}.", experiment_id=experiment_id, reason=selected["reason"])
                best_metrics = self.store.load()["best"]["metrics"]
                selected["compute_profile_id"] = state["run"].get("compute_profile_id", "local-cpu")
                timeout_seconds = min(int(selected["estimated_minutes"] * 90), max(30, budget_seconds - elapsed))
                result = self.executor.run(selected, best_metrics, timeout_seconds)
                self._record_result(selected, result)
        except Exception as exc:
            self.store.mutate(lambda state: state["run"].update(status="error", error=str(exc)))
            self.store.event("error", "Controller encountered an unrecovered error.", error=str(exc))

    def _real_campaign_loop(self, started: float) -> None:
        from .campaign import AutonomousKaggleCampaign

        campaign = AutonomousKaggleCampaign(self.root, self.config)
        try:
            while not self._stop.is_set():
                state = self.store.load()
                if state["run"].get("status") == "paused":
                    time.sleep(1.0)
                    continue
                elapsed = int(time.monotonic() - started)
                state["run"]["elapsed_seconds"] = elapsed
                self.store.save(state)
                if elapsed >= int(state["run"]["budget_seconds"]):
                    self._finish("budget_exhausted", "Research budget exhausted; champion retained.")
                    return
                if len(state["experiments"]) >= self.config["max_experiments"]:
                    self._finish("completed", "Maximum autonomous experiment count reached.")
                    return

                sequence = len(state["experiments"]) + 1

                def report_stage(stage_name: str, message: str, payload: dict[str, Any] | None) -> None:
                    payload = payload or {}

                    def update(current: dict[str, Any]) -> None:
                        proposal = payload.get("proposal")
                        if proposal:
                            current["current_experiment"] = copy.deepcopy(proposal)
                        elif not current.get("current_experiment"):
                            current["current_experiment"] = {
                                "id": f"campaign-{sequence:03d}",
                                "experiment_id": f"iteration-{sequence:03d}",
                                "title": "Autonomous research iteration",
                                "family": "model-authored research",
                                "hypothesis": "The planner is diagnosing prior results and selecting the next bounded experiment.",
                                "status": "running",
                                "started_at": utc_now(),
                            }
                        current["run"]["executor_mode"] = self.config["executor_mode"]
                        current["run"]["llm_mode"] = self.config["llm"]["mode"]
                        current["run"]["llm_model"] = self.config["llm"]["model"]
                        current["run"]["search"] = self.fidelity.public_summary()
                        current["run"]["worker"] = {
                            "stage": stage_name,
                            "message": message,
                            "heartbeat_at": utc_now(),
                            "kernel_ref": payload.get("kernel_ref"),
                        }
                        if current.get("current_experiment"):
                            current["current_experiment"]["stage"] = stage_name
                            current["current_experiment"]["status"] = "running"

                    self.store.mutate(update)
                    self.store.event("worker", message, stage=stage_name, kernel_ref=payload.get("kernel_ref"))

                try:
                    proposal, result = campaign.run_iteration(
                        sequence,
                        state["run"].get("compute_profile_id", "kaggle-cpu"),
                        report_stage,
                        self._stop.is_set,
                    )
                except Exception as exc:
                    proposal = {
                        "id": f"campaign-{sequence:03d}",
                        "experiment_id": f"iteration-{sequence:03d}",
                        "title": "Autonomous research iteration",
                        "hypothesis": "The iteration failed before a proposal could be executed.",
                        "family": "autonomous",
                        "status": "failed",
                    }
                    result = {"status": "failed", "stage": "controller", "metrics": {}, "error": str(exc)}
                if result.get("counts_as_experiment", result.get("status") == "completed"):
                    self._record_result(proposal, result)
                else:
                    self._record_implementation_attempt(proposal, result)
                    attempt_count = self.store.load()["run"].get("implementation_attempts", 0)
                    if attempt_count >= int(self.config.get("campaign", {}).get("max_implementation_failures", 12)):
                        self._finish("error", "Implementation failure budget exhausted before enough valid experiments completed.")
                        return
                if self._stop.is_set():
                    self._finish("stopped", "Run stopped after the active remote experiment boundary.")
                    return
                if self._converged(self.store.load()):
                    self._finish("converged", "Convergence rule satisfied; champion designated final.")
                    return
        except Exception as exc:
            self.store.mutate(lambda state: state["run"].update(status="error", error=str(exc)))
            self.store.event("error", "Real campaign encountered an unrecovered error.", error=str(exc))

    def _record_implementation_attempt(self, selected: dict[str, Any], result: dict[str, Any]) -> None:
        attempt = {
            "attempt_id": uuid.uuid4().hex[:10],
            "timestamp": utc_now(),
            "experiment_id": selected.get("experiment_id"),
            "candidate_id": selected.get("id"),
            "title": selected.get("title", "Autonomous candidate"),
            "hypothesis": selected.get("hypothesis"),
            "failure_type": result.get("failure_type", "implementation_failed"),
            "stage": result.get("stage"),
            "error": result.get("error"),
            "details": result.get("implementation_attempts", selected.get("implementation_attempts", [])),
        }

        def record(state: dict[str, Any]) -> None:
            state.setdefault("implementation_attempts", []).append(attempt)
            state["run"]["implementation_attempts"] = state["run"].get("implementation_attempts", 0) + 1
            state["current_experiment"] = None

        self.store.mutate(record)
        self.store.event(
            "repair",
            f"Implementation attempt failed; scientific iteration {selected.get('experiment_id', '')} will be retried.",
            failure_type=attempt["failure_type"], stage=attempt["stage"], error=attempt["error"],
        )

    def _record_result(self, selected: dict[str, Any], result: dict[str, Any]) -> None:
        completed = copy.deepcopy(selected)
        completed.update(result)
        completed["completed_at"] = utc_now()
        completed["action_id"] = completed.pop("id")
        completed["title"] = selected["title"]

        def record(state: dict[str, Any]) -> None:
            champion = state["best"]["metrics"]["primary"]
            score = completed.get("metrics", {}).get("primary", -math.inf)
            external = completed.get("external_validated", completed.get("status") == "completed")
            completed["delta_vs_champion"] = round(score - champion, 6) if external and math.isfinite(score) else None
            completed["improved"] = external and completed["status"] == "completed" and score > champion
            state["experiments"].append(completed)
            state["current_experiment"] = None
            if completed["improved"]:
                state["best"] = {"experiment_id": completed["experiment_id"], "title": completed["title"], "metrics": completed["metrics"]}

        state = self.store.mutate(record)
        if completed["status"] == "completed":
            self.store.event(
                "result", f"{completed['title']} completed.", experiment_id=completed["experiment_id"],
                metrics=completed["metrics"], improved=completed["improved"], delta=completed["delta_vs_champion"],
            )
        else:
            self.store.event("recovery", f"{completed['title']} failed; the search will route around it.", experiment_id=completed["experiment_id"], error=completed.get("error"))

    def _converged(self, state: dict[str, Any]) -> bool:
        patience = self.config["convergence_patience"]
        completed = [item for item in state["experiments"] if item.get("status") == "completed"]
        minimum = int(self.config.get("convergence_min_valid_experiments", patience))
        if len(completed) < max(patience, minimum):
            return False
        required = set(self.config.get("convergence_required_families", []))
        covered = {self._family_bucket(item) for item in completed}
        if required and not required.issubset(covered):
            return False
        return all((item.get("delta_vs_champion") or 0) <= self.config["convergence_epsilon"] for item in completed[-patience:])

    @staticmethod
    def _family_bucket(item: dict[str, Any]) -> str:
        text = " ".join(str(item.get(key, "")) for key in ("model_family", "family", "title", "change_kind")).lower()
        if any(term in text for term in ("hybrid", "blend", "ensemble", "residual")):
            return "hybrid"
        if any(term in text for term in ("sequence", "sequential", "din", "dien", "transformer", "sasrec")):
            return "sequence"
        if any(term in text for term in ("collaborative", "bpr", "matrix", "graph", "cf")):
            return "collaborative"
        if any(term in text for term in ("tree", "lightgbm", "lambdarank", "catboost", "boost")):
            return "tree"
        if any(term in text for term in ("factorization", " fm", "fm ")):
            return "factorization"
        return "other"

    def _finish(self, status: str, message: str) -> None:
        self.store.event("complete", message, status=status)
        # Publish the terminal status last so observers know no further audit
        # writes from this loop remain pending.
        self.store.mutate(lambda state: state["run"].update(
            status=status, completed_at=utc_now(),
            worker={"stage": status, "message": message, "heartbeat_at": utc_now(), "kernel_ref": None},
        ))
