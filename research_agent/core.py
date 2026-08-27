from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import shlex
import subprocess
import threading
import time
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
    """OpenAI-compatible local LLM client with a deterministic offline fallback."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.get("mode") == "openai_compatible"

    def complete_json(self, system: str, user: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
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
        except Exception:
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
        self.policy = ResearchPolicy(self.actions, self.literature, self.llm, config.get("random_seed", 2026))
        self.executor = ExperimentExecutor(config["executor_mode"], root / paths["workspace"], config.get("random_seed", 2026))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

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
            },
            "baseline": {"metrics": baseline_metrics, "artifact": str(artifact_path), "status": artifact.get("status", "unknown")},
            "best": {"experiment_id": "iteration-000", "title": "Official FM baseline", "metrics": baseline_metrics},
            "current_experiment": None,
            "candidate_queue": [],
            "experiments": [],
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

    def start(self) -> dict[str, Any]:
        state = self.initialize()
        if self._thread and self._thread.is_alive():
            return state
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
        state = self.store.mutate(lambda current: current["run"].update(status="stopped"))
        self.store.event("control", "Run stopped by a human operator.")
        return state

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
                if elapsed >= self.config["budget_seconds"]:
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
                timeout_seconds = min(int(selected["estimated_minutes"] * 90), max(30, self.config["budget_seconds"] - elapsed))
                result = self.executor.run(selected, best_metrics, timeout_seconds)
                self._record_result(selected, result)
        except Exception as exc:
            self.store.mutate(lambda state: state["run"].update(status="error", error=str(exc)))
            self.store.event("error", "Controller encountered an unrecovered error.", error=str(exc))

    def _record_result(self, selected: dict[str, Any], result: dict[str, Any]) -> None:
        completed = copy.deepcopy(selected)
        completed.update(result)
        completed["completed_at"] = utc_now()
        completed["action_id"] = completed.pop("id")
        completed["title"] = selected["title"]

        def record(state: dict[str, Any]) -> None:
            champion = state["best"]["metrics"]["primary"]
            score = completed.get("metrics", {}).get("primary", -math.inf)
            completed["delta_vs_champion"] = round(score - champion, 6) if math.isfinite(score) else None
            completed["improved"] = completed["status"] == "completed" and score > champion
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
        if len(completed) < patience:
            return False
        return all((item.get("delta_vs_champion") or 0) <= self.config["convergence_epsilon"] for item in completed[-patience:])

    def _finish(self, status: str, message: str) -> None:
        self.store.mutate(lambda state: state["run"].update(status=status, completed_at=utc_now()))
        self.store.event("complete", message, status=status)
