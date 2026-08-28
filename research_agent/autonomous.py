from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .benchmark import Benchmark
from .core import LLMClient, LiteratureIndex, StateStore, utc_now, write_json_atomic
from .safety import CodeSafetyGate, IsolatedPythonRunner


class ResearchModel(ABC):
    """Reasoning boundary used by the autonomous outer loop."""

    @abstractmethod
    def propose(
        self,
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
        memory: list[dict[str, Any]],
        steering: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def implement(self, proposal: dict[str, Any], context: dict[str, Any], memory: list[dict[str, Any]]) -> str:
        raise NotImplementedError

    @abstractmethod
    def reflect(self, proposal: dict[str, Any], result: dict[str, Any], memory: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleResearchModel(ResearchModel):
    """Production research model backed by OpenAI Responses or a compatible endpoint."""

    def __init__(self, client: LLMClient):
        self.client = client

    def propose(
        self,
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
        memory: list[dict[str, Any]],
        steering: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self.client.complete_json(
            """You are an autonomous ML research planner. Diagnose the supplied ranking benchmark and create 3-5 bounded experiments. The organizer FM is the immutable anchor lineage: do not restart the research from zero or propose unrelated novelty merely for family coverage. Condition every idea on the baseline recipe, prior measured experiments, literature, runtime, and available artifacts. Normally devote at least two candidates to conservative improvements or ablations around the strongest reproducible lineage and at most one candidate to higher-risk exploration. Do not repeat a lineage that lost more than 0.01 primary unless the proposal identifies a concrete mechanism that repairs the measured failure. Prefer bounded, vectorizable changes; avoid Python row loops, repeated full-frame sorts, and elaborate causal accumulators. A trusted training-only temporal tournament screens candidates before the external evaluator, which alone owns official metrics and champion promotion. Require an additional internal holdout only when tuning hyperparameters, thresholds, or blend weights. Every candidate must contain id, title, hypothesis, change_kind, model_family, required_inputs, required_capabilities, compute_preference, research_basis, expected_gain, risk, estimated_minutes, acceptance_rule, and abort_condition. required_inputs may contain only file paths explicitly listed in available_inputs; put trusted_components functions in required_capabilities, never required_inputs. If an idea needs an unavailable checkpoint, prediction vector, feature, label, or library, do not propose it. expected_gain is 0 to 0.03; risk is 0 to 1; estimated_minutes is 0.1 to 10. Use only training labels and validation features. Return JSON with diagnostic, research_query, and candidates.""",
            json.dumps({"benchmark": context, "evidence": evidence, "memory": memory[-8:], "human_steering": steering[-4:]}),
            phase="planning",
        )
        if not response:
            raise RuntimeError("Research planner did not return valid JSON")
        return response

    def implement(self, proposal: dict[str, Any], context: dict[str, Any], memory: list[dict[str, Any]]) -> str:
        response = self.client.complete_json(
            """Write one complete Python experiment program for the supplied hypothesis and benchmark program contract. Return JSON with only a code field. Define main() and call it under if __name__ == '__main__'. The program only needs to train and write predictions; the trusted external evaluator calculates official metrics and compares with the champion. Do not reproduce the official evaluator. Prefer the exact supplied trusted_components signatures; for FM work use fit_predict_fm or paired_fm_predictions instead of inventing wrapper methods or optimizer APIs. TrustedFM supports fit(...), step(...), and predict(...), with only the documented constructor arguments. Use vectorized pandas/numpy/LightGBM operations and never loop over dataset rows. If the proposal tunes hyperparameters, thresholds, or blend weights, use a strictly training-only chronological holdout; otherwise do not add an unnecessary internal evaluator. Obey the supplied input paths, output schema, allowed libraries, and label boundary exactly. Never use network, subprocess, dynamic execution, absolute paths, parent paths, validation labels, test data, or evaluator files. Scores must be finite. The validation input intentionally has no labels.""",
            json.dumps({"proposal": proposal, "benchmark_contract": context, "relevant_memory": memory[-5:]}),
            phase="implementing",
        )
        if not response or not isinstance(response.get("code"), str):
            raise RuntimeError("Research coder did not return a code string")
        return response["code"]

    def reflect(self, proposal: dict[str, Any], result: dict[str, Any], memory: list[dict[str, Any]]) -> dict[str, Any]:
        response = self.client.complete_json(
            """Act as a rigorous ML research reviewer. Interpret the experiment outcome without inventing metrics. Return JSON with conclusion, causal_claim_strength (none/weak/moderate/strong), reusable_lesson, next_research_question, and whether_to_retry. A failed safety or execution check is evidence about implementation, not model quality.""",
            json.dumps({"proposal": proposal, "result": result, "prior_memory": memory[-5:]}),
            phase="reflecting",
        )
        return response or {
            "conclusion": "No valid reflection was returned.",
            "causal_claim_strength": "none",
            "reusable_lesson": "Retain the raw result and avoid unsupported conclusions.",
            "next_research_question": "How can the experiment be made valid and measurable?",
            "whether_to_retry": result.get("status") != "completed",
        }


class ScriptedValidationModel(ResearchModel):
    """Deterministic model double proving the outer loop, not a competition policy.

    The first generated implementation deliberately violates the safety policy.
    The second repairs the program and learns a user-category affinity feature.
    Nothing in GenericResearchAgent knows either model choice.
    """

    def __init__(self):
        self.implementation_attempt = 0

    def propose(
        self,
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
        memory: list[dict[str, Any]],
        steering: list[dict[str, Any]],
    ) -> dict[str, Any]:
        retry = any(item.get("stage") == "safety" for item in memory)
        suffix = "safe-repair" if retry else "initial"
        return {
            "diagnostic": "The baseline cannot transfer user-specific category preference to unseen validation items.",
            "research_query": "personalized affinity features for ranking unseen items",
            "candidates": [
                {
                    "id": f"user-category-affinity-{suffix}",
                    "title": "Learn causal user-category affinity",
                    "hypothesis": "A smoothed user-category long-view rate will rank unseen items according to each user's observed interests.",
                    "change_kind": "feature_and_scoring_rule",
                    "research_basis": [card.get("id") for card in evidence[:3]],
                    "expected_gain": 0.25,
                    "risk": 0.12 if retry else 0.22,
                    "estimated_minutes": 1,
                    "acceptance_rule": "Primary exceeds the popularity baseline.",
                    "abort_condition": "Reject if the program accesses private labels or produces misaligned rows.",
                }
            ],
        }

    def implement(self, proposal: dict[str, Any], context: dict[str, Any], memory: list[dict[str, Any]]) -> str:
        self.implementation_attempt += 1
        if self.implementation_attempt == 1:
            return """import json
import urllib.request
from pathlib import Path

def main():
    urllib.request.urlopen('https://example.com')

if __name__ == '__main__':
    main()
"""
        return """import json
from collections import defaultdict
from pathlib import Path

def main():
    train = json.loads(Path('data/train.json').read_text(encoding='utf-8'))
    validation = json.loads(Path('data/validation.json').read_text(encoding='utf-8'))
    sums = defaultdict(float)
    counts = defaultdict(int)
    user_sums = defaultdict(float)
    user_counts = defaultdict(int)
    total_positive = 0.0

    for row in train:
        key = (row['user_id'], row['category'])
        label = float(row['long_view'])
        sums[key] += label
        counts[key] += 1
        user_sums[row['user_id']] += label
        user_counts[row['user_id']] += 1
        total_positive += label

    global_rate = total_positive / max(1, len(train))
    predictions = []
    for row in validation:
        user_rate = (user_sums[row['user_id']] + 4.0 * global_rate) / (user_counts[row['user_id']] + 4.0)
        key = (row['user_id'], row['category'])
        score = (sums[key] + 2.0 * user_rate) / (counts[key] + 2.0)
        predictions.append({'row_id': row['row_id'], 'score': score})

    Path('predictions.json').write_text(json.dumps(predictions), encoding='utf-8')

if __name__ == '__main__':
    main()
"""

    def reflect(self, proposal: dict[str, Any], result: dict[str, Any], memory: list[dict[str, Any]]) -> dict[str, Any]:
        if result.get("stage") == "safety":
            return {
                "conclusion": "The hypothesis was not tested because generated code requested a forbidden network capability.",
                "causal_claim_strength": "none",
                "reusable_lesson": "Regenerate the same bounded hypothesis using only the public data contract and standard-library aggregation.",
                "next_research_question": "Can the affinity feature be implemented without external access?",
                "whether_to_retry": True,
            }
        return {
            "conclusion": "The personalized affinity scorer improved both user-level discrimination and top-five ordering.",
            "causal_claim_strength": "moderate",
            "reusable_lesson": "User-conditioned transferable features can outperform item popularity when validation items are unseen.",
            "next_research_question": "Does the gain persist under stronger smoothing and temporal validation?",
            "whether_to_retry": False,
        }


class GenericResearchAgent:
    """Benchmark-agnostic hypothesis → code → evidence learning loop."""

    def __init__(
        self,
        benchmark: Benchmark,
        research_model: ResearchModel,
        literature: LiteratureIndex,
        workspace: Path,
        max_experiments: int = 6,
        budget_seconds: int = 1800,
        convergence_epsilon: float = 0.002,
        convergence_patience: int = 3,
    ):
        self.benchmark = benchmark
        self.model = research_model
        self.literature = literature
        self.workspace = workspace
        self.experiments_dir = workspace / "experiments"
        self.store = StateStore(workspace / "state.json", workspace / "events.jsonl")
        self.safety = CodeSafetyGate()
        self.runner = IsolatedPythonRunner(timeout_seconds=20)
        self.max_experiments = max_experiments
        self.budget_seconds = budget_seconds
        self.convergence_epsilon = convergence_epsilon
        self.convergence_patience = convergence_patience

    def initialize(self, force: bool = False) -> dict[str, Any]:
        existing = self.store.load()
        if existing and not force:
            return existing
        baseline = self.benchmark.initialize()
        context = self.benchmark.public_context()
        state = {
            "schema_version": 2,
            "run": {
                "id": f"validation-{uuid.uuid4().hex[:8]}",
                "status": "ready",
                "benchmark": context["benchmark"],
                "label": context["label"],
                "metrics": context["metrics"],
                "budget_seconds": self.budget_seconds,
                "elapsed_seconds": 0,
                "max_experiments": self.max_experiments,
                "manual_interventions": 0,
                "executor_mode": "generated_code",
                "llm_mode": type(self.model).__name__,
            },
            "baseline": baseline,
            "best": {"experiment_id": "iteration-000", "title": baseline["title"], "metrics": baseline["metrics"]},
            "benchmark_context": context,
            "current_experiment": None,
            "candidate_queue": [],
            "experiments": [],
            "decisions": [],
            "steering": [],
            "literature_hits": [],
            "memory": [],
            "events": [],
            "created_at": utc_now(),
        }
        self.store.save(state)
        self.store.event("baseline", "Immutable benchmark baseline reproduced.", metrics=baseline["metrics"])
        self.store.event("system", "Generic autonomous research loop is ready.")
        return self.store.load()

    def steer(self, message: str) -> dict[str, Any]:
        clean = " ".join(message.strip().split())[:1000]
        if not clean:
            raise ValueError("Steering message cannot be empty")
        intervention = {"id": uuid.uuid4().hex[:10], "timestamp": utc_now(), "message": clean, "status": "pending"}

        def add(state: dict[str, Any]) -> None:
            state["steering"].append(intervention)
            state["run"]["manual_interventions"] += 1

        state = self.store.mutate(add)
        self.store.event("steering", "Human guidance recorded for the next decision boundary.", steering_id=intervention["id"])
        return state

    def run(self, force: bool = False) -> dict[str, Any]:
        state = self.initialize(force=force)
        started = time.monotonic()

        def mark_running(current: dict[str, Any]) -> None:
            current["run"].update(status="running", started_at=utc_now())

        self.store.mutate(mark_running)
        self.store.event("control", "Autonomous validation run started.")

        while True:
            state = self.store.load()
            elapsed = time.monotonic() - started
            state["run"]["elapsed_seconds"] = round(elapsed, 3)
            self.store.save(state)
            if elapsed >= self.budget_seconds:
                return self._finish("budget_exhausted", "Budget exhausted; best valid experiment retained.")
            if len(state["experiments"]) >= self.max_experiments:
                return self._finish("completed", "Maximum autonomous experiment count reached.")
            if self._converged(state):
                return self._finish("converged", "Convergence rule reached.")
            try:
                self._iteration()
            except Exception as exc:
                self.store.event("error", "Research iteration raised an unrecovered controller error.", error=str(exc))
                return self._finish("error", f"Controller error: {exc}")

    def _iteration(self) -> None:
        state = self.store.load()
        context = state["benchmark_context"]
        memory = state["memory"]
        query = " ".join(context.get("observations", []))
        if memory:
            query += " " + " ".join(str(item.get("next_research_question", "")) for item in memory[-3:])
        evidence = self.literature.search(query, limit=6)
        proposal_bundle = self.model.propose(context, evidence, memory, state["steering"])
        candidates = self._validate_candidates(proposal_bundle.get("candidates"))
        selected = self._select(candidates)
        iteration = len(state["experiments"]) + 1
        experiment_id = f"iteration-{iteration:03d}"
        selected.update({"experiment_id": experiment_id, "status": "planning", "started_at": utc_now()})
        decision = {
            "id": uuid.uuid4().hex[:10],
            "timestamp": utc_now(),
            "diagnostic": proposal_bundle.get("diagnostic"),
            "research_query": proposal_bundle.get("research_query"),
            "selected_action_id": selected["id"],
            "reason": f"Selected by generic acquisition score {selected['policy_score']:.4f} from {len(candidates)} model-proposed candidates.",
            "candidate_scores": [{"id": item["id"], "score": item["policy_score"]} for item in candidates],
        }

        def select(current: dict[str, Any]) -> None:
            current["candidate_queue"] = candidates
            current["current_experiment"] = copy.deepcopy(selected)
            current["decisions"].append(decision)
            current["literature_hits"] = evidence
            for item in current["steering"]:
                if item["status"] == "pending":
                    item["status"] = "considered"

        self.store.mutate(select)
        self.store.event("research", "Hybrid evidence retrieval completed.", query=decision["research_query"], sources=[item["id"] for item in evidence])
        self.store.event("decision", f"Agent selected: {selected['title']}", experiment_id=experiment_id, reason=decision["reason"])

        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(experiment_dir / "proposal.json", selected)
        paths = self.benchmark.prepare_experiment(experiment_dir)
        source = self.model.implement(selected, context, memory)
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        safety = self.safety.inspect(source)
        write_json_atomic(experiment_dir / "safety.json", safety)
        if not safety["passed"]:
            result = {
                "status": "failed", "stage": "safety", "metrics": {},
                "error": "; ".join(safety["findings"]), "safety": safety,
                "source_sha256": source_hash, "artifacts": [str(experiment_dir / "safety.json")],
            }
            (experiment_dir / "rejected_experiment.py").write_text(source, encoding="utf-8")
            self._complete(selected, result)
            return

        execution = self.runner.run(experiment_dir, source)
        result = {**execution, "metrics": {}, "safety": safety, "source_sha256": source_hash, "paths": paths}
        if execution["status"] == "completed":
            try:
                result["metrics"] = self.benchmark.evaluate_predictions(experiment_dir / paths["predictions"])
                result["stage"] = "evaluation"
            except Exception as exc:
                result.update(status="failed", stage="evaluation", error=str(exc))
        self._complete(selected, result)

    @staticmethod
    def _validate_candidates(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError("Planner must return at least one candidate")
        required = {
            "id", "title", "hypothesis", "change_kind", "research_basis",
            "expected_gain", "risk", "estimated_minutes", "acceptance_rule", "abort_condition",
        }
        candidates = []
        for raw in value[:6]:
            if not isinstance(raw, dict) or not required.issubset(raw):
                raise ValueError(f"Candidate is missing required fields: {sorted(required - set(raw or {}))}")
            candidate = copy.deepcopy(raw)
            candidate["expected_gain"] = max(0.0, min(0.03, float(candidate["expected_gain"])))
            candidate["risk"] = max(0.0, min(1.0, float(candidate["risk"])))
            candidate["estimated_minutes"] = max(0.1, min(10.0, float(candidate["estimated_minutes"])))
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        for candidate in candidates:
            # Optimize anytime research value. Expected gains are small absolute
            # metric deltas, so scale them before applying the bounded risk
            # penalty; otherwise risk numerically dominates every useful idea.
            gain_per_sqrt_minute = 100.0 * candidate["expected_gain"] / math.sqrt(candidate["estimated_minutes"])
            candidate["policy_score"] = round(gain_per_sqrt_minute - 0.05 * candidate["risk"], 6)
        candidates.sort(key=lambda item: (-item["policy_score"], item["estimated_minutes"], item["id"]))
        return copy.deepcopy(candidates[0])

    def _complete(self, proposal: dict[str, Any], result: dict[str, Any]) -> None:
        state = self.store.load()
        champion_score = float(state["best"]["metrics"]["primary"])
        score = result.get("metrics", {}).get("primary")
        improved = result.get("status") == "completed" and isinstance(score, (int, float)) and score > champion_score
        delta = round(float(score) - champion_score, 6) if isinstance(score, (int, float)) else None
        reflection = self.model.reflect(proposal, {**result, "improved": improved, "delta_vs_champion": delta}, state["memory"])
        completed = {
            **copy.deepcopy(proposal), **copy.deepcopy(result),
            "action_id": proposal["id"], "completed_at": utc_now(),
            "improved": improved, "delta_vs_champion": delta, "reflection": reflection,
        }
        lesson = {
            "experiment_id": proposal["experiment_id"],
            "stage": result.get("stage"),
            "status": result.get("status"),
            "hypothesis": proposal["hypothesis"],
            "metrics": result.get("metrics", {}),
            "improved": improved,
            **reflection,
        }

        def record(current: dict[str, Any]) -> None:
            current["experiments"].append(completed)
            current["memory"].append(lesson)
            current["current_experiment"] = None
            if improved:
                current["best"] = {
                    "experiment_id": proposal["experiment_id"],
                    "title": proposal["title"],
                    "metrics": result["metrics"],
                }

        self.store.mutate(record)
        if result.get("status") == "completed":
            self.store.event("result", f"Experiment completed: {proposal['title']}", experiment_id=proposal["experiment_id"], metrics=result["metrics"], improved=improved, delta=delta)
        else:
            self.store.event("recovery", f"Experiment rejected or failed: {proposal['title']}", experiment_id=proposal["experiment_id"], stage=result.get("stage"), error=result.get("error"))
        self.store.event("reflection", reflection.get("reusable_lesson", "Experiment result stored."), experiment_id=proposal["experiment_id"])

    def _converged(self, state: dict[str, Any]) -> bool:
        valid = [item for item in state["experiments"] if item.get("status") == "completed"]
        if len(valid) < self.convergence_patience:
            return False
        return all((item.get("delta_vs_champion") or 0.0) <= self.convergence_epsilon for item in valid[-self.convergence_patience:])

    def _finish(self, status: str, message: str) -> dict[str, Any]:
        self.store.mutate(lambda state: state["run"].update(status=status, completed_at=utc_now()))
        self.store.event("complete", message, status=status)
        return self.store.load()
