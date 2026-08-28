from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from pathlib import Path
from typing import Any, Callable

from .autonomous import GenericResearchAgent, OpenAICompatibleResearchModel
from .core import LLMClient, LiteratureIndex, read_json, utc_now, write_json_atomic
from .safety import CodeSafetyGate


BASELINE = {"GAUC": 0.6674002647399903, "nDCG@5": 0.5357441067695617, "primary": 0.601572185754776}
REVIEW_PROMPT = """You are the independent methodological reviewer in an autonomous recommender-research agent. Audit the proposed Python program against the hypothesis and benchmark contract. You diagnose only; you never rewrite code. The program's sole deliverable is one prediction per validation row and a trusted evaluator owns official metrics. If the proposal tunes hyperparameters, thresholds, gates, or ensemble weights, require a training-only chronological holdout. Verify projected columns separately against train_columns and validation_columns; row_id is validation-only and long_view is training-only. Check leakage, row order, unavailable inputs, hypothesis-to-code fidelity, finite scores, runtime, and outcome access. Return JSON with exactly approved (boolean) and issues (array of concise, actionable strings). Do not approve a different hypothesis."""

REPAIR_PROMPT = """You are the coding agent repairing one autonomous ML experiment. Return JSON with only a code field containing a complete Python program. Preserve the proposal's hypothesis and model family exactly; do not substitute another model or weaken the acceptance/abort rules. Fix every supplied reviewer or deterministic issue. Obey the benchmark paths, schemas, label boundary, runtime, output contract, and allowed libraries. Prefer the supplied trusted_components helpers for frame loading, chronological splitting, FM encoding/training, and prediction validation when relevant. Use vectorized operations, define main(), call it, and write one finite score per validation row in unchanged row_id order. Never use validation outcomes, network, subprocess, dynamic execution, absolute paths, or parent paths."""


def kuairand_context(memory: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "benchmark": "KuaiRand-Pure within-user long-view ranking",
        "task": "Rank exposed validation videos independently within each user.",
        "label": "long_view",
        "metrics": ["GAUC", "nDCG@5", "primary = arithmetic mean"],
        "exact_metric_contract": {
            "ranking_scope": "Rank independently within each user; preserve validation row_id for deterministic tie-breaking.",
            "GAUC": "For each user with both classes, compute standard ROC AUC with 0.5 credit for score ties; aggregate user AUC weighted by that user's positive count.",
            "nDCG@5": "Sort each user's rows by descending score then ascending row_id; compute binary DCG@5 / ideal DCG@5; users with no positives contribute 0 and remain in the mean.",
            "primary": "Arithmetic mean of GAUC and nDCG@5.",
        },
        "train": {"dates": "20220408-20220421", "rows": 1_141_112},
        "validation": {
            "dates": "20220422-20220428",
            "rows": 124_909,
            "labels": "held by the external evaluator and absent from validation.parquet",
        },
        "official_fm_five_seed": BASELINE,
        "observed_experiments": [
            {
                "title": "Causal history LightGBM LambdaRank",
                "metrics": {"GAUC": 0.6636374904605207, "nDCG@5": 0.5331166190352395, "primary": 0.5983770547478802},
                "lesson": "Ranking-loss alignment alone did not compensate for weaker generalization than the five-field FM.",
            },
            {
                "title": "Validation-calibrated rank ensemble",
                "metrics": {"GAUC": 0.6693836797580355, "nDCG@5": 0.5365475685771528, "primary": 0.6029656241675941},
                "lesson": "Diversity helped directionally, but the 0.00139 gain is below the 0.002 convergence epsilon and weights were chosen on validation.",
            },
        ],
        "program_contract": {
            "train_path": "data/train.parquet",
            "validation_path": "data/validation.parquet",
            "training_label": "long_view",
            "validation_has_label": False,
            "output_path": "predictions.npy",
            "output_schema": "one finite float per validation row in unchanged row_id order",
            "allowed_libraries": ["pandas", "numpy", "lightgbm", "sklearn", "math", "collections", "gc", "json", "pathlib", "time", "trusted_components"],
            "runtime_limit_minutes": 10,
            "memory_limit_gb": 24,
            "available_inputs": ["data/train.parquet", "data/validation.parquet"],
            "trusted_components": [
                "trusted_components.load_frames(train_columns, validation_columns)",
                "trusted_components.chronological_split(frame, holdout_dates=2)",
                "trusted_components.save_predictions(scores, validation)",
                "trusted_components.fit_predict_fm(train, validation, columns, label='long_view', factors=16, learning_rate=0.001, l2=1e-6, epochs=2, batch_size=8192, seed=2026)",
                "trusted_components.paired_fm_predictions(train, validation, control_columns, treatment_columns, **settings)",
                "trusted_components.TrustedFM(dimension, factors=16, learning_rate=0.001, l2=1e-6, seed=0).fit(matrix, labels, epochs=2, batch_size=8192, seed=2026).predict(matrix)",
                "trusted_components.encode_fm(train, validation, columns)",
                "trusted external GAUC/nDCG evaluator and smoke-test runner",
            ],
        },
        "schema": {
            "row_id": "validation-only alignment key; absent from training",
            "user_id": "categorical",
            "video_id": "categorical",
            "author_id": "categorical",
            "date": "integer YYYYMMDD",
            "hourmin": "integer time context",
            "time_ms": "chronological timestamp",
            "duration_ms": "request-time duration/context",
            "tab": "categorical request context",
            "video_type": "categorical metadata",
            "upload_type": "categorical metadata",
            "visible_status": "categorical metadata",
            "video_duration": "numeric metadata",
            "server_width": "numeric metadata",
            "server_height": "numeric metadata",
            "music_id": "categorical metadata",
            "music_type": "categorical metadata",
            "tag": "categorical/list-like metadata",
            "long_view": "binary training-only outcome",
        },
        "train_columns": [
            "user_id", "video_id", "author_id", "date", "hourmin", "time_ms", "duration_ms", "tab",
            "video_type", "upload_dt", "upload_type", "visible_status", "video_duration", "server_width",
            "server_height", "music_id", "music_type", "tag", "long_view"
        ],
        "validation_columns": [
            "row_id", "user_id", "video_id", "author_id", "date", "hourmin", "time_ms", "duration_ms", "tab",
            "video_type", "upload_dt", "upload_type", "visible_status", "video_duration", "server_width",
            "server_height", "music_id", "music_type", "tag"
        ],
        "hard_boundaries": [
            "No validation outcomes, post-exposure outcomes, test data, evaluator files, network, subprocess, dynamic execution, or absolute/parent paths.",
            "Same-row click/like/follow/comment/watch outcomes are unavailable at inference and forbidden as features.",
            "If tuning is necessary, create a temporal holdout from training only.",
            "The temporal holdout is a directional candidate gate, not a requirement to reproduce the external five-seed official FM inside every candidate program.",
            "Only the trusted external evaluator compares a completed candidate with the immutable official FM baseline and decides champion promotion.",
            "The experiment must define main(), call it, and preserve validation row order.",
            "An implementation may train its own FM, but no precomputed FM score, checkpoint, or out-of-fold prediction artifact is currently available.",
        ],
        "memory": memory[-10:],
    }


def deterministic_findings(source: str, proposal: dict[str, Any]) -> list[str]:
    gate = CodeSafetyGate({"gc", "lightgbm", "numpy", "pandas", "sklearn", "time", "trusted_components"}).inspect(source)
    findings = list(gate["findings"])
    lowered = source.lower()
    # save_predictions owns the canonical output path. Requiring the literal
    # filename made valid helper-based programs enter needless repair loops.
    if "predictions.npy" not in lowered and "save_predictions" not in lowered:
        findings.append("Program must write predictions.npy")
    if "isfinite" not in lowered and "save_predictions" not in lowered:
        findings.append("Program must explicitly reject non-finite predictions")
    hypothesis = " ".join(str(proposal.get(key, "")) for key in ("title", "hypothesis", "model_family")).lower()
    if ("factorization machine" in hypothesis or re.search(r"\bfm\b", hypothesis)) and not any(
        token in lowered for token in (
            "factorization", "class numpyfm", "class fm", "embedding",
            "fit_predict_fm", "paired_fm_predictions", "trustedfm",
        )
    ):
        findings.append("Hypothesis requires an FM implementation, but the program does not contain one")
    findings.extend(trusted_api_findings(source))
    return sorted(set(findings))


def trusted_api_findings(source: str) -> list[str]:
    """Validate model-authored calls against the concrete trusted component API."""

    allowed_exports = {
        "TrustedFM", "load_frames", "chronological_split", "save_predictions", "sigmoid",
        "encode_fm", "fit_predict_fm", "paired_fm_predictions",
    }
    trusted_fm_keywords = {"dimension", "factors", "learning_rate", "l2", "seed"}
    findings: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    module_aliases = {"trusted_components"}
    imported_names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "trusted_components":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "trusted_components":
            for alias in node.names:
                imported_names[alias.asname or alias.name] = alias.name
                if alias.name not in allowed_exports:
                    findings.append(f"Unknown trusted_components export: {alias.name}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        export = None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in module_aliases:
            export = node.func.attr
        elif isinstance(node.func, ast.Name) and node.func.id in imported_names:
            export = imported_names[node.func.id]
        if export and export not in allowed_exports:
            findings.append(f"Unknown trusted_components export: {export}")
        if export == "TrustedFM":
            invalid = sorted(keyword.arg for keyword in node.keywords if keyword.arg and keyword.arg not in trusted_fm_keywords)
            if invalid:
                findings.append(f"Unsupported TrustedFM constructor arguments: {', '.join(invalid)}")
    return sorted(set(findings))


def normalize_requirements(proposal: dict[str, Any], context: dict[str, Any]) -> None:
    """Keep file dependencies separate from callable trusted capabilities."""

    available = set(context["program_contract"]["available_inputs"])
    raw = proposal.get("required_inputs", [])
    if not isinstance(raw, list):
        return
    files, capabilities = [], list(proposal.get("required_capabilities", []))
    for item in map(str, raw):
        if item in available or item.startswith("data/"):
            files.append(item)
        elif item.startswith("trusted_components"):
            capabilities.append(item)
        else:
            files.append(item)
    proposal["required_inputs"] = list(dict.fromkeys(files))
    proposal["required_capabilities"] = list(dict.fromkeys(map(str, capabilities)))


def feasibility_findings(proposal: dict[str, Any], context: dict[str, Any]) -> list[str]:
    available = set(context["program_contract"]["available_inputs"])
    required = proposal.get("required_inputs", [])
    if not isinstance(required, list):
        return ["required_inputs must be an array"]
    missing = sorted(str(item) for item in required if str(item) not in available)
    findings = [f"Unavailable required input: {item}" for item in missing]
    text = " ".join(str(proposal.get(key, "")) for key in ("title", "hypothesis", "change_kind")).lower()
    capabilities = " ".join(map(str, proposal.get("required_capabilities", []))).lower()
    can_train_fm = any(
        token in capabilities
        for token in ("fit_predict_fm", "paired_fm_predictions", "trustedfm", "encode_fm")
    )
    if any(term in text for term in ("fm score", "fm residual", "residual over fm", "fm checkpoint")) and not can_train_fm:
        findings.append("FM residual/blend requires a score or checkpoint artifact that is not currently available")
    return findings


def repair_runtime_candidate(
    config_path: Path,
    memory_path: Path | None,
    proposal: dict[str, Any],
    source: str,
    runtime_issues: list[str],
    output: Path,
    progress: Callable[[str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Repair the same hypothesis from a trusted runtime trace, then re-review it."""

    config = read_json(config_path)
    memory = read_json(memory_path) if memory_path and memory_path.exists() else []
    context = kuairand_context(memory)
    client = LLMClient(config["llm"], progress=progress, cancelled=should_stop)
    output.mkdir(parents=True, exist_ok=True)
    repaired = repair_program(client, proposal, context, source, runtime_issues)
    (output / "candidate.runtime-raw.py").write_text(repaired, encoding="utf-8")
    reviewed, reviews = review_program(
        client, proposal, context, repaired, output, max_rounds=2, stage="runtime", progress=progress
    )
    safety = CodeSafetyGate({"gc", "lightgbm", "numpy", "pandas", "sklearn", "time", "trusted_components"}).inspect(reviewed)
    deterministic = deterministic_findings(reviewed, proposal)
    approved = bool(reviews and reviews[-1]["approved"] and safety["passed"] and not deterministic)
    (output / "candidate.runtime-final.py").write_text(reviewed, encoding="utf-8")
    return {"approved": approved, "source": reviewed, "reviews": reviews, "safety": safety, "findings": deterministic}


def repair_program(
    client: LLMClient,
    proposal: dict[str, Any],
    context: dict[str, Any],
    source: str,
    issues: list[str],
) -> str:
    response = client.complete_json(
        REPAIR_PROMPT,
        json.dumps({"proposal": proposal, "benchmark_contract": context, "current_program": source, "issues": issues}),
        phase="repairing",
    )
    if not response or not isinstance(response.get("code"), str):
        raise RuntimeError(f"Coding repair failed: {client.last_error or 'invalid repair contract'}")
    return response["code"]


def review_program(
    client: LLMClient,
    proposal: dict[str, Any],
    context: dict[str, Any],
    source: str,
    output: Path,
    max_rounds: int = 2,
    stage: str = "sol",
    progress: Callable[[str, str], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    reviews = []
    current = source
    for index in range(1, max_rounds + 1):
        issues = deterministic_findings(current, proposal)
        source_kind = "deterministic"
        approved = False
        if not issues:
            review = client.complete_json(
                REVIEW_PROMPT,
                json.dumps({"proposal": proposal, "benchmark_contract": context, "program": current}),
                phase="reviewing",
            )
            if not review or not isinstance(review.get("issues"), list):
                raise RuntimeError(f"Research code review failed: {client.last_error or 'invalid review contract'}")
            issues = [str(item) for item in review.get("issues", [])]
            approved = bool(review.get("approved")) and not issues
            source_kind = "methodology"
        record = {"stage": stage, "round": index, "approved": approved, "source": source_kind, "issues": issues}
        reviews.append(record)
        (output / f"candidate.{stage}-review-{index}.py").write_text(current, encoding="utf-8")
        if progress:
            verdict = "approved" if record["approved"] else "requested a repair"
            progress("reviewing", f"{stage.title()} review round {index} {verdict}.")
        if record["approved"]:
            break
        current = repair_program(client, proposal, context, current, issues)
        (output / f"candidate.{stage}-repair-{index}.py").write_text(current, encoding="utf-8")
    return current, reviews


def generate(
    output: Path,
    config_path: Path,
    literature_path: Path,
    memory_path: Path | None = None,
    progress: Callable[[str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    memory = read_json(memory_path) if memory_path and memory_path.exists() else []
    literature = LiteratureIndex.from_path(literature_path)
    context = kuairand_context(memory)
    evidence = literature.search(
        "recommender ranking causal history hybrid factorization machine residual multi behavior sequence ensemble",
        limit=8,
    )
    client = LLMClient(config["llm"], progress=progress, cancelled=should_stop)
    model = OpenAICompatibleResearchModel(client)
    bundle = model.propose(context, evidence, memory, [])
    if progress:
        progress("proposed", "The research planner generated and ranked new hypotheses.")
    candidates = GenericResearchAgent._validate_candidates(bundle.get("candidates"))
    for candidate in candidates:
        candidate.setdefault("required_inputs", ["data/train.parquet", "data/validation.parquet"])
        normalize_requirements(candidate, context)
        candidate.setdefault("compute_preference", "auto")
        if not candidate.get("model_family"):
            text = " ".join(str(candidate.get(key, "")) for key in ("title", "hypothesis", "change_kind")).lower()
            if any(term in text for term in ("sequence", "din", "dien", "transformer")):
                family = "sequence"
            elif any(term in text for term in ("lightgbm", "lambda", "tree", "boost")):
                family = "tree"
            elif any(term in text for term in ("collaborative", "bpr", "matrix factor")):
                family = "collaborative"
            elif any(term in text for term in ("hybrid", "ensemble", "blend", "stack")):
                family = "hybrid"
            else:
                family = "factorization"
            candidate["model_family"] = family
    GenericResearchAgent._select(candidates)  # assigns scores and sorts in-place
    severe_negative_terms = []
    for item in memory:
        delta = item.get("delta_vs_champion")
        if isinstance(delta, (int, float)) and delta <= -0.01 and not item.get("whether_to_retry", False):
            severe_negative_terms.extend(re.findall(r"[a-z0-9]+", str(item.get("hypothesis", "")).lower()))
    negative_vocabulary = {term for term in severe_negative_terms if len(term) >= 5}
    for candidate in candidates:
        text = " ".join(str(candidate.get(key, "")) for key in ("title", "hypothesis", "model_family")).lower()
        overlap = len(negative_vocabulary.intersection(re.findall(r"[a-z0-9]+", text)))
        repair = any(term in text for term in ("repair", "ablation", "anchor", "baseline", "residual"))
        if overlap >= 3 and not repair:
            candidate["policy_score"] = round(float(candidate["policy_score"]) - 0.12, 6)
    candidates.sort(key=lambda item: (-item["policy_score"], item["estimated_minutes"], item["id"]))
    max_candidate_attempts = int(config.get("campaign", {}).get("candidate_attempts_per_iteration", 3))
    attempts: list[dict[str, Any]] = []
    selected = copy.deepcopy(candidates[0])
    source = ""
    raw_source = ""
    reviews: list[dict[str, Any]] = []
    safety = {"passed": False, "findings": ["No executable candidate was produced"]}
    approved = False
    for attempt_index, candidate in enumerate(candidates[:max_candidate_attempts], start=1):
        selected = copy.deepcopy(candidate)
        attempt_dir = output / f"attempt-{attempt_index:02d}-{selected['id']}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        feasibility = feasibility_findings(selected, context)
        if feasibility:
            attempts.append({"candidate_id": selected["id"], "status": "infeasible_proposal", "issues": feasibility})
            if progress:
                progress("backtracking", f"Candidate {attempt_index} required unavailable inputs; selecting another hypothesis.")
            continue
        try:
            raw_source = model.implement(selected, context, memory)
        except RuntimeError as exc:
            attempts.append({"candidate_id": selected["id"], "status": "implementation_failed", "stage": "implementing", "issues": [str(exc)]})
            if progress:
                progress("backtracking", f"Candidate {attempt_index} implementation failed or timed out; selecting another hypothesis.")
            continue
        (attempt_dir / "candidate.raw.py").write_text(raw_source, encoding="utf-8")
        if progress:
            progress("implemented", f"The coding agent implemented candidate {attempt_index}.")
        try:
            source, reviews = review_program(client, selected, context, raw_source, attempt_dir, progress=progress)
        except RuntimeError as exc:
            attempts.append({"candidate_id": selected["id"], "status": "implementation_failed", "stage": "reviewing", "issues": [str(exc)]})
            if progress:
                progress("backtracking", f"Candidate {attempt_index} review or repair failed; selecting another hypothesis.")
            continue
        safety = CodeSafetyGate({"gc", "lightgbm", "numpy", "pandas", "sklearn", "time", "trusted_components"}).inspect(source)
        approved = bool(reviews and reviews[-1]["approved"] and safety["passed"])
        attempts.append({
            "candidate_id": selected["id"],
            "status": "approved" if approved else "implementation_failed",
            "issues": safety["findings"] if not safety["passed"] else reviews[-1]["issues"],
            "review_rounds": reviews,
        })
        if approved:
            break
        if progress:
            progress("backtracking", f"Candidate {attempt_index} could not pass preflight; selecting another hypothesis.")
    (output / "candidate.raw.py").write_text(raw_source, encoding="utf-8")
    (output / "candidate.py").write_text(source, encoding="utf-8")
    write_json_atomic(output / "proposal.json", selected)
    write_json_atomic(output / "safety.json", safety)
    report = {
        "timestamp": utc_now(),
        "model": config["llm"]["model"],
        "fallback_model": config["llm"].get("fallback_model"),
        "diagnostic": bundle.get("diagnostic"),
        "research_query": bundle.get("research_query"),
        "candidates": candidates,
        "selected": selected,
        "implementation_attempts": attempts,
        "evidence_ids": [item["id"] for item in evidence],
        "code_review": {
            "approved": bool(reviews and reviews[-1]["approved"]),
            "rounds": reviews,
        },
        "approved_for_execution": approved,
        "safety": safety,
        "client_error": client.last_error,
        "artifacts": [
            "proposal.json", "candidate.raw.py",
            "candidate.py", "safety.json",
        ],
    }
    write_json_atomic(output / "generation.json", report)
    return report


def reflect(result_path: Path, proposal_path: Path, config_path: Path, memory_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    prior = read_json(memory_path) if memory_path.exists() else []
    result_report = read_json(result_path)
    proposal = read_json(proposal_path)
    client = LLMClient(config["llm"])
    model = OpenAICompatibleResearchModel(client)
    review = model.reflect(proposal, result_report.get("result", result_report), prior)
    memory_item = {
        "timestamp": utc_now(),
        "experiment_id": proposal["id"],
        "hypothesis": proposal["hypothesis"],
        "title": proposal.get("title"),
        "model_family": proposal.get("model_family"),
        "status": result_report.get("result", {}).get("status"),
        "metrics": result_report.get("result", {}).get("metrics", {}),
        "delta_vs_champion": result_report.get("result", {}).get(
            "delta_vs_official_fm", result_report.get("result", {}).get("delta_vs_anchor")
        ),
        "client_error": client.last_error,
        **review,
    }
    if prior and prior[-1].get("experiment_id") == proposal["id"]:
        prior[-1] = memory_item
    else:
        prior.append(memory_item)
    write_json_atomic(memory_path, prior)
    return memory_item


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one bounded real KuaiRand experiment with the configured research model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--literature", type=Path, default=Path("knowledge/literature.json"))
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--reflect-result", type=Path)
    parser.add_argument("--proposal", type=Path)
    args = parser.parse_args()
    if args.reflect_result:
        if not args.proposal or not args.memory:
            parser.error("--reflect-result requires --proposal and --memory")
        print(json.dumps(reflect(args.reflect_result, args.proposal, args.config, args.memory), indent=2))
        return
    if not args.output:
        parser.error("--output is required when generating an experiment")
    report = generate(args.output, args.config, args.literature, args.memory)
    print(json.dumps(report, indent=2))
    if not report["approved_for_execution"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
