from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .autonomous import GenericResearchAgent, OpenAICompatibleResearchModel
from .core import LLMClient, LiteratureIndex, read_json, utc_now, write_json_atomic
from .safety import CodeSafetyGate


BASELINE = {"GAUC": 0.6674002647399903, "nDCG@5": 0.5357441067695617, "primary": 0.601572185754776}
REVIEW_PROMPT = """You are the independent methodological and code reviewer in an autonomous recommender-research agent. Audit the proposed Python program against the hypothesis and benchmark contract. Check temporal leakage in internal holdout features, exact official metric definitions, within-user ranking transforms, validation row order, unavailable columns, hypothesis-to-code fidelity, memory/runtime, and forbidden outcome access. Return JSON with approved (boolean), issues (array of concise strings), and revised_code (a complete corrected program string). Always provide revised_code; if approved, return the original program unchanged. Do not weaken the safety boundary or substitute a different hypothesis."""


def kuairand_context(memory: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "benchmark": "KuaiRand-Pure within-user long-view ranking",
        "task": "Rank exposed validation videos independently within each user.",
        "label": "long_view",
        "metrics": ["GAUC", "nDCG@5", "primary = arithmetic mean"],
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
            "allowed_libraries": ["pandas", "numpy", "lightgbm", "sklearn", "math", "collections", "gc", "json", "pathlib"],
            "runtime_limit_minutes": 10,
            "memory_limit_gb": 24,
        },
        "schema": {
            "row_id": "alignment key",
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
        "hard_boundaries": [
            "No validation outcomes, post-exposure outcomes, test data, evaluator files, network, subprocess, dynamic execution, or absolute/parent paths.",
            "Same-row click/like/follow/comment/watch outcomes are unavailable at inference and forbidden as features.",
            "If tuning is necessary, create a temporal holdout from training only.",
            "The experiment must define main(), call it, and preserve validation row order.",
        ],
        "memory": memory[-6:],
    }


def review_program(
    client: LLMClient,
    proposal: dict[str, Any],
    context: dict[str, Any],
    source: str,
    output: Path,
    max_rounds: int = 3,
    stage: str = "luna",
) -> tuple[str, list[dict[str, Any]]]:
    reviews = []
    current = source
    for index in range(1, max_rounds + 1):
        review = client.complete_json(
            REVIEW_PROMPT,
            json.dumps({"proposal": proposal, "benchmark_contract": context, "program": current}),
        )
        if not review or not isinstance(review.get("revised_code"), str):
            raise RuntimeError(f"Research code review failed: {client.last_error or 'invalid review contract'}")
        record = {"stage": stage, "round": index, "approved": bool(review.get("approved")), "issues": review.get("issues", [])}
        reviews.append(record)
        current = review["revised_code"]
        (output / f"candidate.{stage}-review-{index}.py").write_text(current, encoding="utf-8")
        if record["approved"]:
            break
    return current, reviews


def generate(output: Path, config_path: Path, literature_path: Path, memory_path: Path | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    memory = read_json(memory_path) if memory_path and memory_path.exists() else []
    literature = LiteratureIndex.from_path(literature_path)
    context = kuairand_context(memory)
    evidence = literature.search(
        "recommender ranking causal history hybrid factorization machine residual multi behavior sequence ensemble",
        limit=8,
    )
    client = LLMClient(config["llm"])
    model = OpenAICompatibleResearchModel(client)
    bundle = model.propose(context, evidence, memory, [])
    candidates = GenericResearchAgent._validate_candidates(bundle.get("candidates"))
    selected = GenericResearchAgent._select(candidates)
    raw_source = model.implement(selected, context, memory)
    source, reviews = review_program(client, selected, context, raw_source, output)
    review_client = client
    if not reviews[-1]["approved"] and config["llm"].get("fallback_model"):
        escalation_config = copy.deepcopy(config["llm"])
        escalation_config["model"] = escalation_config.pop("fallback_model")
        review_client = LLMClient(escalation_config)
        source, escalated_reviews = review_program(
            review_client, selected, context, source, output, max_rounds=2, stage="terra"
        )
        reviews.extend(escalated_reviews)
    safety = CodeSafetyGate({"gc", "lightgbm", "numpy", "pandas", "sklearn"}).inspect(source)
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
        "evidence_ids": [item["id"] for item in evidence],
        "code_review": {
            "approved": bool(reviews and reviews[-1]["approved"]),
            "rounds": reviews,
        },
        "approved_for_execution": bool(reviews and reviews[-1]["approved"] and safety["passed"]),
        "safety": safety,
        "client_error": review_client.last_error or client.last_error,
        "artifacts": [
            "proposal.json", "candidate.raw.py",
            *[f"candidate.{item['stage']}-review-{item['round']}.py" for item in reviews],
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
        "status": result_report.get("result", {}).get("status"),
        "metrics": result_report.get("result", {}).get("metrics", {}),
        "delta_vs_champion": result_report.get("result", {}).get("delta_vs_official_fm"),
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
