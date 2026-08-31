from __future__ import annotations

import json
from pathlib import Path


METHOD_CARDS_PATH = Path(__file__).with_name("method_cards.json")
RESEARCH_PRIORS_PATH = Path(__file__).with_name("research_priors.json")


def load_method_cards() -> list[dict]:
    cards = json.loads(METHOD_CARDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(cards, list) or not cards:
        raise ValueError("Method-card library must contain at least one card")
    required = {"id", "category", "status", "paper", "source", "method", "fit", "smallest_ablation", "risk"}
    for card in cards:
        missing = required - set(card)
        if missing:
            raise ValueError(f"Method card {card.get('id', '<unknown>')} is missing {sorted(missing)}")
    return cards


def load_research_priors() -> dict:
    priors = json.loads(RESEARCH_PRIORS_PATH.read_text(encoding="utf-8"))
    if not isinstance(priors, dict) or not priors:
        raise ValueError("Research-prior library must contain at least one prior")
    required = {"status", "teacher", "transferable_strategy", "submission_safe_translation", "priority_experiments", "prohibited_claim"}
    for prior_id, prior in priors.items():
        if not isinstance(prior, dict):
            raise ValueError(f"Research prior {prior_id} must be an object")
        missing = required - set(prior)
        if missing:
            raise ValueError(f"Research prior {prior_id} is missing {sorted(missing)}")
    return priors


def summarize_search_tree(iterations: list[dict], limit: int = 20) -> list[dict]:
    """Return concise AIDE-style nodes rather than an ever-growing raw history."""
    nodes = []
    for item in iterations[-limit:]:
        metrics = item.get("metrics") or {}
        executor_review = item.get("executor_review") or {}
        nodes.append({
            "node": int(item.get("number", 0)),
            "parent": item.get("parent_iteration"),
            "status": item.get("status"),
            "strategy": item.get("strategy"),
            "operator": item.get("operator"),
            "component": item.get("component"),
            "title": item.get("title"),
            "primary": metrics.get("primary"),
            "gauc": metrics.get("gauc"),
            "ndcg5": metrics.get("ndcg5"),
            "gain": item.get("gain"),
            "failure": item.get("error"),
            "experiment_type": item.get("experiment_type"),
            "executor_slug": executor_review.get("slug"),
            "executor_status": executor_review.get("status"),
        })
    return nodes
