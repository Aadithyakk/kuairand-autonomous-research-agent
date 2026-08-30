from __future__ import annotations

import json
from pathlib import Path


METHOD_CARDS_PATH = Path(__file__).with_name("method_cards.json")


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


def summarize_search_tree(iterations: list[dict], limit: int = 20) -> list[dict]:
    """Return concise AIDE-style nodes rather than an ever-growing raw history."""
    nodes = []
    for item in iterations[-limit:]:
        metrics = item.get("metrics") or {}
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
        })
    return nodes
