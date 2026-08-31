from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .paper_executor import SUPPORTED_PAPER_SIGNALS, build_paper_signal_scores


EXECUTOR_FAMILY = "declarative_signal_reranker_v1"
REQUIRED_CONTRACT_TESTS = {
    "validation_label_invariance",
    "temporal_fit_boundary",
    "finite_output",
    "deterministic_output",
    "output_shape",
    "resource_budget",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _program_from_extension(extension: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_executor_slug": str(extension.get("slug", "")),
        "paper_signals": list(extension.get("signals", [])),
        "paper_signal_weights": list(extension.get("signal_weights", [])),
        "paper_smoothing": float(extension.get("smoothing", 8.0)),
        "paper_item_smoothing": float(extension.get("entity_smoothing", 20.0)),
        "paper_blend_weight": float(extension.get("blend_weight", 0.01)),
    }


def _contract_test(program: dict[str, Any]) -> dict[str, Any]:
    train = [
        (20220408, "u1", "v1", "a1", "0", 10_000.0, 1, 8, 1_000, 9_000.0),
        (20220408, "u1", "v2", "a2", "0", 20_000.0, 0, 8, 2_000, 2_000.0),
        (20220409, "u2", "v1", "a1", "1", 10_000.0, 1, 9, 3_000, 8_000.0),
        (20220409, "u2", "v3", "a3", "1", 30_000.0, 0, 9, 4_000, 3_000.0),
        (20220410, "u3", "v2", "a2", "0", 20_000.0, 1, 10, 5_000, 15_000.0),
        (20220410, "u3", "v3", "a3", "1", 30_000.0, 0, 10, 6_000, 4_000.0),
    ]
    valid = [
        (20220422, "u1", "v1", "a1", "0", 10_000.0, 0, 11, 7_000, 0.0),
        (20220422, "u1", "v3", "a3", "1", 30_000.0, 1, 11, 8_000, 0.0),
        (20220422, "u2", "v2", "a2", "0", 20_000.0, 1, 11, 9_000, 0.0),
        (20220422, "u2", "v3", "a3", "1", 30_000.0, 0, 11, 10_000, 0.0),
    ]
    changed = [tuple([*row[:6], 1 - row[6], *row[7:]]) for row in valid]
    metadata = {
        "v1": {"author_id": "a1", "music_id": "m1", "tag": "dance", "video_type": "NORMAL"},
        "v2": {"author_id": "a2", "music_id": "m1", "tag": "comedy", "video_type": "NORMAL"},
        "v3": {"author_id": "a3", "music_id": "m2", "tag": "dance", "video_type": "AD"},
    }
    started = time.monotonic()
    scores, diagnostics = build_paper_signal_scores(
        train, valid, Path("."), program, video_metadata=metadata,
    )
    repeated, _ = build_paper_signal_scores(
        train, valid, Path("."), program, video_metadata=metadata,
    )
    changed_scores, _ = build_paper_signal_scores(
        train, changed, Path("."), program, video_metadata=metadata,
    )
    elapsed = time.monotonic() - started
    tests = {
        "validation_label_invariance": bool(np.allclose(scores, changed_scores)),
        "temporal_fit_boundary": diagnostics.get("validation_outcomes_accessed") is False,
        "finite_output": bool(np.all(np.isfinite(scores))),
        "deterministic_output": bool(np.array_equal(scores, repeated)),
        "output_shape": scores.shape == (len(valid),),
        "resource_budget": elapsed < 5.0 and len(program["paper_signals"]) <= 6,
    }
    return {
        "passed": all(tests.values()),
        "tests": tests,
        "synthetic_rows": {"train": len(train), "validation": len(valid)},
        "elapsed_seconds": round(elapsed, 6),
        "diagnostics": diagnostics,
    }


def review_and_register_executor(
    registry_root: Path,
    workspace: Path,
    extension: dict[str, Any],
    research_sources: list[dict[str, str]],
) -> dict[str, Any]:
    """Scaffold and register only an exact, paper-backed declarative program."""
    errors: list[str] = []
    slug = str(extension.get("slug", ""))
    paper_url = str(extension.get("paper_url", ""))
    signals = [str(value) for value in extension.get("signals", [])]
    weights = extension.get("signal_weights", [])
    requested_tests = set(extension.get("required_tests", []))
    if extension.get("requested") is not True:
        errors.append("executor_extension.requested must be true for incubation")
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,47}", slug):
        errors.append("slug must be a 3-48 character lowercase identifier")
    parsed = urlparse(paper_url)
    source_urls = {source.get("url") for source in research_sources}
    if parsed.scheme != "https" or not parsed.hostname or paper_url not in source_urls:
        errors.append("paper_url must exactly match an audited HTTPS research source")
    if extension.get("family") != EXECUTOR_FAMILY:
        errors.append(f"family must be {EXECUTOR_FAMILY}")
    if not 1 <= len(signals) <= 6 or len(set(signals)) != len(signals):
        errors.append("signals must contain one to six distinct values")
    if any(signal not in SUPPORTED_PAPER_SIGNALS for signal in signals):
        errors.append("signals contain an unsupported primitive")
    if len(weights) != len(signals):
        errors.append("signal_weights must align one-to-one with signals")
    if requested_tests != REQUIRED_CONTRACT_TESTS:
        errors.append("all mandatory leakage, determinism, shape, and resource tests are required")
    if extension.get("resource_class") not in {"small", "medium"}:
        errors.append("large-resource executor extensions cannot be auto-approved")
    if len(str(extension.get("method_summary", "")).strip()) < 20:
        errors.append("method_summary is too short for review")
    if len(str(extension.get("why_new_executor", "")).strip()) < 20:
        errors.append("why_new_executor must explain the missing capability")

    program = _program_from_extension(extension)
    contract = None
    if not errors:
        try:
            contract = _contract_test(program)
            if not contract["passed"]:
                errors.append("one or more executor contract tests failed")
        except (TypeError, ValueError, KeyError) as error:
            errors.append(f"executor contract rejected the program: {error}")

    status = "approved" if not errors else "rejected"
    review = {
        "slug": slug,
        "status": status,
        "family": extension.get("family"),
        "paper": {"title": extension.get("paper_title"), "url": paper_url},
        "method_summary": extension.get("method_summary"),
        "why_new_executor": extension.get("why_new_executor"),
        "resource_class": extension.get("resource_class"),
        "program": program,
        "contract": contract,
        "errors": errors,
        "reviewed_at": _utc_now(),
        "generated_code_executed": False,
        "arbitrary_python_allowed": False,
    }

    scaffold = workspace / "executor-incubator" / (slug or "invalid-extension")
    scaffold.mkdir(parents=True, exist_ok=True)
    (scaffold / "manifest.json").write_text(
        json.dumps(review, indent=2, sort_keys=True), encoding="utf-8"
    )
    (scaffold / "executor.py").write_text(
        "\"\"\"Engine-generated wrapper; no model-generated Python is executed.\"\"\"\n"
        "from backend.kuailab.paper_executor import build_paper_signal_scores\n\n"
        f"PROGRAM = {program!r}\n\n"
        "def score(train_rows, prediction_rows, data_dir, **kwargs):\n"
        "    return build_paper_signal_scores(train_rows, prediction_rows, data_dir, PROGRAM, **kwargs)\n",
        encoding="utf-8",
    )
    (scaffold / "test_contract.py").write_text(
        "\"\"\"Contract requirements enforced before registry admission.\"\"\"\n"
        f"REQUIRED_TESTS = {sorted(REQUIRED_CONTRACT_TESTS)!r}\n"
        "# See manifest.json for the immutable synthetic contract-test results.\n",
        encoding="utf-8",
    )

    if status == "approved":
        registry_root.mkdir(parents=True, exist_ok=True)
        registry_path = registry_root / f"{slug}.json"
        if registry_path.exists():
            existing = json.loads(registry_path.read_text(encoding="utf-8"))
            if existing.get("program") != program or existing.get("paper", {}).get("url") != paper_url:
                review["status"] = "rejected"
                review["errors"].append("slug already belongs to a different approved executor")
                (scaffold / "manifest.json").write_text(
                    json.dumps(review, indent=2, sort_keys=True), encoding="utf-8"
                )
                return review
        registry_path.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    return review


def load_executor_registry(registry_root: Path) -> list[dict[str, Any]]:
    if not registry_root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(registry_root.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") == "approved":
            records.append(record)
    return records


def require_registered_program(registry_root: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    slug = str(parameters.get("paper_executor_slug", ""))
    path = registry_root / f"{slug}.json"
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,47}", slug) or not path.exists():
        raise ValueError(f"paper executor {slug!r} is not in the approved registry")
    record = json.loads(path.read_text(encoding="utf-8"))
    requested = {
        "paper_executor_slug": slug,
        "paper_signals": list(parameters.get("paper_signals", [])),
        "paper_signal_weights": list(parameters.get("paper_signal_weights", [])),
        "paper_smoothing": float(parameters.get("paper_smoothing", 8.0)),
        "paper_item_smoothing": float(parameters.get("paper_item_smoothing", 20.0)),
        "paper_blend_weight": float(parameters.get("paper_blend_weight", 0.01)),
    }
    if record.get("status") != "approved" or record.get("program") != requested:
        raise ValueError("paper executor parameters differ from the exact reviewed registry program")
    return record
