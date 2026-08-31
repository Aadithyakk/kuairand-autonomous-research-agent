#!/usr/bin/env python3
"""Fail closed if the judge walkthrough drifts from checked-in evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def require_close(label: str, displayed: float, recorded: float, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(float(displayed), float(recorded), rel_tol=0.0, abs_tol=tolerance):
        raise SystemExit(f"FAIL {label}: showcase={displayed!r}, evidence={recorded!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    showcase = load("public/judge-showcase.json")
    champion = load("results/verified-slate-consensus/summary.json")
    wave = load("results/calibrated-ranking/summary.json")
    campaign = load("results/run-9ecfd2aa09/summary.json")
    worker = load("results/final-model/autonomous-worker-smoke.json")
    manifest = load("results/final-model/manifest.json")

    result = showcase["result"]
    benchmark = showcase["benchmark"]
    verified = champion["champion"]
    require_close("champion primary", result["champion_primary"], verified["primary"])
    require_close("champion GAUC", result["gauc"], verified["gauc"])
    require_close("champion nDCG@5", result["ndcg5"], verified["ndcg5"])
    require_close("baseline primary", result["baseline_primary"], champion["baseline_primary"])
    require_close("absolute gain", result["absolute_gain"], champion["gain_over_baseline"])
    require_close(
        "primary formula",
        result["champion_primary"],
        0.5 * (result["gauc"] + result["ndcg5"]),
        tolerance=5e-8,
    )
    require_close(
        "relative gain",
        result["relative_gain_percent"],
        100.0 * result["absolute_gain"] / result["baseline_primary"],
    )
    if benchmark["validation_rows"] != verified["rows"]:
        raise SystemExit("FAIL validation row count does not match champion evidence")
    if benchmark["validation_users"] != verified["users"]:
        raise SystemExit("FAIL validation user count does not match champion evidence")

    displayed_campaign = showcase["autonomous_campaign"]
    require_close("campaign baseline", displayed_campaign["baseline_primary"], campaign["baseline"]["primary"])
    require_close("campaign converged score", displayed_campaign["converged_primary"], campaign["champion"]["primary"])
    require_close(
        "campaign absolute gain",
        displayed_campaign["absolute_gain"],
        campaign["champion"]["primary"] - campaign["baseline"]["primary"],
    )
    require_close("campaign wall-clock", displayed_campaign["wall_seconds"], campaign["usage"]["wall_seconds"])
    require_close("campaign tokens", displayed_campaign["total_tokens"], campaign["usage"]["total_tokens"])
    require_close("campaign interventions", displayed_campaign["manual_interventions"], campaign["manual_interventions"])
    if displayed_campaign["iterations_used"] != len(campaign["iterations"]):
        raise SystemExit("FAIL autonomous campaign iteration count drifted")
    if displayed_campaign["recovered_failures"] != sum(item["outcome"] == "invalidated" for item in campaign["iterations"]):
        raise SystemExit("FAIL autonomous campaign recovery count drifted")
    if displayed_campaign["stop_reason"] != campaign["stop_reason"]:
        raise SystemExit("FAIL autonomous campaign stop reason drifted")

    expected_weights = {
        "Technical execution": 35,
        "Innovation & insight": 20,
        "Impact & relevance": 20,
        "Feasibility & practicality": 15,
        "Presentation & communication": 10,
    }
    displayed_weights = {item["name"]: item["weight_percent"] for item in showcase["criteria"]}
    if displayed_weights != expected_weights or sum(displayed_weights.values()) != 100:
        raise SystemExit("FAIL judging-criteria weights do not match the Track 2 brief")

    totals = wave["totals"]
    displayed_wave = showcase["experiment_wave"]
    for field in (
        "methods_tested",
        "screen_survivors",
        "confirmed_standalone_improvements",
        "champion_promotions",
        "aggregate_trainer_wall_seconds",
        "aggregate_cpu_hours",
        "gpu_hours",
        "largest_single_process_peak_rss_mb",
    ):
        require_close(f"experiment wave {field}", displayed_wave[field], totals[field])
    rcr = next(item for item in wave["experiments"] if item["id"] == "rcr")
    case = displayed_wave["case_study"]
    require_close("RCR screen gain", case["screen_primary_gain"], rcr["screen_delta"]["primary"])
    require_close("RCR confirmation gain", case["confirmation_primary_gain"], rcr["confirmation_delta"]["primary"])
    require_close("RCR champion residual", case["champion_residual_gain"], rcr["champion_residual_delta"]["primary"])

    displayed_worker = showcase["worker_smoke"]
    usage = worker["resource_usage"]
    require_close("worker train seconds", displayed_worker["train_seconds"], usage["train_seconds"])
    require_close("worker CPU hours", displayed_worker["cpu_hours"], usage["cpu_hours"])
    require_close("worker peak RAM", displayed_worker["peak_rss_mb"], usage["peak_rss_mb"])
    require_close("worker candidate", displayed_worker["candidate_primary"], worker["candidate_metrics"]["primary"])
    require_close("worker retained score", displayed_worker["returned_primary"], worker["worker_output_metrics"]["primary"])

    if any(
        item
        for item in (
            benchmark["hidden_test_accessed"],
            champion["split"]["hidden_test_accessed"],
            wave["protocol"]["hidden_outcomes"] != "No 2022-04-29+ outcome field was parsed by any completed run.",
            worker["protocol"]["hidden_test_accessed"],
            manifest["hidden_test_accessed"],
        )
    ):
        raise SystemExit("FAIL hidden-test integrity claim is inconsistent")

    for artifact in showcase["artifacts"]:
        if not (ROOT / artifact).is_file():
            raise SystemExit(f"FAIL missing showcase artifact: {artifact}")

    score_archive = ROOT / manifest["validation_scores"]
    actual_digest = sha256(score_archive)
    if actual_digest != manifest["validation_scores_sha256"]:
        raise SystemExit("FAIL frozen champion score archive checksum mismatch")

    print("PASS judge walkthrough evidence is internally consistent")
    print(f"  champion  {result['champion_primary']:.9f}")
    print(f"  lift      {result['absolute_gain']:+.9f} ({result['relative_gain_percent']:+.2f}%)")
    print(f"  metric    GAUC {result['gauc']:.6f} · nDCG@5 {result['ndcg5']:.6f}")
    print(f"  evidence  {len(showcase['artifacts'])} artifacts · score archive SHA-256 verified")
    print(
        "  autonomy  "
        f"{displayed_campaign['iterations_used']} iterations · "
        f"{displayed_campaign['total_tokens']:,} tokens · "
        f"{displayed_campaign['manual_interventions']} manual intervention"
    )
    print("  integrity hidden test untouched · synthetic scores excluded")


if __name__ == "__main__":
    main()
