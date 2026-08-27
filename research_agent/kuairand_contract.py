from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any


LOG_FILES = (
    "KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv",
    "KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv",
    "KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv",
)
FEATURE_FILES = (
    "KuaiRand-Pure/data/user_features_pure.csv",
    "KuaiRand-Pure/data/video_features_basic_pure.csv",
    "KuaiRand-Pure/data/video_features_statistic_pure.csv",
)
REQUIRED_LOG_COLUMNS = {
    "user_id", "video_id", "date", "hourmin", "time_ms", "long_view",
    "duration_ms", "is_click", "is_like", "is_follow", "is_comment",
    "is_forward", "is_hate", "play_time_ms", "is_rand", "tab",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_header(archive: tarfile.TarFile, name: str) -> list[str]:
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Archive member cannot be read: {name}")
    wrapper = io.TextIOWrapper(member, encoding="utf-8-sig", newline="")
    return next(csv.reader(wrapper))


def validate_kuairand_inputs(dataset_archive: Path, baseline_artifact: Path) -> dict[str, Any]:
    """Validate real benchmark inputs without exposing post-cutoff rows to experiments."""

    checks: dict[str, bool] = {
        "dataset_archive_present": dataset_archive.is_file(),
        "baseline_artifact_present": baseline_artifact.is_file(),
    }
    details: dict[str, Any] = {
        "dataset_archive": str(dataset_archive),
        "baseline_artifact": str(baseline_artifact),
        "public_research_cutoff": 20220428,
        "label": "long_view",
        "metrics": ["GAUC", "nDCG@5"],
        "primary": "mean(GAUC, nDCG@5)",
    }
    errors: list[str] = []

    if checks["dataset_archive_present"]:
        try:
            with tarfile.open(dataset_archive, "r:gz") as archive:
                members = set(archive.getnames())
                missing = sorted(set(LOG_FILES + FEATURE_FILES) - members)
                checks["required_dataset_files_present"] = not missing
                details["missing_dataset_files"] = missing
                headers = {name: _csv_header(archive, name) for name in LOG_FILES if name in members}
                missing_columns = {
                    name: sorted(REQUIRED_LOG_COLUMNS - set(header))
                    for name, header in headers.items()
                }
                checks["log_schema_matches_contract"] = bool(headers) and not any(missing_columns.values())
                checks["label_present_in_logs"] = bool(headers) and all("long_view" in header for header in headers.values())
                details["log_headers"] = headers
                details["missing_log_columns"] = missing_columns
                details["dataset_sha256"] = _sha256(dataset_archive)
        except (OSError, tarfile.TarError, csv.Error, StopIteration, ValueError) as exc:
            errors.append(f"Dataset archive validation failed: {exc}")
    else:
        checks.update(required_dataset_files_present=False, log_schema_matches_contract=False, label_present_in_logs=False)

    if checks["baseline_artifact_present"]:
        try:
            with zipfile.ZipFile(baseline_artifact) as archive:
                names = set(archive.namelist())
                required = {"iteration_000_baseline.json", "organizer_baseline_scores.json"}
                checks["baseline_files_present"] = required.issubset(names)
                run = json.loads(archive.read("iteration_000_baseline.json"))
                organizer = json.loads(archive.read("organizer_baseline_scores.json"))
                expected = organizer["scores"]["fm_official"]["valid"]
                observed = run["validation_mean"]
                tolerance = float(run["acceptance_tolerance_primary"])
                checks["baseline_status_passed"] = run.get("status") == "passed"
                checks["baseline_contract_matches"] = (
                    run.get("label") == "long_view"
                    and run.get("metrics") == ["GAUC", "nDCG@5"]
                    and organizer.get("split", {}).get("valid") == "20220422-20220428"
                )
                checks["baseline_within_tolerance"] = abs(float(observed["primary"]) - float(expected["primary"])) <= tolerance
                checks["baseline_no_manual_intervention"] = run.get("manual_interventions_during_run") == 0
                details["baseline"] = {
                    "observed": observed,
                    "published": expected,
                    "tolerance": tolerance,
                    "seeds": len(run.get("seed_results", [])),
                    "validation_rows": run.get("seed_results", [{}])[0].get("rows"),
                    "validation_users": run.get("seed_results", [{}])[0].get("users"),
                    "artifact_sha256": _sha256(baseline_artifact),
                }
        except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"Baseline artifact validation failed: {exc}")
    else:
        checks.update(
            baseline_files_present=False,
            baseline_status_passed=False,
            baseline_contract_matches=False,
            baseline_within_tolerance=False,
            baseline_no_manual_intervention=False,
        )

    return {
        "status": "passed" if checks and all(checks.values()) and not errors else "failed",
        "checks": checks,
        "details": details,
        "errors": errors,
        "scope": (
            "Readiness validation only: confirms the real KuaiRand inputs and organizer baseline. "
            "It does not claim an autonomous challenger has yet beaten the KuaiRand baseline."
        ),
    }
