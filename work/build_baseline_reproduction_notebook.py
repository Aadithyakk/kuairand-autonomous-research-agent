"""Build Notebook 01: exact KuaiRand organizer baseline reproduction."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "01_official_baseline_reproduction.ipynb"
STARTER_ZIP = Path(os.environ.get("KUAIRAND_STARTER_ZIP", Path.home() / "Downloads/kuairand-starter-kit.zip"))


def lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def main() -> None:
    payload = STARTER_ZIP.read_bytes()
    starter_b64 = base64.b64encode(payload).decode("ascii")
    starter_sha = hashlib.sha256(payload).hexdigest()

    cells = [
        md(
            """
# Notebook 01 — Official KuaiRand-Pure Baseline Reproduction

This notebook performs **Iteration 0** of the Autonomous ML Research Agent:

1. restores and verifies the organizer-provided Starter Kit;
2. discovers the attached KuaiRand-Pure data;
3. creates a development-only view containing **train + validation dates only**;
4. runs the organizer's random, popularity, and FM implementations unchanged;
5. verifies that the five-seed FM validation mean reproduces the published baseline;
6. writes a machine-readable audit log for the future autonomous controller.

**Compute:** CPU only. A T4 is unnecessary. Expected runtime is roughly 3–5 minutes for five FM seeds.

**Important contract note:** the prose brief currently conflicts with the executable Starter Kit. This notebook reproduces the executable reference exactly: `long_view`, `GAUC`, `nDCG@5`, and their mean. It does not settle which contract organizers will use for final judging.

### What to attach to Kaggle

Attach **KuaiRand-Pure** as either:

- the extracted dataset containing `log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv`, and `video_features_basic_pure.csv`; or
- the official `KuaiRand-Pure.tar.gz` archive.

The Starter Kit is embedded in this notebook and checked byte-for-byte.
"""
        ),
        code(
            f'''# Configuration — keep these values unchanged for the official reproduction.
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/kuairand_iteration_000")
DATA_DIR_OVERRIDE = None  # Example: "/kaggle/input/kuairand-pure/KuaiRand-Pure/data"

SEEDS = [0, 1, 2, 3, 4]
EXPECTED = {{
    "random_valid_primary": 0.4834,
    "pop_valid_primary": 0.5807,
    "fm_valid": {{"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}},
}}
RANDOM_TOLERANCE = 0.0020
POP_TOLERANCE = 0.0010
FM_MEAN_TOLERANCE = 0.0015

STARTER_SHA256 = "{starter_sha}"
WORK.mkdir(parents=True, exist_ok=True)
print("Working directory:", WORK)
'''
        ),
        md(
            """
## 1. Restore the exact organizer Starter Kit

The embedded ZIP is the organizer-provided archive. Its SHA-256 digest is verified before extraction. We will import its `data.py`, `evaluate.py`, and `baseline.py` without editing them.
"""
        ),
        code(
            f'''import base64, hashlib, zipfile

STARTER_B64 = """{starter_b64}"""
starter_zip = WORK / "kuairand-starter-kit.zip"
starter_zip.write_bytes(base64.b64decode(STARTER_B64))

actual_sha = hashlib.sha256(starter_zip.read_bytes()).hexdigest()
assert actual_sha == STARTER_SHA256, (actual_sha, STARTER_SHA256)

with zipfile.ZipFile(starter_zip) as zf:
    names = zf.namelist()
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
    zf.extractall(WORK)

STARTER_DIR = WORK / "kuairand-starter-kit"
required_kit_files = {{"baseline.py", "data.py", "evaluate.py", "baseline_scores.json"}}
assert required_kit_files.issubset({{p.name for p in STARTER_DIR.iterdir()}})
print("Starter Kit SHA-256:", actual_sha)
print("Starter Kit files:", sorted(p.name for p in STARTER_DIR.iterdir()))
'''
        ),
        md(
            """
## 2. Locate or extract KuaiRand-Pure

The discovery cell accepts either extracted CSV files or the official tarball. It does not download anything and will fail with an actionable message if the Kaggle dataset is missing.
"""
        ),
        code(
            '''import os, tarfile

REQUIRED_DATA_FILES = {
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "video_features_basic_pure.csv",
}

def has_required_files(folder: Path) -> bool:
    return folder.is_dir() and REQUIRED_DATA_FILES.issubset({p.name for p in folder.iterdir()})

def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        root = destination.resolve()
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tf.extractall(destination)

if DATA_DIR_OVERRIDE:
    source_data_dir = Path(DATA_DIR_OVERRIDE)
else:
    candidates = []
    if KAGGLE_INPUT.exists():
        for marker in KAGGLE_INPUT.rglob("video_features_basic_pure.csv"):
            if has_required_files(marker.parent):
                candidates.append(marker.parent)
    candidates = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))
    if candidates:
        source_data_dir = candidates[0]
    else:
        archives = sorted(KAGGLE_INPUT.rglob("KuaiRand-Pure.tar.gz")) if KAGGLE_INPUT.exists() else []
        if not archives:
            raise FileNotFoundError(
                "KuaiRand-Pure was not found. Attach an extracted KuaiRand-Pure Kaggle dataset "
                "or the official KuaiRand-Pure.tar.gz, then rerun this cell."
            )
        extract_root = WORK / "source_extract"
        safe_extract_tar(archives[0], extract_root)
        matches = [p.parent for p in extract_root.rglob("video_features_basic_pure.csv") if has_required_files(p.parent)]
        if not matches:
            raise FileNotFoundError(f"Required CSV files were not found inside {archives[0]}")
        source_data_dir = sorted(matches, key=lambda p: (len(p.parts), str(p)))[0]

assert has_required_files(source_data_dir), source_data_dir
print("Source data directory:", source_data_dir)
for name in sorted(REQUIRED_DATA_FILES):
    path = source_data_dir / name
    print(f"  {name}: {path.stat().st_size / 1_000_000:.1f} MB")
'''
        ),
        md(
            """
## 3. Build a train + validation–only development view

The public archive may contain dates after 28 April. This cell filters the second log using **only the `date` column** and excludes all later rows before the organizer loader runs. No hidden/test outcome is evaluated or reported.
"""
        ),
        code(
            '''import csv, shutil

DEV_DATA = WORK / "dev_data"
DEV_DATA.mkdir(parents=True, exist_ok=True)

shutil.copy2(
    source_data_dir / "log_standard_4_08_to_4_21_pure.csv",
    DEV_DATA / "log_standard_4_08_to_4_21_pure.csv",
)
shutil.copy2(
    source_data_dir / "video_features_basic_pure.csv",
    DEV_DATA / "video_features_basic_pure.csv",
)

late_source = source_data_dir / "log_standard_4_22_to_5_08_pure.csv"
late_target = DEV_DATA / "log_standard_4_22_to_5_08_pure.csv"
kept = excluded = 0
with late_source.open(newline="") as src, late_target.open("w", newline="") as dst:
    reader = csv.reader(src)
    writer = csv.writer(dst)
    header = next(reader)
    writer.writerow(header)
    date_index = header.index("date")
    for row in reader:
        if int(row[date_index]) <= 20220428:
            writer.writerow(row)
            kept += 1
        else:
            excluded += 1

assert kept == 124_909, f"Expected 124,909 validation rows, found {kept:,}"
print(f"Validation rows retained: {kept:,}")
print(f"Post-validation rows excluded without scoring: {excluded:,}")
'''
        ),
        md(
            """
## 4. Import the untouched organizer implementation and verify the split

`run_fm` expects a `test` split even when we only need validation. We therefore provide a second reference to the validation rows in that slot. The returned `test` value is deliberately ignored; this only satisfies the unmodified function interface.
"""
        ),
        code(
            '''import importlib, json, sys

sys.path.insert(0, str(STARTER_DIR))
data_module = importlib.import_module("data")
baseline_module = importlib.import_module("baseline")
evaluate_module = importlib.import_module("evaluate")

dev_splits = data_module.load(str(DEV_DATA))
assert len(dev_splits["train"]) == 1_141_112
assert len(dev_splits["valid"]) == 124_909
assert len(dev_splits["test"]) == 0

# Compatibility slot for the untouched organizer run_fm implementation.
dev_splits["test"] = dev_splits["valid"]

published = json.loads((STARTER_DIR / "baseline_scores.json").read_text())
assert published["label"] == "long_view"
assert published["metrics"] == ["GAUC", "nDCG@5"]
assert published["scores"]["fm_official"]["valid"] == EXPECTED["fm_valid"]

print("Development split verified:", {"train": 1_141_112, "valid": 124_909})
print("Fields:", data_module.FIELDS)
print("Label:", data_module.LABEL)
print("Published FM validation target:", EXPECTED["fm_valid"])
'''
        ),
        md(
            """
## 5. Evaluator self-checks

Before spending time on FM training, reproduce the published random and popularity validation rungs. If either fails, stop and repair the data/evaluation harness.
"""
        ),
        code(
            '''import numpy as np

random_runs = [baseline_module.run_random(dev_splits, seed=s)["valid"] for s in SEEDS]
random_mean = float(np.mean([r["primary"] for r in random_runs]))
pop_result = baseline_module.run_pop(dev_splits)["valid"]

print("Random five-seed validation primary mean:", round(random_mean, 6))
print("Published random target:", EXPECTED["random_valid_primary"])
print("Popularity validation:", pop_result)

assert abs(random_mean - EXPECTED["random_valid_primary"]) <= RANDOM_TOLERANCE, (
    "Random rung mismatch", random_mean, EXPECTED["random_valid_primary"]
)
assert abs(pop_result["primary"] - EXPECTED["pop_valid_primary"]) <= POP_TOLERANCE, (
    "Popularity rung mismatch", pop_result["primary"], EXPECTED["pop_valid_primary"]
)
print("PASS — evaluator and data alignment checks succeeded.")
'''
        ),
        md(
            """
## 6. Reproduce the official FM baseline over five seeds

This calls the organizer's unchanged `run_fm` with its published configuration: `k=16`, `lr=0.001`, batch size `8192`, at most `40` epochs, patience `4`.
"""
        ),
        code(
            '''import time

fm_runs = []
for seed in SEEDS:
    print(f"\\n===== OFFICIAL FM — SEED {seed} =====")
    started = time.time()
    result = baseline_module.run_fm(
        dev_splits,
        k=16,
        lr=0.001,
        epochs=40,
        bs=8192,
        patience=4,
        seed=seed,
        verbose=True,
    )["valid"]
    # The organizer evaluator returns NumPy scalar types. Convert them here so
    # the audit record is portable JSON rather than failing after training.
    serialized_result = {
        name: int(value) if name in {"users", "rows"} else float(value)
        for name, value in result.items()
    }
    fm_runs.append({
        "seed": int(seed),
        **serialized_result,
        "elapsed_seconds": float(time.time() - started),
    })

print("\\nCompleted", len(fm_runs), "official FM runs.")
'''
        ),
        md(
            """
## 7. Acceptance test and Iteration 0 artifacts

The published number is compared with the five-seed mean. A tolerance is necessary because the published score is rounded and floating-point execution can vary slightly by platform. Failure raises an exception and prevents a false success claim.
"""
        ),
        code(
            '''from datetime import datetime, timezone

metric_names = ["GAUC", "nDCG@5", "primary"]
fm_mean = {m: float(np.mean([r[m] for r in fm_runs])) for m in metric_names}
fm_std = {m: float(np.std([r[m] for r in fm_runs])) for m in metric_names}
target = EXPECTED["fm_valid"]
deltas = {m: fm_mean[m] - target[m] for m in metric_names}

print("Five-seed mean:", {k: round(v, 6) for k, v in fm_mean.items()})
print("Five-seed std: ", {k: round(v, 6) for k, v in fm_std.items()})
print("Published target:", target)
print("Delta:           ", {k: round(v, 6) for k, v in deltas.items()})

baseline_passed = abs(deltas["primary"]) <= FM_MEAN_TOLERANCE
assert baseline_passed, (
    f"Official baseline was NOT reproduced: mean primary={fm_mean['primary']:.6f}, "
    f"target={target['primary']:.6f}, tolerance={FM_MEAN_TOLERANCE:.6f}"
)

run_log = {
    "iteration": 0,
    "stage": "official_baseline_reproduction",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "hypothesis": "The untouched organizer FM pipeline should reproduce its published validation score.",
    "contract_source": "organizer executable starter kit",
    "task": "within-user ranking over logged impressions",
    "label": "long_view",
    "metrics": ["GAUC", "nDCG@5"],
    "primary": "mean(GAUC, nDCG@5)",
    "data_policy": "train and validation only; dates after 20220428 excluded before loading",
    "starter_zip_sha256": STARTER_SHA256,
    "code_diff": [],
    "config": {
        "model": "FM", "k": 16, "lr": 0.001, "batch": 8192,
        "max_epochs": 40, "patience": 4, "seeds": SEEDS,
        "fields": data_module.FIELDS,
    },
    "published_validation": target,
    "seed_results": fm_runs,
    "validation_mean": fm_mean,
    "validation_std": fm_std,
    "validation_delta": deltas,
    "acceptance_tolerance_primary": FM_MEAN_TOLERANCE,
    "status": "passed",
    "errors": [],
    "recoveries": [],
    "manual_interventions_during_run": 0,
    "gpu_hours": 0.0,
}

log_path = WORK / "iteration_000_baseline.json"
log_path.write_text(json.dumps(run_log, indent=2))

import csv as csv_module
seed_csv = WORK / "baseline_seed_results.csv"
with seed_csv.open("w", newline="") as fh:
    seed_fields = ["seed", "GAUC", "nDCG@5", "primary", "users", "rows", "elapsed_seconds"]
    writer = csv_module.DictWriter(fh, fieldnames=seed_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(fm_runs)

print("PASS — official FM validation baseline reproduced.")
print("Iteration log:", log_path)
print("Seed results: ", seed_csv)
'''
        ),
        md(
            """
## 8. Package the evidence

Download the resulting ZIP and keep it as the immutable Iteration 0 evidence. Notebook 02/the autonomous controller should read `iteration_000_baseline.json` before proposing its first improvement.
"""
        ),
        code(
            '''import zipfile

artifact_zip = Path("/kaggle/working/iteration_000_baseline_artifacts.zip")
with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(WORK / "iteration_000_baseline.json", arcname="iteration_000_baseline.json")
    zf.write(WORK / "baseline_seed_results.csv", arcname="baseline_seed_results.csv")
    zf.write(STARTER_DIR / "baseline_scores.json", arcname="organizer_baseline_scores.json")

print("Ready to download:", artifact_zip)
'''
        ),
        md(
            """
### Completion condition

This notebook is complete only when the acceptance cell prints:

`PASS — official FM validation baseline reproduced.`

That establishes the required baseline. It does **not** yet constitute the full autonomous research agent; it becomes the controller's Iteration 0 input.
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"accelerator": "none", "dataSources": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1))
    print(OUT)


if __name__ == "__main__":
    main()
