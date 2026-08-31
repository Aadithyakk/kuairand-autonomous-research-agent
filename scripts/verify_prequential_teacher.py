#!/usr/bin/env python3
"""Verify the saved completion-safe prequential teacher artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_terminal_gate_search import exact_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = ROOT / "runtime" / "prequential-teacher" / "best_verified_online_scores.npz"
DEFAULT_MANIFEST = ROOT / "results" / "prequential-online-teacher" / "manifest.json"
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_hash = manifest["artifacts"]["score_sha256"]
    measured_hash = sha256(args.scores)
    if measured_hash != expected_hash:
        raise RuntimeError(f"Score hash mismatch: {measured_hash} != {expected_hash}")

    rows = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=["user_id", "date", "long_view"],
        dtype={"user_id": "string"},
    )
    rows = rows.loc[rows["date"] <= 20220428].reset_index(drop=True)
    with np.load(args.scores) as archive:
        scores = np.asarray(archive["selected"], dtype=np.float32)
    if len(scores) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} scores, found {len(scores)}")

    measured = exact_metrics(
        rows["user_id"].astype(str).to_numpy(),
        rows["long_view"].to_numpy(dtype=np.float32),
        scores,
    )
    expected = manifest["validation"]["metrics"]
    expected_normalized = {
        "primary": expected["primary"],
        "gauc": expected["gauc"],
        "ndcg5": expected["ndcg_at_5"],
    }
    drift = {key: measured[key] - expected_normalized[key] for key in measured}
    if max(abs(value) for value in drift.values()) > 1e-7:
        raise RuntimeError(f"Metric drift: {drift}")
    print(json.dumps({"sha256": measured_hash, "metrics": measured, "drift": drift}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
