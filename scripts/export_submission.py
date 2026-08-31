#!/usr/bin/env python3
"""Export and validate the organizer submission schema without reading outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPLITS = {
    "validation": (20220422, 20220428, 124_909),
    "hidden-test": (20220429, 20220508, 170_588),
}


def load_scores(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise KeyError(f"{path} has no {key!r} array")
            values = archive[key]
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            values = np.asarray([float(row[key]) for row in csv.DictReader(stream)])
    else:
        raise ValueError("Scores must be .npy, .npz, or .csv")
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Scores contain NaN or infinite values")
    return values


def load_identities(path: Path, start: int, end: int) -> list[tuple[str, str]]:
    identities: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"date", "user_id", "video_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Interaction log must contain {sorted(required)}")
        for row in reader:
            date = int(row["date"])
            if start <= date <= end:
                identities.append((row["user_id"], row["video_id"]))
    return identities


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reserve_hidden_test(receipt: Path, confirmed: bool) -> None:
    if not confirmed:
        raise RuntimeError("Hidden-test export requires --confirm-final-hidden-test")
    if receipt.exists():
        raise RuntimeError(f"Hidden-test access receipt already exists: {receipt}")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "accessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "single final hidden-test submission export",
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-key", default="scores")
    parser.add_argument("--split", choices=tuple(SPLITS), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--confirm-final-hidden-test", action="store_true")
    parser.add_argument("--hidden-test-receipt", type=Path, default=Path("runtime/hidden-test-access.json"))
    args = parser.parse_args()

    start, end, expected_rows = SPLITS[args.split]
    scores = load_scores(args.scores, args.score_key)
    if args.split == "hidden-test":
        reserve_hidden_test(args.hidden_test_receipt, args.confirm_final_hidden_test)
    identities = load_identities(args.interactions, start, end)
    if len(identities) != expected_rows:
        raise RuntimeError(f"Unexpected {args.split} rows: {len(identities)} != {expected_rows}")
    if len(scores) != len(identities):
        raise RuntimeError(f"Score alignment failed: {len(scores)} != {len(identities)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="submission-", suffix=".csv", dir=args.output.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["row_id", "user_id", "video_id", "score"])
            for row_id, ((user_id, video_id), score) in enumerate(zip(identities, scores)):
                if not user_id or not video_id or not math.isfinite(float(score)):
                    raise ValueError(f"Invalid output row {row_id}")
                writer.writerow([row_id, user_id, video_id, f"{float(score):.10g}"])
        os.replace(temporary, args.output)
        os.chmod(args.output, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    manifest = {
        "schema": ["row_id", "user_id", "video_id", "score"],
        "split": args.split,
        "date_range": [start, end],
        "rows": len(scores),
        "row_id": {"start": 0, "end": len(scores) - 1, "strictly_increasing": True},
        "finite_scores": True,
        "submission_sha256": sha256(args.output),
        "hidden_test_accessed": args.split == "hidden-test",
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
