from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "results" / "final-model" / "manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def champion_available(project_root: Path = PROJECT_ROOT) -> bool:
    try:
        load_champion_scores(project_root=project_root)
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def load_champion_scores(
    *, project_root: Path = PROJECT_ROOT, expected_rows: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Load the frozen validation champion after verifying its manifest and hash."""
    manifest_path = project_root / "results" / "final-model" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target") != "long_view":
        raise ValueError("Frozen champion target must be long_view")
    if manifest.get("hidden_test_accessed") is not False:
        raise ValueError("Frozen champion manifest must certify that hidden test is untouched")
    relative = manifest.get("validation_scores")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Frozen champion manifest has no validation_scores path")
    score_path = (project_root / relative).resolve()
    if not score_path.is_relative_to(project_root.resolve()):
        raise ValueError("Frozen champion score path escapes the project root")
    expected_hash = manifest.get("validation_scores_sha256")
    if not isinstance(expected_hash, str) or file_sha256(score_path) != expected_hash:
        raise ValueError("Frozen champion score checksum does not match its manifest")
    with np.load(score_path, allow_pickle=False) as archive:
        if "scores" not in archive.files:
            raise KeyError("Frozen champion archive has no 'scores' array")
        scores = np.asarray(archive["scores"], dtype=np.float64).reshape(-1)
    manifest_rows = int(manifest.get("validation_metrics", {}).get("rows", -1))
    required_rows = manifest_rows if expected_rows is None else expected_rows
    if manifest_rows != len(scores) or len(scores) != required_rows:
        raise ValueError(
            f"Frozen champion alignment failed: manifest={manifest_rows}, "
            f"scores={len(scores)}, expected={required_rows}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("Frozen champion contains non-finite scores")
    return scores, manifest


def within_user_rank(users: Sequence[object], values: np.ndarray) -> np.ndarray:
    """Return stable within-user standardized ordinal ranks."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(users) != len(scores):
        raise ValueError(f"User/score alignment failed: {len(users)} != {len(scores)}")
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    output = np.empty(len(scores), dtype=np.float64)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        group = scores[indices]
        order = np.argsort(group, kind="stable")
        ranks = np.empty(len(group), dtype=np.float64)
        ranks[order] = np.arange(len(group), dtype=np.float64)
        output[indices] = (ranks - ranks.mean()) / max(float(ranks.std()), 1e-8)
    return output


def blend_with_champion(
    users: Sequence[object], champion_scores: np.ndarray,
    candidate_scores: np.ndarray, weight: float,
) -> np.ndarray:
    """Blend a freshly trained candidate into the frozen champion ranking."""
    if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
        raise ValueError("champion_blend_weight must be finite")
    if not -0.25 <= float(weight) <= 0.25:
        raise ValueError("champion_blend_weight must be between -0.25 and 0.25")
    champion_rank = within_user_rank(users, champion_scores)
    candidate_rank = within_user_rank(users, candidate_scores)
    blended = champion_rank + float(weight) * (candidate_rank - champion_rank)
    return blended.astype(np.float32)
