from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORBIDDEN_CANDIDATE_FIELDS = {
    "long_view",
    "play_time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
}


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def score_candidate(artifact: dict[str, Any], candidate: dict[str, Any]) -> float:
    model = artifact["model"]
    weights = model["candidate_weights"]
    logit = float(model["intercept"])
    for index in candidate["categorical_indices"]:
        logit += float(weights.get(str(index), 0.0))
    for value, mean, scale, weight in zip(
        candidate["numeric_values"],
        model["numeric_means"],
        model["numeric_scales"],
        model["numeric_weights"],
    ):
        logit += ((float(value) - float(mean)) / float(scale)) * float(weight)
    return sigmoid(logit)


def predict_slate(
    artifact: dict[str, Any], user_id: str | None = None, limit: int = 10
) -> dict[str, Any]:
    candidates = artifact["candidates"]
    users = artifact["users"]
    if not users:
        raise ValueError("The live-prediction artifact contains no candidate users")
    selected = str(user_id or users[0]["user_id"])
    rows = [row for row in candidates if str(row["user_id"]) == selected]
    if not rows:
        raise ValueError(f"User {selected} is not in the exported April 29 demo cohort")
    safe_limit = max(1, min(int(limit), 50))
    ranked = sorted(
        ({**row, "score": score_candidate(artifact, row)} for row in rows),
        key=lambda row: (-row["score"], row["exposure_index"]),
    )[:safe_limit]
    public_rows = []
    for rank, row in enumerate(ranked, start=1):
        public_rows.append(
            {
                "rank": rank,
                "user_id": str(row["user_id"]),
                "video_id": str(row["video_id"]),
                "author_id": str(row["author_id"]),
                "video_type": row["video_type"],
                "tab": str(row["tab"]),
                "hour": int(row["hour"]),
                "duration_seconds": float(row["duration_seconds"]),
                "score": round(float(row["score"]), 8),
            }
        )
    return {
        "prediction_id": datetime.now(timezone.utc).isoformat(),
        "user_id": selected,
        "target_date": artifact["target"]["date"],
        "model": artifact["model"]["name"],
        "execution": "server",
        "ranking": public_rows,
        "evaluation": artifact["evaluation"],
        "integrity": artifact["integrity"],
    }


class LivePredictor:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self._artifact: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return self.artifact_path.exists()

    def _load(self) -> dict[str, Any]:
        if self._artifact is None:
            artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
            forbidden = FORBIDDEN_CANDIDATE_FIELDS.intersection(
                key for row in artifact.get("candidates", []) for key in row
            )
            if forbidden:
                raise RuntimeError(
                    f"Live artifact contains forbidden outcome fields: {sorted(forbidden)}"
                )
            if artifact.get("integrity", {}).get("target_outcomes_accessed") is not False:
                raise RuntimeError("Live artifact does not attest that target outcomes are sealed")
            self._artifact = artifact
        return self._artifact

    def options(self) -> dict[str, Any]:
        artifact = self._load()
        return {
            "available": True,
            "users": artifact["users"],
            "target": artifact["target"],
            "model": {
                "name": artifact["model"]["name"],
                "kind": artifact["model"]["kind"],
            },
            "evaluation": artifact["evaluation"],
            "integrity": artifact["integrity"],
        }

    def predict(self, user_id: str | None, limit: int = 10) -> dict[str, Any]:
        return predict_slate(self._load(), user_id=user_id, limit=limit)
