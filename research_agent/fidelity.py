from __future__ import annotations

import copy
import math
from typing import Any


DEFAULT_RUNGS = [
    {
        "id": "smoke",
        "label": "Smoke screen",
        "train_fraction": 0.08,
        "validation_fraction": 0.20,
        "seeds": 1,
        "max_seconds": 90,
    },
    {
        "id": "screen",
        "label": "Directional screen",
        "train_fraction": 0.35,
        "validation_fraction": 0.50,
        "seeds": 1,
        "max_seconds": 240,
    },
    {
        "id": "confirm",
        "label": "Full confirmation",
        "train_fraction": 1.0,
        "validation_fraction": 1.0,
        "seeds": 3,
        "max_seconds": 900,
    },
]


class MultiFidelityPolicy:
    """Budget policy for cheap screening followed by full confirmation.

    Proxy rungs may reject weak ideas, but only the final rung may promote the
    champion. This prevents a fast, noisy subset score becoming the reported
    competition result.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        settings = config or {}
        self.enabled = bool(settings.get("enabled", True))
        self.reduction_factor = max(2, int(settings.get("reduction_factor", 3)))
        self.min_proxy_gain = float(settings.get("min_proxy_gain", -0.003))
        self.rungs = copy.deepcopy(settings.get("rungs") or DEFAULT_RUNGS)
        self._validate()

    def _validate(self) -> None:
        if not self.rungs:
            raise ValueError("At least one fidelity rung is required")
        previous_train = previous_validation = 0.0
        ids: set[str] = set()
        for index, rung in enumerate(self.rungs):
            rung_id = str(rung.get("id", "")).strip()
            if not rung_id or rung_id in ids:
                raise ValueError("Fidelity rung IDs must be unique and non-empty")
            ids.add(rung_id)
            train_fraction = float(rung["train_fraction"])
            validation_fraction = float(rung["validation_fraction"])
            if not previous_train < train_fraction <= 1.0:
                raise ValueError("Training fractions must increase and end at or below 1")
            if not previous_validation < validation_fraction <= 1.0:
                raise ValueError("Validation fractions must increase and end at or below 1")
            if int(rung["seeds"]) < 1 or int(rung["max_seconds"]) < 1:
                raise ValueError("Fidelity seeds and timeouts must be positive")
            rung["index"] = index
            rung["train_fraction"] = train_fraction
            rung["validation_fraction"] = validation_fraction
            rung["seeds"] = int(rung["seeds"])
            rung["max_seconds"] = int(rung["max_seconds"])
            rung["can_promote_champion"] = index == len(self.rungs) - 1
            previous_train, previous_validation = train_fraction, validation_fraction

    def initial_rung(self) -> dict[str, Any]:
        return copy.deepcopy(self.rungs[0] if self.enabled else self.rungs[-1])

    def next_rung(self, current_id: str) -> dict[str, Any] | None:
        for index, rung in enumerate(self.rungs):
            if rung["id"] == current_id:
                return copy.deepcopy(self.rungs[index + 1]) if index + 1 < len(self.rungs) else None
        raise ValueError(f"Unknown fidelity rung: {current_id}")

    def should_promote(self, result: dict[str, Any], baseline_primary: float) -> bool:
        if result.get("status") != "completed":
            return False
        score = result.get("metrics", {}).get("primary")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            return False
        return float(score) - baseline_primary >= self.min_proxy_gain

    def acquisition_score(
        self,
        expected_gain: float,
        risk: float,
        estimated_minutes: float,
        novelty: float = 1.0,
    ) -> float:
        """Anytime utility: reward plausible gain and novelty per unit time."""

        minutes = max(0.1, float(estimated_minutes))
        efficiency = float(expected_gain) / math.sqrt(minutes)
        return round(2.2 * efficiency + 0.12 * novelty - 0.22 * float(risk), 6)

    def public_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "name": "multi-fidelity successive halving" if self.enabled else "full-fidelity only",
            "reduction_factor": self.reduction_factor,
            "promotion_rule": (
                f"retain roughly 1 in {self.reduction_factor}; proxy gain must be at least "
                f"{self.min_proxy_gain:+.4f}; only the final rung may replace the champion"
            ),
            "rungs": copy.deepcopy(self.rungs),
        }
