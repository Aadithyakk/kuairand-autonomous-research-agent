from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("KUAILAB_HOST", "127.0.0.1")
    port: int = _int("KUAILAB_PORT", 8787)
    model: str = os.getenv("KUAILAB_MODEL", "gpt-5.6-sol")
    reasoning_effort: str = os.getenv("KUAILAB_REASONING_EFFORT", "high")
    max_iterations: int = _int("KUAILAB_MAX_ITERATIONS", 50)
    max_hours: float = _float("KUAILAB_MAX_HOURS", 6.0)
    convergence_epsilon: float = _float("KUAILAB_CONVERGENCE_EPSILON", 0.002)
    convergence_patience: int = _int("KUAILAB_CONVERGENCE_PATIENCE", 3)
    stage_delay_seconds: float = _float("KUAILAB_STAGE_DELAY_SECONDS", 0.45)
    run_timeout_seconds: int = _int("KUAILAB_RUN_TIMEOUT_SECONDS", 1200)
    state_dir: Path = Path(os.getenv("KUAILAB_STATE_DIR", str(PROJECT_ROOT / "runtime"))).resolve()
    dataset_path: str = os.getenv("KUAIRAND_DATA_PATH", "")
    experiment_command: str = os.getenv("KUAI_EXPERIMENT_COMMAND", "")

    @property
    def api_key_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def public_dict(self) -> dict:
        data = asdict(self)
        data["state_dir"] = str(self.state_dir)
        data["api_key_available"] = self.api_key_available
        data["dataset_available"] = bool(self.dataset_path and Path(self.dataset_path).exists())
        data["adapter_available"] = bool(self.experiment_command)
        return data
