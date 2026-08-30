from __future__ import annotations

import signal
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runner_python() -> Path:
    candidates = [
        ROOT / ".venv" / "bin" / "python3",
        Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "bin" / "python3",
        Path(sys.executable),
    ]
    numpy_candidates: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        has_numpy = subprocess.run(
            [str(candidate), "-c", "import numpy"], capture_output=True,
        ).returncode == 0
        if not has_numpy:
            continue
        numpy_candidates.append(candidate)
        has_deep_runtime = subprocess.run(
            [
                str(candidate), "-c",
                "import torch; parts=torch.__version__.split('+')[0].split('.'); "
                "assert tuple(map(int, parts[:2])) >= (2, 1)",
            ],
            capture_output=True,
        ).returncode == 0
        if has_deep_runtime:
            return candidate
    return numpy_candidates[0] if numpy_candidates else Path(sys.executable)


def main() -> int:
    environment = os.environ.copy()
    bundled_data = ROOT / "external" / "KuaiRand-Pure" / "data"
    bundled_starter = ROOT / "external" / "kuairand-starter-kit"
    if bundled_data.exists() and bundled_starter.exists():
        environment.setdefault("KUAIRAND_DATA_PATH", str(bundled_data))
        environment.setdefault("KUAI_STARTER_KIT_DIR", str(bundled_starter))
        environment.setdefault("KUAI_EXPERIMENT_COMMAND", shlex.join([str(runner_python()), str(ROOT / "scripts" / "kuairand_runner.py")]))
    processes = [
        subprocess.Popen([sys.executable, "-m", "backend.kuailab.server"], cwd=ROOT, env=environment),
        subprocess.Popen(["npm", "run", "dashboard"], cwd=ROOT, env=environment),
    ]

    def stop(*_args) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.25)
    finally:
        stop()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return next((process.returncode for process in processes if process.returncode), 0)


if __name__ == "__main__":
    raise SystemExit(main())
