from __future__ import annotations

import json
import os
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .resources import empty_campaign_usage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initial_state(config: dict) -> dict:
    baseline = {"primary": 0.6016, "gauc": 0.6612, "ndcg5": 0.5310}
    return {
        "version": 3,
        "campaign": {
            "id": None,
            "status": "idle",
            "mode": "demo",
            "provider": "demo",
            "started_at": None,
            "ended_at": None,
            "stop_reason": None,
            "steering": None,
        },
        "config": config,
        "current": None,
        "metrics": {"baseline": baseline, "champion": baseline, "delta": 0.0},
        "usage": empty_campaign_usage(),
        "iterations": [{
            "number": 0,
            "title": "Demo FM baseline",
            "status": "baseline",
            "stage": "complete",
            "metrics": baseline,
            "delta": 0.0,
            "accepted": True,
            "duration_seconds": 0.0,
            "provider": "demo",
            "evidence": "synthetic-demo",
        }],
        "events": [{"id": 1, "time": utc_now(), "kind": "system", "title": "Control room ready", "detail": "Start a demo or connect the KuaiRand runner."}],
    }


class StateStore:
    """Thread-safe, atomic JSON state plus append-only JSONL evidence."""

    def __init__(self, directory: Path, config: dict):
        self.directory = directory
        self.path = directory / "state.json"
        self.events_path = directory / "events.jsonl"
        self.lock = threading.RLock()
        directory.mkdir(parents=True, exist_ok=True)
        self._state = self._read() or initial_state(config)
        self._migrate(config)
        self._write()

    def _migrate(self, config: dict) -> None:
        defaults = initial_state(config)
        self._state["version"] = 3
        self._state.setdefault("config", config)
        usage = self._state.setdefault("usage", {})
        for key, value in defaults["usage"].items():
            usage.setdefault(key, value)
        for item in self._state.get("iterations", []):
            item.setdefault("resource_usage", None)

    def _read(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write(self) -> None:
        handle, temp_path = tempfile.mkstemp(prefix="state-", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self._state, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def snapshot(self) -> dict:
        with self.lock:
            return deepcopy(self._state)

    def update(self, mutate: Callable[[dict], None]) -> dict:
        with self.lock:
            mutate(self._state)
            self._write()
            return deepcopy(self._state)

    def reset(self, config: dict) -> dict:
        with self.lock:
            self._state = initial_state(config)
            self._write()
            return deepcopy(self._state)

    def event(self, kind: str, title: str, detail: str, iteration: int | None = None, stage: str | None = None) -> dict:
        with self.lock:
            item = {
                "id": (self._state["events"][-1]["id"] + 1) if self._state["events"] else 1,
                "time": utc_now(),
                "kind": kind,
                "title": title,
                "detail": detail,
                "iteration": iteration,
                "stage": stage,
            }
            self._state["events"].append(item)
            self._state["events"] = self._state["events"][-200:]
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
            self._write()
            return deepcopy(item)
