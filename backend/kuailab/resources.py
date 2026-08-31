from __future__ import annotations

import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Mapping

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on the supported Unix runners
    resource = None


def _finite_number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    converted = float(value)
    return converted if math.isfinite(converted) and converted >= 0 else default


def rss_to_mb(value: float) -> float:
    """Convert ru_maxrss to MiB (bytes on macOS, KiB on Linux)."""
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(max(0.0, float(value)) / divisor, 3)


def normalize_resource_usage(
    raw: Mapping[str, object] | None = None,
    *,
    wall_seconds: float = 0.0,
    cpu_seconds: float = 0.0,
    peak_rss_mb: float = 0.0,
) -> dict:
    """Return one stable, JSON-safe resource record for every benchmark adapter."""
    source = dict(raw or {})
    wall = _finite_number(source.get("wall_seconds"), _finite_number(wall_seconds))
    train = _finite_number(source.get("train_seconds"), wall)
    cpu = _finite_number(source.get("cpu_seconds"), _finite_number(cpu_seconds))
    gpu_count = int(_finite_number(source.get("gpu_count"), 0.0))
    gpu_seconds = _finite_number(source.get("gpu_seconds"), 0.0)
    gpu_hours = _finite_number(source.get("gpu_hours"), gpu_seconds / 3600)
    if not gpu_seconds and gpu_hours:
        gpu_seconds = gpu_hours * 3600
    usage = {
        "wall_seconds": round(wall, 3),
        "train_seconds": round(train, 3),
        "cpu_seconds": round(cpu, 3),
        "cpu_hours": round(_finite_number(source.get("cpu_hours"), cpu / 3600), 6),
        "cpu_utilization_percent": round(
            _finite_number(source.get("cpu_utilization_percent"), (cpu / wall * 100) if wall else 0.0), 2
        ),
        "peak_rss_mb": round(_finite_number(source.get("peak_rss_mb"), _finite_number(peak_rss_mb)), 3),
        "gpu_count": gpu_count,
        "gpu_seconds": round(gpu_seconds, 3),
        "gpu_hours": round(gpu_hours, 6),
        "peak_gpu_memory_mb": round(_finite_number(source.get("peak_gpu_memory_mb"), 0.0), 3),
        "device": str(source.get("device") or ("gpu" if gpu_count or gpu_hours else "cpu")),
    }
    hardware = source.get("hardware")
    if isinstance(hardware, Mapping):
        usage["hardware"] = {str(key): value for key, value in hardware.items()}
    return usage


def empty_campaign_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "wall_seconds": 0.0,
        "train_seconds": 0.0,
        "cpu_seconds": 0.0,
        "cpu_hours": 0.0,
        "gpu_hours": 0.0,
        "peak_rss_mb": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "experiments_measured": 0,
    }


def add_resource_usage(campaign_usage: dict, run_usage: Mapping[str, object] | None) -> None:
    if not run_usage:
        return
    normalized = normalize_resource_usage(run_usage)
    for key in ("train_seconds", "cpu_seconds", "cpu_hours", "gpu_hours"):
        campaign_usage[key] = round(_finite_number(campaign_usage.get(key)) + normalized[key], 6)
    for key in ("peak_rss_mb", "peak_gpu_memory_mb"):
        campaign_usage[key] = max(_finite_number(campaign_usage.get(key)), normalized[key])
    campaign_usage["experiments_measured"] = int(campaign_usage.get("experiments_measured", 0)) + 1


def combine_resource_usage(*records: Mapping[str, object] | None) -> dict:
    """Combine retry attempts without hiding compute spent on failed work."""
    normalized = [normalize_resource_usage(record) for record in records if record]
    if not normalized:
        return normalize_resource_usage()
    return normalize_resource_usage({
        "wall_seconds": sum(item["wall_seconds"] for item in normalized),
        "train_seconds": sum(item["train_seconds"] for item in normalized),
        "cpu_seconds": sum(item["cpu_seconds"] for item in normalized),
        "gpu_hours": sum(item["gpu_hours"] for item in normalized),
        "peak_rss_mb": max(item["peak_rss_mb"] for item in normalized),
        "peak_gpu_memory_mb": max(item["peak_gpu_memory_mb"] for item in normalized),
        "gpu_count": max(item["gpu_count"] for item in normalized),
        "device": "gpu" if any(item["gpu_count"] or item["gpu_hours"] for item in normalized) else "cpu",
    })


@dataclass
class ProcessResourceTracker:
    """Low-overhead wall, CPU, and peak-RAM tracking for the trusted runner."""

    _wall_started: float = field(default_factory=time.monotonic)
    _cpu_started: float = field(default_factory=time.process_time)

    def finish(self, *, train_seconds: float | None = None) -> dict:
        wall = time.monotonic() - self._wall_started
        cpu = time.process_time() - self._cpu_started
        peak_rss = 0.0
        if resource is not None:
            peak_rss = rss_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return normalize_resource_usage({
            "wall_seconds": wall,
            "train_seconds": wall if train_seconds is None else train_seconds,
            "cpu_seconds": cpu,
            "peak_rss_mb": peak_rss,
            "gpu_count": 0,
            "gpu_seconds": 0,
            "peak_gpu_memory_mb": 0,
            "device": "cpu",
            "hardware": {
                "logical_cpu_count": os.cpu_count() or 1,
                "architecture": platform.machine() or "unknown",
                "processor": platform.processor() or "unknown",
            },
        })


def child_usage_snapshot() -> tuple[float, float]:
    if resource is None:
        return (0.0, 0.0)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (float(usage.ru_utime + usage.ru_stime), float(usage.ru_maxrss))


def child_usage_delta(before: tuple[float, float], *, wall_seconds: float) -> dict:
    if resource is None:
        return normalize_resource_usage(wall_seconds=wall_seconds)
    after = child_usage_snapshot()
    return normalize_resource_usage(
        wall_seconds=wall_seconds,
        cpu_seconds=max(0.0, after[0] - before[0]),
        peak_rss_mb=rss_to_mb(after[1]),
    )
