"""Speed benchmark helpers: warm-up excluded, hardware logged. Reports measurements only (never fabricated)."""
from __future__ import annotations

import os
import platform
import time
from typing import Callable, Sequence

import numpy as np


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "processor": platform.processor() or "unknown"}
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "not installed"
    try:
        import ultralytics  # type: ignore

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = "not installed"
    return info


def benchmark(fn: Callable[[object], object], items: Sequence[object], warmup: int) -> dict:
    """Time ``fn(item)`` for each item after ``warmup`` untimed calls. Returns ms/frame statistics and FPS."""
    if len(items) <= warmup:
        raise ValueError("need more items than warm-up calls")
    for it in items[:warmup]:
        fn(it)
    times = []
    for it in items[warmup:]:
        t0 = time.perf_counter()
        fn(it)
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    return {"n_timed": len(a), "warmup": warmup, "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "fps_mean": float(1000.0 / a.mean())}
