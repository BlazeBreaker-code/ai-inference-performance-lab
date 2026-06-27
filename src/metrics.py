from __future__ import annotations

from typing import Iterable

import numpy as np


def summarize_latencies(latencies_ms: Iterable[float], batch_size: int) -> dict:
    values = np.array(list(latencies_ms), dtype=np.float64)

    if values.size == 0:
        return {
            "count": 0,
            "batch_size": batch_size,
            "total_requests": 0,
            "throughput_req_s": 0.0,
            "min_ms": 0.0,
            "mean_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

    total_ms = float(values.sum())
    total_requests = int(values.size * batch_size)
    throughput = total_requests / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return {
        "count": int(values.size),
        "batch_size": batch_size,
        "total_requests": total_requests,
        "throughput_req_s": round(float(throughput), 2),
        "min_ms": round(float(values.min()), 4),
        "mean_ms": round(float(values.mean()), 4),
        "max_ms": round(float(values.max()), 4),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "p99_ms": round(float(np.percentile(values, 99)), 4),
    }
