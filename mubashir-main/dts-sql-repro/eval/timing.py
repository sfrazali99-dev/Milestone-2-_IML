"""
Timing and latency measurement utilities for text-to-SQL system benchmarks.
"""

import time
from typing import List, Dict, Any


class LatencyTracker:
    def __init__(self):
        self.latencies: List[float] = []

    def record(self, duration_seconds: float):
        self.latencies.append(duration_seconds)

    def summary(self) -> Dict[str, float]:
        if not self.latencies:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}

        sorted_lats = sorted(self.latencies)
        count = len(sorted_lats)
        total = sum(sorted_lats)
        mean = total / count
        median = sorted_lats[count // 2] if count % 2 == 1 else (sorted_lats[count // 2 - 1] + sorted_lats[count // 2]) / 2.0
        
        return {
            "count": count,
            "mean": round(mean, 4),
            "median": round(median, 4),
            "min": round(sorted_lats[0], 4),
            "max": round(sorted_lats[-1], 4),
            "total": round(total, 4)
        }
