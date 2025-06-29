"""Profiler for XCS - following Page's "measure everything" principle.

Hidden from users but provides insights when requested.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from ember.xcs._internal.parallelism import GraphParallelismAnalysis


@dataclass
class FunctionStats:
    """Statistics for a single function."""

    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    avg_parallelism_speedup: float = 1.0
    cache_hits: int = 0
    cache_misses: int = 0

    def record_execution(self, time_ms: float, speedup: float = 1.0):
        """Record an execution."""
        self.call_count += 1
        self.total_time_ms += time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)

        # Update average speedup
        if self.call_count == 1:
            self.avg_parallelism_speedup = speedup
        else:
            # Running average
            self.avg_parallelism_speedup = (
                self.avg_parallelism_speedup * (self.call_count - 1) + speedup
            ) / self.call_count

    @property
    def avg_time_ms(self) -> float:
        """Average execution time."""
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate."""
        total_accesses = self.cache_hits + self.cache_misses
        return self.cache_hits / total_accesses if total_accesses > 0 else 0.0


class Profiler:
    """Collects and reports performance metrics."""

    def __init__(self):
        self.function_stats = defaultdict(lambda: FunctionStats("unknown"))
        self.global_start_time = time.time()

    def record(
        self,
        func_name: str,
        elapsed_ms: float,
        parallelism_info: Optional[GraphParallelismAnalysis] = None,
        graph_size: int = 0,
    ):
        """Record execution metrics."""
        stats = self.function_stats[func_name]
        if stats.name == "unknown":
            stats.name = func_name

        speedup = 1.0
        if parallelism_info:
            speedup = parallelism_info.estimated_speedup

        stats.record_execution(elapsed_ms, speedup)

    def record_cache_hit(self, func_name: str):
        """Record a cache hit."""
        self.function_stats[func_name].cache_hits += 1

    def record_cache_miss(self, func_name: str):
        """Record a cache miss."""
        self.function_stats[func_name].cache_misses += 1

    def get_function_stats(self, func_name: str) -> dict:
        """Get statistics for a specific function."""
        stats = self.function_stats.get(func_name)
        if not stats:
            return {"error": f"No statistics available for {func_name}"}

        return {
            "function": func_name,
            "calls": stats.call_count,
            "total_time_ms": round(stats.total_time_ms, 2),
            "avg_time_ms": round(stats.avg_time_ms, 2),
            "min_time_ms": round(stats.min_time_ms, 2),
            "max_time_ms": round(stats.max_time_ms, 2),
            "avg_speedup": round(stats.avg_parallelism_speedup, 2),
            "cache_hit_rate": round(stats.cache_hit_rate, 2),
            "optimization_suggestion": self._get_optimization_suggestion(stats),
        }

    def get_global_stats(self) -> dict:
        """Get global statistics."""
        total_calls = sum(s.call_count for s in self.function_stats.values())
        total_time_ms = sum(s.total_time_ms for s in self.function_stats.values())

        # Find top functions by time
        top_functions = sorted(
            self.function_stats.items(), key=lambda x: x[1].total_time_ms, reverse=True
        )[:5]

        # Calculate overall metrics
        avg_speedup = 1.0
        if self.function_stats:
            speedups = [s.avg_parallelism_speedup for s in self.function_stats.values()]
            avg_speedup = sum(speedups) / len(speedups)

        return {
            "total_functions": len(self.function_stats),
            "total_calls": total_calls,
            "total_time_ms": round(total_time_ms, 2),
            "avg_speedup": round(avg_speedup, 2),
            "top_functions_by_time": [
                {
                    "name": name,
                    "time_ms": round(stats.total_time_ms, 2),
                    "calls": stats.call_count,
                    "avg_time_ms": round(stats.avg_time_ms, 2),
                }
                for name, stats in top_functions
            ],
            "optimization_potential": self._get_global_optimization_potential(),
        }

    def _get_optimization_suggestion(self, stats: FunctionStats) -> str:
        """Get optimization suggestion for a function."""
        # Smart suggestions based on metrics
        if stats.avg_parallelism_speedup < 1.2 and stats.avg_time_ms > 100:
            return "Consider restructuring for better parallelism"
        elif stats.cache_hit_rate < 0.5 and stats.call_count > 10:
            return "Would benefit from better caching"
        elif stats.max_time_ms > stats.avg_time_ms * 3:
            return "Has high variability - investigate outliers"
        elif stats.avg_time_ms < 1:
            return "Already well optimized"
        else:
            return "Running efficiently"

    def _get_global_optimization_potential(self) -> str:
        """Estimate global optimization potential."""
        if not self.function_stats:
            return "No data yet"

        # Find functions with low speedup but high time
        slow_functions = [
            (name, stats)
            for name, stats in self.function_stats.items()
            if stats.avg_parallelism_speedup < 1.5 and stats.total_time_ms > 1000
        ]

        if slow_functions:
            return f"Could optimize {len(slow_functions)} slow functions for up to 3x speedup"
        else:
            return "System is well optimized"

    def clear_stats(self):
        """Clear all statistics."""
        self.function_stats.clear()

    def get_report(self) -> str:
        """Get human-readable performance report."""
        lines = ["XCS Performance Report", "=" * 40]

        # Global stats
        global_stats = self.get_global_stats()
        lines.append(f"Total functions: {global_stats['total_functions']}")
        lines.append(f"Total calls: {global_stats['total_calls']}")
        lines.append(f"Total time: {global_stats['total_time_ms']}ms")
        lines.append(f"Average speedup: {global_stats['avg_speedup']}x")
        lines.append("")

        # Top functions
        lines.append("Top Functions by Time:")
        for func in global_stats["top_functions_by_time"]:
            lines.append(
                f"  {func['name']}: {func['time_ms']}ms "
                f"({func['calls']} calls, {func['avg_time_ms']}ms avg)"
            )

        lines.append("")
        lines.append(f"Optimization potential: {global_stats['optimization_potential']}")

        return "\n".join(lines)
