"""Metrics system for efficient instrumentation.

Provides a clean API for recording counters, gauges, histograms, and timers
with minimal overhead and thread-safety.
"""

from .api import (
    counter,
    gauge,
    get_metrics_snapshot,
    get_prometheus_metrics,
    histogram,
    measure_time,
    timed,
)
from .integration import ComponentMetrics
from .metrics import Metrics

__all__ = [
    # Public API functions
    "counter",
    "gauge",
    "histogram",
    "timed",
    "measure_time",
    "get_metrics_snapshot",
    "get_prometheus_metrics",
    # Core implementation classes (for advanced use)
    "Metrics",
    "ComponentMetrics",
]
