"""Clean, minimal metrics facade for application use.

Provides a simplified API for the metrics system that's easy to use
with minimal complexity and overhead.
"""

from typing import Any, ContextManager, Dict, Optional, TypeVar

from .metrics import Metrics, measure_time

T = TypeVar("T")


def counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
    """Increment counter.

    Simple interface for counting events.

    Args:
        name: Counter name
        value: Increment amount (default: 1)
        tags: Optional dimension tags
    """
    Metrics.get().counter(name, value, tags)


def gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Set gauge value.

    Simple interface for setting gauge values.

    Args:
        name: Gauge name
        value: Current value
        tags: Optional dimension tags
    """
    Metrics.get().gauge(name, value, tags)


def histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record histogram value.

    Simple interface for recording values in a histogram.

    Args:
        name: Histogram name
        value: Value to record
        tags: Optional dimension tags
    """
    Metrics.get().histogram(name, value, tags)


def timed(name: str, tags: Optional[Dict[str, str]] = None) -> ContextManager[None]:
    """Context manager for timing operations.

    Simple interface for timing code blocks.

    Args:
        name: Timer name
        tags: Optional dimension tags

    Returns:
        Context manager that records duration when exited
    """
    return Metrics.get().timed(name, tags)


def get_metrics_snapshot() -> Dict[str, Any]:
    """Get raw snapshot of all metrics.

    Returns:
        Dict with all metric values
    """
    return Metrics.get().get_snapshot()


def get_prometheus_metrics() -> str:
    """Get all metrics in Prometheus format.

    Returns:
        Prometheus-formatted metrics string
    """
    # Import here to avoid circular imports and keep this optional
    from .exporters.prometheus import PrometheusExporter

    exporter = PrometheusExporter(Metrics.get())
    return exporter.export()


__all__ = [
    "counter",
    "gauge",
    "histogram",
    "timed",
    "measure_time",
    "get_metrics_snapshot",
    "get_prometheus_metrics",
]
