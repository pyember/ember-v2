"""Metrics integration component.

This module provides a component for metrics collection and reporting
with minimal overhead and thread-safe access.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from .component import Component
from .registry import Registry


class MetricsComponent(Component):
    """Metrics management component.

    This component provides a lightweight interface for collecting
    metrics with minimal overhead and efficient aggregation.

    Features:
    - Counters: Increment/decrement values
    - Gauges: Set absolute values
    - Histograms: Track value distributions
    - Timers: Measure operation duration
    - Tags: Add dimensions to metrics
    """

    def __init__(self, registry: Optional[Registry] = None):
        """Initialize with registry.

        Args:
            registry: Registry to use (current thread's if None)
        """
        super().__init__(registry)
        self._metrics: Dict[str, Dict[str, Any]] = {
            "counters": {},
            "gauges": {},
            "histograms": {},
        }
        self._component_metrics: Dict[str, "ComponentMetrics"] = {}

    def _register(self) -> None:
        """Register in registry as 'metrics'."""
        self._registry.register("metrics", self)

    def counter(
        self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment counter.

        Args:
            name: Counter name
            value: Increment amount (default: 1)
            tags: Optional dimension tags
        """
        self._ensure_initialized()
        key = self._get_key(name, tags)

        with self._lock:
            current = self._metrics["counters"].get(key, 0)
            self._metrics["counters"][key] = current + value

    def gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set gauge value.

        Args:
            name: Gauge name
            value: Current value
            tags: Optional dimension tags
        """
        self._ensure_initialized()
        key = self._get_key(name, tags)

        with self._lock:
            self._metrics["gauges"][key] = value

    def histogram(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record histogram value.

        Args:
            name: Histogram name
            value: Value to record
            tags: Optional dimension tags
        """
        self._ensure_initialized()
        key = self._get_key(name, tags)

        with self._lock:
            if key not in self._metrics["histograms"]:
                self._metrics["histograms"][key] = {
                    "count": 0,
                    "sum": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                }

            hist = self._metrics["histograms"][key]
            hist["count"] += 1
            hist["sum"] += value
            hist["min"] = min(hist["min"], value)
            hist["max"] = max(hist["max"], value)

    @contextmanager
    def timed(self, name: str, tags: Optional[Dict[str, str]] = None) -> Iterator[None]:
        """Context manager for timing operations.

        Usage:
            with metrics.timed("operation_duration"):
                # Code to time

        Args:
            name: Timer name
            tags: Optional dimension tags

        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.histogram(name, duration * 1000, tags)  # Convert to milliseconds

    def get_component_metrics(self, component_name: str) -> "ComponentMetrics":
        """Get component-specific metrics interface.

        This provides a scoped metrics interface that automatically
        adds the component name to all metrics.

        Args:
            component_name: Component identifier

        Returns:
            Component-scoped metrics interface
        """
        self._ensure_initialized()

        # Return cached component metrics if available
        if component_name in self._component_metrics:
            return self._component_metrics[component_name]

        # Create new component metrics
        component_metrics = ComponentMetrics(self, component_name)
        self._component_metrics[component_name] = component_metrics
        return component_metrics

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all collected metrics.

        Returns:
            Dictionary of all metrics by type
        """
        self._ensure_initialized()
        return self._metrics.copy()

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics = {
                "counters": {},
                "gauges": {},
                "histograms": {},
            }

    def _initialize(self) -> None:
        """Initialize metrics system."""
        # Nothing to initialize
        pass

    def _get_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Get unique key for a metric.

        Args:
            name: Metric name
            tags: Optional dimension tags

        Returns:
            Unique key string
        """
        if not tags:
            return name

        # Sort tags for consistent key generation
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"


class ComponentMetrics:
    """Component-scoped metrics interface.

    This class provides a scoped metrics interface that automatically
    adds the component name to all metrics.
    """

    def __init__(self, metrics: MetricsComponent, component_name: str):
        """Initialize with parent metrics and component name.

        Args:
            metrics: Parent metrics component
            component_name: Component identifier
        """
        self._metrics = metrics
        self._component_name = component_name

    def counter(
        self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment counter with component scope.

        Args:
            name: Counter name
            value: Increment amount (default: 1)
            tags: Optional dimension tags
        """
        name = f"{self._component_name}.{name}"
        self._metrics.counter(name, value, tags)

    def gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set gauge value with component scope.

        Args:
            name: Gauge name
            value: Current value
            tags: Optional dimension tags
        """
        name = f"{self._component_name}.{name}"
        self._metrics.gauge(name, value, tags)

    def histogram(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record histogram value with component scope.

        Args:
            name: Histogram name
            value: Value to record
            tags: Optional dimension tags
        """
        name = f"{self._component_name}.{name}"
        self._metrics.histogram(name, value, tags)

    def timed(self, name: str, tags: Optional[Dict[str, str]] = None) -> Iterator[None]:
        """Context manager for timing operations with component scope.

        Args:
            name: Timer name
            tags: Optional dimension tags

        Returns:
            Context manager that times operation
        """
        name = f"{self._component_name}.{name}"
        return self._metrics.timed(name, tags)
