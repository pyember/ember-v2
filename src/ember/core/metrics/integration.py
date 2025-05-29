import time
from typing import Dict, Optional

# Import the core Metrics facade
from .metrics import Metrics


def measure_time(name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to measure function execution time using the global Metrics instance.

    Records the duration in milliseconds to a histogram named '{name}_duration_ms'.

    Args:
        name: Base name for the metric.
        tags: Optional dictionary of tags to attach to the metric.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get the singleton Metrics instance
            metrics = Metrics.get()
            # Use the timed context manager for measurement
            with metrics.timed(name, tags):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Placeholder for potential EmberContext integration. This can be fleshed out
# when the context system design is finalized and integrated.
# class EmberContextWithMetrics:
#     """Extends EmberContext with minimal metrics integration."""
#
#     def __init__(self, config=None):
#         # Initialize base context
#         self._metrics = Metrics.get()
#         # Component metrics cache for efficiency
#         self._component_metrics = {}
#
#     def get_metrics(self, component: str) -> 'ComponentMetrics':
#         """Get metrics helper for a specific component."""
#         if component not in self._component_metrics:
#             self._component_metrics[component] = ComponentMetrics(self._metrics, component)
#         return self._component_metrics[component]


class ComponentMetrics:
    """Provides a clean, component-specific interface for recording metrics.

    This helps organize metrics by prefixing them with the component name
    and applying common tags automatically.
    """

    def __init__(
        self,
        metrics_facade: Metrics,
        component_name: str,
        base_tags: Optional[Dict[str, str]] = None):
        """Initialize ComponentMetrics.

        Args:
            metrics_facade: The core Metrics instance to use.
            component_name: The name of the component (e.g., 'data_loader', 'model_server').
            base_tags: Optional tags to apply to all metrics from this component.
        """
        self._metrics = metrics_facade
        self._component_name = component_name
        self._base_tags = base_tags or {}

    def _prepare_tags(
        self, operation_tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Merge base tags with operation-specific tags."""
        merged_tags = self._base_tags.copy()
        if operation_tags:
            merged_tags.update(operation_tags)
        return merged_tags

    def count(
        self, operation: str, value: int = 1, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Count operation occurrences for this component.

        Metric name: '{component_name}.operations_total'
        Tags: Includes base tags + operation tag + provided tags.
        """
        metric_name = f"{self._component_name}.operations_total"
        operation_tags = {"operation": operation}
        if tags:
            operation_tags.update(tags)
        final_tags = self._prepare_tags(operation_tags)
        self._metrics.counter(metric_name, value, final_tags)

    def time(
        self, operation: str, value_ms: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record operation time (in ms) for this component.

        Metric name: '{component_name}.duration_ms'
        Tags: Includes base tags + operation tag + provided tags.
        """
        metric_name = f"{self._component_name}.duration_ms"
        operation_tags = {"operation": operation}
        if tags:
            operation_tags.update(tags)
        final_tags = self._prepare_tags(operation_tags)
        self._metrics.histogram(metric_name, value_ms, final_tags)

    def gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a gauge value specific to this component.

        Metric name: '{component_name}.{name}'
        Tags: Includes base tags + provided tags.
        """
        metric_name = f"{self._component_name}.{name}"
        final_tags = self._prepare_tags(tags)
        self._metrics.gauge(metric_name, value, final_tags)

    def timed(
        self, operation: str, tags: Optional[Dict[str, str]] = None
    ) -> "ComponentTimerContext":
        """Context manager for timing operations within this component.

        Records duration to '{component_name}.duration_ms' histogram.
        """
        # Return the specialized context manager for components
        return ComponentTimerContext(self, operation, tags)


# Helper context manager for ComponentMetrics.timed
class ComponentTimerContext:
    """Context manager helper for ComponentMetrics.timed."""

    __slots__ = ("_component_metrics", "_operation", "_tags", "_start_time")

    def __init__(
        self,
        component_metrics: ComponentMetrics,
        operation: str,
        tags: Optional[Dict[str, str]]):
        self._component_metrics = component_metrics
        self._operation = operation
        self._tags = tags
        self._start_time = time.perf_counter()

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        # Use the ComponentMetrics instance to record the time
        self._component_metrics.time(self._operation, duration_ms, self._tags)
        # Don't suppress exceptions
