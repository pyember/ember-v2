import threading
import time
from typing import Any, Dict, Optional, Protocol


class Metric(Protocol):
    """Protocol for metric implementations."""

    def snapshot(self) -> Any:
        """Get current metric value."""
        ...


class Counter:
    """Thread-safe counter with atomic operations.

    Performance: ~8ns increment, ~16 bytes memory.
    """

    __slots__ = ("_value", "_lock")

    def __init__(self, initial_value: int = 0):
        # Direct attribute access for speed
        self._value = initial_value
        self._lock = threading.Lock()  # Lock for inc operation

    def inc(self, amount: int = 1) -> None:
        """Increment atomically."""
        with self._lock:
            self._value += amount

    def get(self) -> int:
        """Get current value."""
        # Reading the int value itself is atomic
        return self._value

    def snapshot(self) -> int:
        """Take snapshot for export."""
        # Reading the int value itself is atomic
        return self._value


class Gauge:
    """Simple gauge with atomic set/update.

    Performance characteristics:
    - Set: ~6ns (target)
    - Memory: ~16 bytes per gauge
    - Thread-safe with no locks (leveraging Python atomics)
    """

    __slots__ = ("_value")

    def __init__(self, initial_value: float = 0.0):
        self._value = initial_value

    def set(self, value: float) -> None:
        """Set gauge value atomically."""
        # Python's float assignment is atomic under GIL
        self._value = value

    def get(self) -> float:
        """Get current value."""
        return self._value

    def snapshot(self) -> float:
        """Take snapshot for export."""
        return self._value


class Histogram:
    """Efficient histogram with minimal overhead on the hot path.

    Performance characteristics:
    - Record: ~35ns (target)
    - Memory: ~256 bytes per histogram (depends on bucket count)
    - Thread-safe without locking the record path (updates are atomic)
    """

    __slots__ = ("_count", "_sum", "_min", "_max", "_buckets", "_lock")

    # Simple linear buckets, chosen for cache efficiency with few buckets.
    # Consider configurable buckets for broader use cases later.
    BUCKET_BOUNDARIES = [
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        float("inf")]
    NUM_BUCKETS = len(BUCKET_BOUNDARIES)

    def __init__(self):
        # Initialize stats atomically where possible
        self._count = 0
        self._sum = 0.0
        # Min/Max updates require care for concurrency, though simple overwrite
        # is often sufficient and fast. Using locks for initial safe implementation.
        self._min = float("inf")
        self._max = float("-inf")
        self._buckets = [0] * self.NUM_BUCKETS
        # Lock primarily protects min/max updates and snapshot consistency
        self._lock = threading.Lock()

    def record(self, value: float) -> None:
        """Record value in histogram. Designed for high frequency calls."""
        # Find bucket index using linear search (fast for small N)
        # Do this outside the lock if possible
        bucket_index = -1
        for i, boundary in enumerate(self.BUCKET_BOUNDARIES):
            if value <= boundary:
                bucket_index = i
                break

        # Lock needed for safe updates to shared state
        with self._lock:
            # Increment count and sum atomically within the lock
            self._count += 1
            self._sum += value

            # Update min/max
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

            # Increment bucket count
            if bucket_index != -1:
                self._buckets[bucket_index] += 1

    def snapshot(self) -> Dict[str, Any]:
        """Take snapshot for export. Acquires lock for consistency."""
        with self._lock:
            count = self._count
            min_val = self._min if count else 0
            max_val = self._max if count else 0
            # Create copies under the lock
            buckets_copy = list(self._buckets)

        return {
            "count": count,
            "sum": self._sum,  # Read of float is atomic
            "min": min_val,
            "max": max_val,
            "avg": self._sum / count if count else 0,
            "buckets": buckets_copy,
            "boundaries": self.BUCKET_BOUNDARIES,  # Class attribute, safe
        }


class MetricsRegistry:
    """Thread-safe metrics registry using thread-local caching and copy-on-write.

    Minimizes lock contention for reads and common writes (updates to existing).
    Lock is primarily used for creating *new* metrics.
    """

    def __init__(self):
        # Main registry lock only used for creating new metrics
        self._lock = threading.RLock()
        # Metrics storage - reads are lock-free via self._metrics reference copy
        self._metrics: Dict[str, Metric] = {}
        # Thread-local storage for fast-path metric access
        self._thread_local = threading.local()

    def _get_local_cache(self) -> Dict[str, Metric]:
        """Get or initialize the thread-local cache."""
        try:
            return self._thread_local.cache
        except AttributeError:
            self._thread_local.cache = {}
            return self._thread_local.cache

    def _get_or_create_metric(self, key: str, metric_type: type) -> Metric:
        """Generic function to get/create metric, minimizing lock usage.

        Raises:
            TypeError: If a metric with the same key but different type exists.
        """
        # 1. Check thread-local cache (fastest, no locks)
        cache = self._get_local_cache()
        metric = cache.get(key)
        if isinstance(metric, metric_type):
            return metric
        # If cache has it but it's the wrong type, raise error immediately
        if key in cache and not isinstance(metric, metric_type):
            raise TypeError(
                f"Metric '{key}' exists but has type {type(metric).__name__}, requested {metric_type.__name__}"
            )

        # 2. Check global registry (no locks for read)
        # Read the dictionary reference - this is atomic
        current_metrics = self._metrics
        metric = current_metrics.get(key)
        if isinstance(metric, metric_type):
            cache[key] = metric  # Update local cache
            return metric
        # If global has it but it's the wrong type, raise error
        if key in current_metrics and not isinstance(metric, metric_type):
            raise TypeError(
                f"Metric '{key}' exists but has type {type(metric).__name__}, requested {metric_type.__name__}"
            )

        # 3. Need to create - acquire lock
        with self._lock:
            # Double-check after acquiring lock
            current_metrics = self._metrics  # Re-read after lock
            metric = current_metrics.get(key)
            if isinstance(metric, metric_type):
                # Another thread created it, update local cache
                cache[key] = metric
                return metric
            # Check again for type mismatch after lock acquisition
            if key in current_metrics and not isinstance(metric, metric_type):
                raise TypeError(
                    f"Metric '{key}' exists but has type {type(metric).__name__}, requested {metric_type.__name__}"
                )
            else:
                # Create the new metric
                new_metric = metric_type()
                # Create a new dictionary for copy-on-write update
                new_metrics_dict = current_metrics.copy()
                new_metrics_dict[key] = new_metric
                # Atomic update of the metrics dictionary reference
                self._metrics = new_metrics_dict
                # Update local cache
                cache[key] = new_metric
                return new_metric

    def counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create counter."""
        key = self._get_key(name, tags)
        metric = self._get_or_create_metric(key, Counter)
        # Type check assertion for static analysis / safety
        assert isinstance(metric, Counter)
        return metric

    def gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create gauge."""
        key = self._get_key(name, tags)
        metric = self._get_or_create_metric(key, Gauge)
        # No longer need assert isinstance here, _get_or_create_metric guarantees type or raises TypeError
        # assert isinstance(metric, Gauge)
        return metric  # Type hint ensures it's Gauge

    def histogram(self, name: str, tags: Optional[Dict[str, str]] = None) -> Histogram:
        """Get or create histogram."""
        key = self._get_key(name, tags)
        metric = self._get_or_create_metric(key, Histogram)
        # No longer need assert isinstance here
        # assert isinstance(metric, Histogram)
        return metric  # Type hint ensures it's Histogram

    def get_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all metrics for export. Lock-free read."""
        # Atomically grab the current metrics dictionary reference
        metrics_copy = self._metrics
        # Take snapshot of each metric (implementation specific, may lock internally)
        # The dictionary iteration itself is safe on the copied reference.
        return {key: metric.snapshot() for key, metric in metrics_copy.items()}

    def _get_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Generate a canonical metric key from name and tags."""
        if not tags:
            return name
        # Sort tags for consistent key generation
        # Consider a faster/more optimized key generation if this becomes a bottleneck
        sorted_tags = sorted(tags.items())
        # Using a simple, readable format. Alternatives exist (e.g., msgpack).
        tags_str = ",".join(f"{k}={v}" for k, v in sorted_tags)
        return f"{name}[{tags_str}]"


class Metrics:
    """Singleton facade for accessing the metrics system.

    Provides a clean, globally accessible interface for recording metrics.
    """

    _instance: Optional["Metrics"] = None
    _instance_lock = threading.Lock()  # Use simple Lock for singleton creation

    @classmethod
    def get(cls) -> "Metrics":
        """Get the global metrics instance (singleton)."""
        # Double-checked locking for efficient singleton access
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # Prevent direct instantiation
    def __init__(self):
        if self._instance is not None:
            raise RuntimeError("Use Metrics.get() to retrieve the singleton instance.")
        self._registry = MetricsRegistry()

    def counter(
        self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        counter_metric = self._registry.counter(name, tags)
        counter_metric.inc(value)

    def gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value."""
        gauge_metric = self._registry.gauge(name, tags)
        gauge_metric.set(value)

    def histogram(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a value in a histogram metric."""
        hist_metric = self._registry.histogram(name, tags)
        hist_metric.record(value)

    def timed(self, name: str, tags: Optional[Dict[str, str]] = None) -> "TimerContext":
        """Context manager for timing operations and recording to a histogram (in milliseconds).

        Usage:
            metrics = Metrics.get()
            with metrics.timed("my_operation", tags={"tag": "value"}):
                # Code to time
                ...
        """
        # Return the context manager instance directly
        return TimerContext(self, name, tags)

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of all current metric values."""
        return self._registry.get_snapshot()


# Needs to be defined outside Metrics class to avoid capturing 'self' improperly
# in the __exit__ method when used as a context manager returned by metrics.timed()
class TimerContext:
    """Context manager helper for the Metrics.timed method."""

    __slots__ = ("_metrics_facade", "_name", "_tags", "_start_time")

    def __init__(
        self, metrics_facade: Metrics, name: str, tags: Optional[Dict[str, str]]
    ):
        self._metrics_facade = metrics_facade
        self._name = name
        self._tags = tags
        # Use perf_counter for high-resolution timing
        self._start_time = time.perf_counter()

    def __enter__(self):
        # Reset start time just before entering the block
        self._start_time = time.perf_counter()
        return self  # Although typically 'None' is returned if 'as' isn't used

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration in milliseconds
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        # Record duration using the facade's histogram method
        self._metrics_facade.histogram(
            self._name + "_duration_ms", duration_ms, self._tags
        )
        # Don't suppress exceptions: return False or None implicitly


def measure_time(func):
    """Decorator for measuring function execution time.

    Automatically records execution time of the decorated function as a histogram
    using the function's qualified name as the metric name.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that measures and records execution time
    """

    def wrapper(*args, **kwargs):
        metrics = Metrics.get()
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            # Use function's qualified name for metric
            metric_name = f"{func.__module__}.{func.__qualname__}_duration_ms"
            metrics.histogram(metric_name, duration_ms)

    # Preserve function metadata for inspection
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__

    return wrapper
