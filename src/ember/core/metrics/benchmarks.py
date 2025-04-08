"""Benchmarking tools for measuring metrics system performance.

Provides tools for measuring key performance characteristics of the
metrics system under various workloads.
"""

import gc
import threading
import time
from typing import Any, Dict

from .metrics import Metrics


def benchmark_metrics_performance() -> Dict[str, float]:
    """Benchmark metrics system performance.

    Runs various benchmarks to measure critical metrics operations
    under controlled conditions.

    Returns:
        Dict of benchmark results
    """
    metrics = Metrics.get()

    # Force GC to reduce interference
    gc.collect()

    # Warmup
    for _ in range(1000):
        metrics.counter("warmup")
        metrics.gauge("warmup", 1.0)
        metrics.histogram("warmup", 1.0)

    results = {}

    # Counter benchmark
    iterations = 10_000_000
    start = time.perf_counter_ns()
    for _ in range(iterations):
        metrics.counter("test_counter")
    elapsed = time.perf_counter_ns() - start
    results["counter_ns"] = elapsed / iterations

    # Histogram benchmark
    iterations = 1_000_000
    start = time.perf_counter_ns()
    for i in range(iterations):
        metrics.histogram("test_histogram", i % 100)
    elapsed = time.perf_counter_ns() - start
    results["histogram_ns"] = elapsed / iterations

    # Gauge benchmark
    iterations = 5_000_000
    start = time.perf_counter_ns()
    for i in range(iterations):
        metrics.gauge("test_gauge", i % 100)
    elapsed = time.perf_counter_ns() - start
    results["gauge_ns"] = elapsed / iterations

    # Snapshot benchmark
    iterations = 10_000
    start = time.perf_counter_ns()
    for _ in range(iterations):
        metrics.get_snapshot()
    elapsed = time.perf_counter_ns() - start
    results["snapshot_ns"] = elapsed / iterations

    return results


def test_thread_safety() -> Dict[str, Any]:
    """Verify thread safety of metrics system.

    Runs concurrent operations across multiple threads to verify
    that metrics are recorded correctly without data races.

    Returns:
        Dict with verification results
    """
    metrics = Metrics.get()
    thread_count = 16
    operations = 100_000

    def worker():
        for i in range(operations):
            metrics.counter("test_counter")
            metrics.histogram("test_histogram", i % 100)

    threads = []
    for _ in range(thread_count):
        thread = threading.Thread(target=worker)
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Verify metrics
    snapshot = metrics.get_snapshot()
    total_count = snapshot.get("test_counter", 0)
    expected_count = thread_count * operations

    return {
        "expected_count": expected_count,
        "actual_count": total_count,
        "match": abs(total_count - expected_count) / expected_count < 0.01,
    }
