import os
import time

import pytest

from ember.core.metrics.metrics import Metrics

# Marker for performance tests, can be skipped using `pytest -m 'not performance'`
performance_marker = pytest.mark.performance

# Check if running in CI or explicitly asked to skip performance tests
# to avoid long run times in certain environments.
skip_performance = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("SKIP_PERFORMANCE_TESTS") == "true",
    reason="Performance tests are skipped in CI or when SKIP_PERFORMANCE_TESTS is set",
)


@performance_marker
@skip_performance
def test_metrics_performance():
    """Benchmark core metrics operations: counter, histogram, snapshot."""
    metrics = Metrics.get()
    registry = metrics._registry

    # Unique names for benchmark metrics
    counter_name = "perf_test_counter"
    hist_name = "perf_test_histogram"

    # Warmup phase - run operations beforehand to stabilize JIT, caches, etc.
    print("\nStarting metrics performance warmup...")
    warmup_iterations = 10_000
    _ = registry.counter(counter_name)  # Ensure creation before loop
    _ = registry.histogram(hist_name)
    for _ in range(warmup_iterations):
        metrics.counter(counter_name)
        metrics.gauge("perf_test_gauge_warmup", 1.0)  # Use unique name for gauge
        metrics.histogram(hist_name, 1.0)
        metrics.get_snapshot()  # Warmup snapshotting
    print("Warmup complete.")

    results = {}

    # --- Counter Benchmark ---
    print("Benchmarking Counter increment...")
    iterations_counter = 10_000_000
    # Ensure counter exists before timing loop
    c = registry.counter(counter_name)
    start_ns = time.perf_counter_ns()
    for _ in range(iterations_counter):
        c.inc()  # Direct increment for tight loop
    elapsed_ns = time.perf_counter_ns() - start_ns
    results["counter_inc_ns"] = elapsed_ns / iterations_counter
    print(f"Counter increment: {results['counter_inc_ns']:.2f} ns/op")

    # --- Histogram Benchmark ---
    print("Benchmarking Histogram record...")
    iterations_hist = 1_000_000
    # Ensure histogram exists before timing loop
    h = registry.histogram(hist_name)
    start_ns = time.perf_counter_ns()
    for i in range(iterations_hist):
        h.record(float(i % 100))  # Direct record for tight loop
    elapsed_ns = time.perf_counter_ns() - start_ns
    results["histogram_record_ns"] = elapsed_ns / iterations_hist
    print(f"Histogram record: {results['histogram_record_ns']:.2f} ns/op")

    # --- Snapshot Benchmark ---
    print("Benchmarking Snapshot retrieval...")
    # Add more metrics to make snapshotting non-trivial
    num_extra_metrics = 1000
    for i in range(num_extra_metrics):
        registry.counter(f"snapshot_filler_counter_{i}")
        g = registry.gauge(f"snapshot_filler_gauge_{i}")
        g.set(float(i))

    iterations_snapshot = 10_000
    start_ns = time.perf_counter_ns()
    for _ in range(iterations_snapshot):
        # Use facade method as intended for snapshots
        _ = metrics.get_snapshot()
    elapsed_ns = time.perf_counter_ns() - start_ns
    results["snapshot_get_ns"] = elapsed_ns / iterations_snapshot
    snapshot_us = results["snapshot_get_ns"] / 1000
    print(
        f"Snapshot retrieval: {snapshot_us:.2f} µs/op ({num_extra_metrics + 2} base metrics)"
    )

    # Basic assertions based on design doc targets (allowing some leeway)
    # These might be flaky depending on the execution environment.
    # Consider adjusting thresholds or making them informational only.
    target_counter_ns = 25  # Target from doc
    target_histogram_ns = 100  # Target from doc (was <100ns, using 100)
    target_snapshot_us = 200  # Target from doc (was <200µs, using 200)

    print("\n--- Performance Targets --- (from 11_METRICS_SYSTEM_DESIGN.md)")
    print(f"Counter Increment Target: < {target_counter_ns} ns/op")
    print(f"Histogram Record Target:  < {target_histogram_ns} ns/op")
    print(f"Snapshot Retrieval Target: < {target_snapshot_us} µs/op")

    # Optional: Assertions (might fail depending on machine speed)
    # assert results["counter_inc_ns"] < target_counter_ns * 2 # Allow 2x leeway
    # assert results["histogram_record_ns"] < target_histogram_ns * 2 # Allow 2x leeway
    # assert snapshot_us < target_snapshot_us * 2 # Allow 2x leeway

    print("\nPerformance benchmark finished.")


# To run only performance tests: pytest -m performance
# To skip performance tests: pytest -m 'not performance'
