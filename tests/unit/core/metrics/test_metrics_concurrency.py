import threading

import pytest

from ember.core.metrics.metrics import Metrics


@pytest.mark.concurrency
def test_metrics_thread_safety():
    """Verify thread safety of metrics system under concurrent load."""
    # Use the singleton instance
    metrics = Metrics.get()
    registry = metrics._registry

    thread_count = 16
    operations_per_thread = 100_000

    # Clear previous metrics if possible, or use unique names
    # For simplicity, assume a clean state or use unique names for this test
    counter_name = "thread_safety_test_counter"
    hist_name = "thread_safety_test_histogram"
    gauge_name = "thread_safety_test_gauge"

    # Ensure metrics are created before threads start to avoid race on creation lock
    # Although the registry is designed to handle concurrent creation safely.
    _ = registry.counter(counter_name)
    _ = registry.histogram(hist_name)
    _ = registry.gauge(gauge_name)

    errors = []  # List to collect errors from threads

    def worker(worker_id):
        try:
            local_metrics = Metrics.get()  # Get instance within thread
            # Test thread-local caching by getting metrics multiple times
            counter = local_metrics._registry.counter(counter_name)
            hist = local_metrics._registry.histogram(hist_name)
            gauge = local_metrics._registry.gauge(gauge_name)

            for i in range(operations_per_thread):
                # Increment counter
                local_metrics.counter(counter_name, 1)

                # Record histogram value
                local_metrics.histogram(hist_name, float(i % 100))

                # Set gauge value (last one wins, but tests concurrent sets)
                local_metrics.gauge(gauge_name, float(worker_id * 1000 + i))

                # Verify thread-local cache returns same object
                assert local_metrics._registry.counter(counter_name) is counter
                assert local_metrics._registry.histogram(hist_name) is hist
                assert local_metrics._registry.gauge(gauge_name) is gauge

        except Exception as e:
            errors.append(f"Worker {worker_id} error: {e}")

    threads = []
    for i in range(thread_count):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check for errors reported by threads
    assert not errors, "\n".join(errors)

    # Verify final metrics state
    snapshot = metrics.get_snapshot()

    # Counter verification
    expected_count = thread_count * operations_per_thread
    actual_count = snapshot.get(counter_name, 0)
    # Allow for minor discrepancies if atomicity isn't perfect across ops, though int += should be safe
    assert (
        abs(actual_count - expected_count) == 0
    ), f"Counter mismatch: expected {expected_count}, got {actual_count}"

    # Histogram verification
    hist_snap = snapshot.get(hist_name)
    assert hist_snap is not None, "Histogram not found in snapshot"
    actual_hist_count = hist_snap.get("count", 0)
    assert (
        abs(actual_hist_count - expected_count) == 0
    ), f"Histogram count mismatch: expected {expected_count}, got {actual_hist_count}"
    # Sum verification is harder due to potential float precision issues with many adds
    # Just check count primarily for concurrency test

    # Gauge verification (difficult to assert exact value, just check it exists)
    assert gauge_name in snapshot, "Gauge not found in snapshot"
    # Value depends on which thread finished last

    print(
        f"Concurrency test passed: Counter={actual_count}, Histogram Count={actual_hist_count}"
    )


# Consider adding more specific concurrency tests, e.g., focusing on
# concurrent creation, concurrent snapshotting while recording, etc.
# if specific race conditions are suspected or need verification.
