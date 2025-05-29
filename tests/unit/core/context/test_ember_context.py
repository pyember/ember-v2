"""Tests for the EmberContext implementation.

These tests verify the core functionality of the new context system:
1. Thread-local storage
2. Performance optimizations
3. Component registration and retrieval
4. Compatibility with existing code
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ember.core.context.ember_context import (
    EmberContext,
    current_context,
    scoped_context,
    temp_component)


def test_context_creation():
    """Tests basic context creation and initialization."""
    # Create context directly
    ctx = EmberContext()
    assert ctx is not None
    assert ctx.config_manager is not None
    assert ctx.logger is not None

    # Access current context
    current = current_context()
    assert current is not None
    assert current.config_manager is not None


def test_thread_local_isolation():
    """Tests that contexts are thread-local and isolated."""
    # Create contexts in different threads
    contexts = []
    threads = []

    # Use an Event instead of a Barrier to avoid potential deadlocks
    thread_ready = [threading.Event() for _ in range(2)]
    main_ready = threading.Event()

    def thread_func(thread_id):
        try:
            # Force new context
            EmberContext._thread_local.context = None

            # Get context for this thread
            ctx = current_context()

            # Create thread-specific model
            ctx.register("model", f"thread_{thread_id}", lambda x: x * thread_id)

            # Signal that this thread is ready
            thread_ready[thread_id].set()

            # Wait for main thread's signal to continue
            if not main_ready.wait(timeout=5.0):
                print(f"Thread {thread_id} timed out waiting for main thread")
                return

            # Check thread isolation
            # Should only see our own thread-specific model
            own_model = ctx.get_model(f"thread_{thread_id}")
            assert own_model is not None
            assert own_model(2) == 2 * thread_id

            # Should not see other thread's model
            other_id = 1 - thread_id  # 0->1, 1->0
            other_model = ctx.get_model(f"thread_{other_id}")
            assert other_model is None

            # Store context for verification in a thread-safe way
            with threading.Lock():
                contexts.append(ctx)
        except Exception as e:
            print(f"Exception in thread {thread_id}: {e}")
            raise

    # Create and start threads
    for i in range(2):
        thread = threading.Thread(target=thread_func, args=(i))
        thread.daemon = True  # Make threads daemon so they don't block test exit
        threads.append(thread)
        thread.start()

    # Wait for threads to complete their setup (with timeout)
    for i, event in enumerate(thread_ready):
        if not event.wait(timeout=5.0):
            pytest.fail(f"Thread {i} did not initialize in time")

    # Signal threads to continue
    main_ready.set()

    # Wait for threads to complete (with timeout)
    for i, thread in enumerate(threads):
        thread.join(timeout=5.0)
        if thread.is_alive():
            pytest.fail(f"Thread {i} did not complete in time")

    # Verify contexts are distinct
    assert len(contexts) == 2
    assert contexts[0] is not contexts[1]


def test_component_registration():
    """Tests component registration and retrieval."""
    ctx = EmberContext()

    # Register test components
    test_model = lambda x: x * 2
    test_operator = lambda x: x + 1
    test_evaluator = lambda x: x > 0

    ctx.register("model", "test_model", test_model)
    ctx.register("operator", "test_operator", test_operator)
    ctx.register("evaluator", "test_evaluator", test_evaluator)

    # Retrieve components
    retrieved_model = ctx.get_model("test_model")
    retrieved_operator = ctx.get_operator("test_operator")
    retrieved_evaluator = ctx.get_evaluator("test_evaluator")

    # Verify retrieval
    assert retrieved_model is test_model
    assert retrieved_operator is test_operator
    assert retrieved_evaluator is test_evaluator

    # Test function execution
    assert retrieved_model(5) == 10
    assert retrieved_operator(5) == 6
    assert retrieved_evaluator(5) is True


def test_scoped_context():
    """Tests the scoped_context context manager."""
    # Get current context
    parent_ctx = current_context()

    # Register component in parent
    parent_ctx.register("model", "parent_model", lambda x: f"parent_{x}")

    # Create scoped context
    with scoped_context() as child_ctx:
        # Verify child context is different
        assert child_ctx is not parent_ctx

        # Verify parent's model is not visible
        # (This is intentional - scoped contexts start fresh)
        assert child_ctx.get_model("parent_model") is None

        # Register model in child
        child_ctx.register("model", "child_model", lambda x: f"child_{x}")

        # Verify child's model is visible in child
        child_model = child_ctx.get_model("child_model")
        assert child_model is not None
        assert child_model("test") == "child_test"

        # Verify current context is child
        assert current_context() is child_ctx

    # After scope ends, verify current context is parent again
    assert current_context() is parent_ctx

    # Verify parent can't see child's model
    assert parent_ctx.get_model("child_model") is None


def test_temp_component():
    """Tests the temp_component context manager."""
    ctx = current_context()

    # Register initial component
    original_model = lambda x: f"original_{x}"
    ctx.register("model", "test_model", original_model)

    # Verify initial component
    assert ctx.get_model("test_model") is original_model

    # Create temporary replacement
    temp_model = lambda x: f"temp_{x}"
    with temp_component("model", "test_model", temp_model) as model:
        # Verify temp component is active
        assert model is temp_model
        assert ctx.get_model("test_model") is temp_model
        assert ctx.get_model("test_model")("xyz") == "temp_xyz"

    # Verify original component is restored
    assert ctx.get_model("test_model") is original_model
    assert ctx.get_model("test_model")("xyz") == "original_xyz"


def test_performance():
    """Tests context performance optimization.

    This test is marked as slow and only runs with --run-perf-tests.
    """
    if not pytest.config.getoption("--run-perf-tests"):
        pytest.skip("Only runs with --run-perf-tests flag")

    ctx = EmberContext()

    # Register test component
    test_model = lambda x: x * 2
    ctx.register("model", "perf_test", test_model)

    # Prime the cache and JIT
    for _ in range(1000):
        ctx.get_model("perf_test")
        ctx.get_model("nonexistent")

    # Measure cached lookup performance
    iterations = 1_000_000
    start = time.perf_counter()
    for _ in range(iterations):
        model = ctx.get_model("perf_test")
    duration = time.perf_counter() - start

    # Calculate nanoseconds per operation
    ns_per_op = (duration * 1_000_000_000) / iterations

    # Output performance metrics
    print(f"\nPerformance: {ns_per_op:.1f} ns per cached lookup")

    # Verify performance is good
    # This test may need adjustment based on the specific hardware
    assert ns_per_op < 100, "Performance regression detected"


def test_thread_scaling():
    """Tests context scaling across threads.

    This test is marked as slow and only runs with --run-perf-tests.
    """
    if not pytest.config.getoption("--run-perf-tests"):
        pytest.skip("Only runs with --run-perf-tests flag")

    # Test parameters
    thread_counts = [1, 2, 4, 8]
    iterations_per_thread = 100_000

    results = {}

    for num_threads in thread_counts:
        # Function to execute in each thread
        def thread_work():
            # Get thread-local context
            ctx = current_context()

            # Register thread-local model
            model_name = f"model_{threading.get_ident()}"
            ctx.register("model", model_name, lambda x: x * 2)

            # Access component repeatedly
            for _ in range(iterations_per_thread):
                model = ctx.get_model(model_name)
                result = model(42)

            return result

        # Execute with thread pool
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_work) for _ in range(num_threads)]
            for f in futures:
                f.result()  # Wait for completion
        duration = time.perf_counter() - start

        # Store result
        total_iterations = num_threads * iterations_per_thread
        ops_per_second = total_iterations / duration
        results[num_threads] = ops_per_second

    # Output scaling results
    print("\nThread scaling results:")
    baseline = results[1]
    for threads, ops in sorted(results.items()):
        scaling = ops / baseline
        print(f"{threads} threads: {ops:,.0f} ops/sec ({scaling:.2f}x scaling)")

    # Verify good scaling (at least 50% of ideal)
    # 8 threads should achieve at least 4x the single-thread performance
    if 8 in results and 1 in results:
        scaling_8_threads = results[8] / results[1]
        assert scaling_8_threads >= 4.0, "Insufficient thread scaling"


# No need for this fixture as the options are already defined in conftest.py


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
