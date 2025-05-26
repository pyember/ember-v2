"""Tests for thread-local isolation in the EmberContext system."""

import queue
import threading

import pytest

from ember.core.context import EmberContext, current_context


def test_thread_local_isolation():
    """Test that contexts are thread-local and properly isolated."""
    # Reset for test isolation
    EmberContext._test_mode = True

    # Create contexts in different threads
    results_queue = queue.Queue()
    barrier = threading.Barrier(4)  # 3 threads + main

    def thread_function(thread_id):
        # Get context for this thread
        ctx = current_context()

        # Register a thread-specific model with the thread ID
        ctx.register("model", "my-model", lambda x: f"Thread {thread_id}: {x}")

        # Synchronize threads
        barrier.wait()

        # Get and use the model
        model = ctx.get_model("my-model")
        result = model("Hello")
        results_queue.put((thread_id, "own_model", result))

        # Try to access another thread's model (should be None)
        other_id = (thread_id + 1) % 3
        other_model = ctx.get_model(f"thread-{other_id}-model")
        results_queue.put((thread_id, "other_model", other_model))

    # Create a model in the main thread context
    ctx = current_context()
    for i in range(3):
        ctx.register(
            "model", f"thread-{i}-model", lambda x, i=i: f"Global thread-{i} model: {x}"
        )

    # Create and start threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=thread_function, args=(i))
        threads.append(thread)
        thread.start()

    # Wait for threads to be ready
    barrier.wait()

    # Wait for threads to complete
    for thread in threads:
        thread.join()

    # Analyze results
    results = {}
    while not results_queue.empty():
        thread_id, test_type, value = results_queue.get()
        if thread_id not in results:
            results[thread_id] = {}
        results[thread_id][test_type] = value

    # Verify results
    for thread_id in range(3):
        # Each thread should see its own model
        assert results[thread_id]["own_model"] == f"Thread {thread_id}: Hello"
        # Each thread should not see other thread's models
        assert results[thread_id]["other_model"] is None

    # Clean up
    EmberContext._test_mode = False


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
