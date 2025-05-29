"""Example demonstrating the high-performance EmberContext system.

This example showcases:
1. Thread-local context access with near-zero overhead
2. Cache-optimized component lookup
3. Thread safety and isolation for concurrent applications
4. Configuration access through context
5. Performance measurements

The example follows the core architectural principles of the Ember framework
and demonstrates how the unified context system improves performance.
"""

import threading
import time

# Use absolute imports following the Google Python Style Guide
from ember.core.context import current_context, temp_component
from ember.core.context.config_integration import config_override


class SimpleModel:
    """Simple model for example purposes."""

    def __init__(self, name: str, temperature: float = 0.7):
        self.name = name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """Simulate text generation."""
        return f"Output from {self.name} (temp={self.temperature}): {prompt}"


def process_with_model(model_name: str, prompt: str) -> str:
    """Process text with a model."""
    # Get current context (thread-local, zero-overhead)
    ctx = current_context()

    # Get model from context (efficient lookup)
    model = ctx.get_model(model_name)
    if model is None:
        return f"Model {model_name} not found"

    # Use model to generate output
    return model.generate(prompt)


def benchmark_context_access(iterations: int = 1_000_000) -> float:
    """Benchmark context access performance.

    Args:
        iterations: Number of iterations

    Returns:
        Average time per operation in nanoseconds
    """
    # Register test model
    ctx = current_context()
    ctx.register("model", "test_model", SimpleModel("test_model"))

    # Warm-up
    for _ in range(10000):
        current_context()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = current_context()
    end = time.time()

    # Calculate average time
    total_ns = (end - start) * 1e9
    return total_ns / iterations


def benchmark_model_lookup(iterations: int = 1_000_000) -> float:
    """Benchmark model lookup performance.

    Args:
        iterations: Number of iterations

    Returns:
        Average time per lookup in nanoseconds
    """
    # Register test model
    ctx = current_context()
    ctx.register("model", "test_model", SimpleModel("test_model"))

    # Warm-up
    for _ in range(10000):
        ctx.get_model("test_model")

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = ctx.get_model("test_model")
    end = time.time()

    # Calculate average time
    total_ns = (end - start) * 1e9
    return total_ns / iterations


def benchmark_config_access(iterations: int = 1_000_000) -> float:
    """Benchmark configuration access performance.

    Args:
        iterations: Number of iterations

    Returns:
        Average time per access in nanoseconds
    """
    # Get context
    ctx = current_context()

    # Warm-up
    for _ in range(10000):
        _ = ctx.config

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = ctx.config
    end = time.time()

    # Calculate average time
    total_ns = (end - start) * 1e9
    return total_ns / iterations


def demo_thread_isolation() -> None:
    """Demonstrate thread isolation with thread-local contexts."""

    def worker(thread_id: int) -> None:
        """Worker function showing thread isolation."""
        # Each thread gets its own context
        ctx = current_context()

        # Register thread-specific model
        model = SimpleModel(f"thread-{thread_id}", temperature=thread_id / 10)
        ctx.register("model", "my-model", model)

        # Use model
        result = process_with_model("my-model", "Hello from thread")
        print(f"Thread {thread_id}: {result}")

        # Now show that another thread's model is separate
        other_result = process_with_model("thread-model", "Test isolation")
        print(f"Thread {thread_id} accessing 'thread-model': {other_result}")

    # Create threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(i))
        threads.append(thread)

    # Start threads
    for thread in threads:
        thread.start()

    # Register a global model
    ctx = current_context()
    ctx.register("model", "thread-model", SimpleModel("thread-model", temperature=0.5))

    # Wait for threads to complete
    for thread in threads:
        thread.join()


def demo_config_override() -> None:
    """Demonstrate configuration overrides."""
    # Get current context
    ctx = current_context()

    # Register model
    ctx.register("model", "demo-model", SimpleModel("demo-model"))

    # Get output with default config
    print("Default configuration:")
    result = process_with_model("demo-model", "Hello world")
    print(result)

    # Override configuration
    print("\nWith configuration override:")
    with config_override({"model": {"temperature": 0.2}}):
        # Register model with config temperature
        temperature = ctx.config.get("model", {}).get("temperature", 0.7)
        ctx.register("model", "demo-model", SimpleModel("demo-model", temperature))

        # Get output with overridden config
        result = process_with_model("demo-model", "Hello world")
        print(result)

    # Back to default configuration
    print("\nBack to default configuration:")
    result = process_with_model("demo-model", "Hello world")
    print(result)


def demo_temporary_components() -> None:
    """Demonstrate temporary component registration."""
    # Get current context
    ctx = current_context()

    # Register permanent model
    ctx.register("model", "permanent", SimpleModel("permanent-model"))

    # Use permanent model
    print("Using permanent model:")
    result = process_with_model("permanent", "Hello")
    print(result)

    # Use temporary model
    print("\nUsing temporary model:")
    with temp_component(
        "model", "temporary", SimpleModel("temporary-model", 0.3)
    ) as model:
        result = process_with_model("temporary", "Hello")
        print(result)

    # Check if temporary model is gone
    print("\nAfter temporary scope:")
    result = process_with_model("temporary", "Hello")
    print(result)

    # Permanent model still exists
    print("\nPermanent model still exists:")
    result = process_with_model("permanent", "Hello")
    print(result)


def demo_xcs_integration() -> None:
    """Demonstrate XCS integration with the unified context.

    This demonstrates how the context system conceptually integrates with XCS
    execution, following patterns shown in the API guidelines.
    """
    # When actually running this example with a proper setup,
    # the XCS integration would work as described below.
    # For demonstration purposes, we show the pattern without
    # actually executing potentially uninitialized XCS code.

    # Get current context
    ctx = current_context()

    print("XCS Integration Pattern")
    print("----------------------")
    print("The unified context system enables seamless XCS integration.")
    print("With a properly initialized system, you could use:")

    print("\n1. JIT execution with context awareness:")
    print("   @jit")
    print("   def process(data):")
    print("       # Access context inside JIT-compiled function")
    print("       ctx = current_context()")
    print("       model = ctx.get_model('my-model')")
    print("       return model.generate(data)")

    print("\n2. Execution options from context configuration:")
    print("   with execution_options(scheduler='parallel'):")
    print("       result = process(data)")

    print("\n3. Thread-local execution contexts:")
    print("   # Each thread has its own execution context")
    print("   # with proper resource isolation")

    print("\nThis integration enables high-performance, thread-safe execution")


def run_example() -> None:
    """Run the complete context system example."""
    print("EmberContext Performance Example")
    print("================================\n")

    # Register example model
    ctx = current_context()
    ctx.register("model", "example", SimpleModel("example-model"))

    # Run performance benchmarks
    print("Performance Benchmarks:")
    ctx_ns = benchmark_context_access()
    print(f"Context access:   {ctx_ns:.2f} ns per operation")

    model_ns = benchmark_model_lookup()
    print(f"Model lookup:     {model_ns:.2f} ns per operation")

    config_ns = benchmark_config_access()
    print(f"Config access:    {config_ns:.2f} ns per operation")

    print("\nFeature Demonstrations:\n")

    # Demo thread isolation
    print("\n1. Thread Isolation")
    print("------------------")
    demo_thread_isolation()

    # Demo configuration overrides
    print("\n\n2. Configuration Overrides")
    print("-------------------------")
    demo_config_override()

    # Demo temporary components
    print("\n\n3. Temporary Components")
    print("---------------------")
    demo_temporary_components()

    # Demo XCS integration
    print("\n\n4. XCS Integration")
    print("-----------------")
    try:
        demo_xcs_integration()
    except (ImportError, AttributeError) as e:
        print(f"XCS integration demo skipped: {e}")


if __name__ == "__main__":
    run_example()
