"""Example of using the new context system.

This example demonstrates how to use the new context system with its core
components and threading features.
"""

import threading
import time
from typing import Dict, Any

from ember.core.context import Registry
from ember.core.context.config import ConfigComponent
from ember.core.context.model import ModelComponent
from ember.core.context.data import DataComponent
from ember.core.context.metrics import MetricsComponent
from ember.core.context.management import scoped_registry, temp_component


def basic_usage() -> None:
    """Demonstrate basic usage of the context system."""
    print("\n=== Basic Component Usage ===")

    # Create configuration
    config = ConfigComponent(
        config_data={"models": {"simple_model": {"type": "mock", "temperature": 0.7}}}
    )

    # Create model component
    model = ModelComponent()

    # Create simple mock model
    class MockModel:
        def __init__(self, name: str):
            self.name = name

        def generate(self, prompt: str) -> str:
            return f"[{self.name}] Response to: {prompt}"

    # Register a model
    model.register_model("example_model", MockModel("Example"))

    # Use the model
    example_model = model.get_model("example_model")
    response = example_model.generate("Hello, world!")
    print(f"Model response: {response}")

    # Check configuration
    temp = config.get_value("models", "simple_model", {}).get("temperature")
    print(f"Model temperature: {temp}")

    # Access through registry
    registry = Registry.current()
    config_from_registry = registry.get("config")
    model_from_registry = registry.get("model")

    print(
        f"Found in registry: config={config_from_registry is config}, "
        f"model={model_from_registry is model}"
    )


def thread_isolation() -> None:
    """Demonstrate thread isolation with the context system."""
    print("\n=== Thread Isolation ===")

    # Create shared metric component for tracking
    metrics = MetricsComponent()

    def worker_thread(thread_id: int) -> None:
        """Thread worker function."""
        # Each thread gets its own registry
        registry = Registry.current()

        # Create thread-local components
        config = ConfigComponent(config_data={"thread_info": {"id": thread_id}})
        model = ModelComponent()

        # Create mock model for this thread
        class ThreadModel:
            def generate(self, prompt: str) -> str:
                return f"Thread {thread_id}: {prompt}"

        # Register thread-specific model
        model.register_model("thread_model", ThreadModel())

        # Use the model
        thread_model = model.get_model("thread_model")
        response = thread_model.generate("Hello from thread")

        # Record metrics (shared across threads)
        with metrics.timed(f"thread_{thread_id}_duration"):
            time.sleep(0.1 * thread_id)  # Simulate work

        metrics.counter("threads_completed")

        # Print result with thread ID from config
        thread_id_from_config = config.get_value("thread_info", "id", thread_id)
        print(f"Thread {thread_id_from_config} response: {response}")

    # Create and start threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Show metrics
    all_metrics = metrics.get_metrics()
    print(f"Threads completed: {all_metrics['counters'].get('threads_completed', 0)}")
    print("Thread timings:")
    for key, value in all_metrics.get("histograms", {}).items():
        if key.startswith("thread_"):
            print(f"  {key}: {value.get('sum', 0):.1f}ms")


def scoped_registry_example() -> None:
    """Demonstrate using scoped registry for isolation."""
    print("\n=== Scoped Registry ===")

    # Create components in main registry
    main_config = ConfigComponent(config_data={"config_info": {"scope": "main"}})

    # Create a scoped registry
    with scoped_registry() as registry:
        # Create components in scoped registry
        scoped_config = ConfigComponent(registry, config_data={"config_info": {"scope": "scoped"}})

        # Check values
        main_scope = main_config.get_value("config_info", "scope", "unknown")
        scoped_scope = scoped_config.get_value("config_info", "scope", "unknown")

        print(f"Inside scope - Main: {main_scope}, Scoped: {scoped_scope}")

    # After scope, only main registry exists
    current_reg = Registry.current()
    config = current_reg.get("config")
    scope = config.get_value("config_info", "scope", "unknown")
    print(f"After scope - Current: {scope}")


def temp_component_example() -> None:
    """Demonstrate using temporary components."""
    print("\n=== Temporary Component ===")

    # Create model component
    model = ModelComponent()

    # Register original model
    class OriginalModel:
        def generate(self, prompt: str) -> str:
            return f"Original: {prompt}"

    model.register_model("test_model", OriginalModel())

    # Create temporary model
    class TemporaryModel:
        def generate(self, prompt: str) -> str:
            return f"Temporary: {prompt}"

    # Use temporary model
    with temp_component("model", TemporaryModel()) as temp_model:
        response = temp_model.generate("Hello")
        print(f"Within temp scope: {response}")

    # After temp component, original is restored
    registry = Registry.current()
    model_component = registry.get("model")
    model_instance = model_component.get_model("test_model")
    response = model_instance.generate("Hello again")
    print(f"After temp scope: {response}")


if __name__ == "__main__":
    # Clear any existing registries
    Registry.clear()

    # Run examples
    basic_usage()
    thread_isolation()
    scoped_registry_example()
    temp_component_example()

    print("\nAll examples completed successfully!")
