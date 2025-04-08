"""Example demonstrating the use of DataContext.

This example shows how to create and use a DataContext with custom configuration
for efficient and thread-safe management of dataset operations.
"""

import logging
from typing import Any, Dict, List

from ember.api.data import DataAPI
from ember.core.utils.data.base.models import TaskType
from ember.core.utils.data.context.data_context import (
    DataConfig,
    DataContext,
    get_default_context,
    set_default_context,
)


def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_custom_context() -> DataContext:
    """Create a custom DataContext with specific configuration.

    This demonstrates how to create a DataContext with custom settings
    for memory usage and caching behavior.

    Returns:
        Custom DataContext
    """
    # Create custom configuration
    config = DataConfig(
        cache_dir="/tmp/ember_cache",  # Use local cache directory
        batch_size=64,  # Process data in larger batches
        cache_ttl=7200,  # Cache data for 2 hours
    )

    # Create context with auto-discovery
    context = DataContext(config=config, auto_discover=True)

    return context


def register_custom_dataset(context: DataContext) -> None:
    """Register a custom dataset with the context.

    This demonstrates how to register a custom dataset with metadata
    without needing to modify global state.

    Args:
        context: DataContext to register with
    """
    # Register a simple custom dataset
    context.register_dataset(
        name="example_dataset",
        source="example/source",
        task_type=TaskType.MULTIPLE_CHOICE,
        description="Example dataset for DataContext demo",
    )

    # Log available datasets
    logging.info("Available datasets: %s", context.registry.list_datasets())


def process_data_with_context(context: DataContext) -> List[Dict[str, Any]]:
    """Process data using the DataContext.

    This demonstrates how to use the DataContext for data operations
    with explicit dependency management.

    Args:
        context: DataContext for dataset operations

    Returns:
        List of processed data items
    """
    # Create API with explicit context
    api = DataAPI(context=context)

    # Create a custom data processing pipeline
    pipeline = (
        api.builder()
        .from_registry("mmlu")  # Use a standard dataset
        .split("validation")  # Select validation split
        .sample(10)  # Get 10 samples
        .transform(
            lambda x: {  # Transform data format
                "question": f"Question: {x.get('question', '')}",
                "options": x.get("choices", {}),
                "answer": x.get("answer", None),
            }
        )
    )

    # Check if custom dataset is available
    available_datasets = api.list()
    if "example_dataset" in available_datasets:
        # Also process custom dataset
        custom_pipeline = api.builder().from_registry("example_dataset").limit(5)

        # Collect data
        custom_data = list(custom_pipeline.build())
        logging.info("Processed %d items from custom dataset", len(custom_data))

    # Collect data from main pipeline
    results = list(pipeline.build())
    logging.info("Processed %d items", len(results))

    # Return processed items
    return [
        {
            "id": i,
            "question": item.question if hasattr(item, "question") else "",
            "options": item.options if hasattr(item, "options") else {},
        }
        for i, item in enumerate(results)
    ]


def demonstrate_thread_safety():
    """Demonstrate that DataContext is thread-safe.

    This shows how the DataContext can be used safely in a
    multi-threaded environment without race conditions.
    """
    import threading

    # Create shared context
    shared_context = create_custom_context()

    # Set as default for all threads
    set_default_context(shared_context)

    # Thread function that uses the default context
    def thread_function(thread_id: int):
        # Get default context (should be our shared context)
        context = get_default_context()

        # Log available datasets
        datasets = context.registry.list_datasets()
        logging.info("Thread %d sees datasets: %s", thread_id, datasets)

        # Process some data
        results = process_data_with_context(context)
        logging.info("Thread %d processed %d items", thread_id, len(results))

    # Create and start threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=thread_function, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    logging.info("All threads completed successfully")


def run_example():
    """Run the DataContext example."""
    # Set up logging
    setup_logging()

    # Create custom context
    context = create_custom_context()

    # Register custom dataset
    register_custom_dataset(context)

    # Process data with context
    results = process_data_with_context(context)

    # Show results
    for item in results[:3]:  # Show first 3 items
        logging.info("Processed item: %s", item)

    # Demonstrate thread safety
    demonstrate_thread_safety()


if __name__ == "__main__":
    run_example()
