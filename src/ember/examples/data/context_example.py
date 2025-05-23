"""Example demonstrating the simplified data API.

This example shows how to use the simplified data API for loading,
transforming, and working with datasets without managing contexts.
"""

import logging
import threading
from typing import Any, Dict, List

from ember.api import data
from ember.api.data import TaskType


def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def demonstrate_basic_loading() -> List[Dict[str, Any]]:
    """Demonstrate basic data loading with the simplified API.

    Shows how to load datasets using the data API which handles
    context management automatically.

    Returns:
        List of processed data items
    """
    results = []
    
    # Direct loading with streaming
    logging.info("Loading MMLU dataset with streaming...")
    for i, item in enumerate(data("mmlu", streaming=True, limit=5)):
        results.append({
            "id": i,
            "question": item.question if hasattr(item, "question") else "",
            "answer": item.answer if hasattr(item, "answer") else ""
        })
    
    logging.info(f"Loaded {len(results)} items")
    return results


def demonstrate_builder_pattern() -> List[Dict[str, Any]]:
    """Demonstrate using the builder pattern for advanced configuration.

    Shows how to use the data API builder for complex data processing
    pipelines with transformations.

    Returns:
        List of transformed data items
    """
    # Create a custom data processing pipeline
    pipeline = (
        data.builder()
        .from_registry("mmlu")  # Use a standard dataset
        .split("validation")    # Select validation split
        .subset("high_school_physics")  # Select specific subject
        .sample(10)            # Get 10 samples
        .transform(            # Transform data format
            lambda x: {
                "formatted_question": f"Q: {x.get('question', '')}",
                "options": x.get("choices", {}),
                "correct_answer": x.get("answer", None),
                "subject": "physics"
            }
        )
    )

    # Build and collect results
    results = list(pipeline.build())
    logging.info(f"Processed {len(results)} items with builder pattern")
    
    # Show sample transformed item
    if results:
        logging.info("Sample transformed item: %s", results[0])
    
    return results


def demonstrate_dataset_info():
    """Demonstrate getting dataset information."""
    # List available datasets
    available = data.list()
    logging.info(f"Available datasets: {available[:5]}...")  # Show first 5
    
    # Get detailed info about a dataset
    if "mmlu" in available:
        info = data.info("mmlu")
        logging.info(f"MMLU Dataset Info:")
        logging.info(f"  Description: {info.description[:100]}...")
        logging.info(f"  Task Type: {info.task_type}")
        logging.info(f"  Available splits: {info.splits}")
        if hasattr(info, "subjects"):
            logging.info(f"  Subjects: {info.subjects[:5]}...")  # Show first 5


def demonstrate_streaming_efficiency():
    """Demonstrate memory-efficient streaming."""
    logging.info("\nDemonstrating streaming efficiency...")
    
    # Process large dataset without loading all into memory
    processed_count = 0
    total_length = 0
    
    for item in data("mmlu", streaming=True, limit=100):
        # Process each item individually
        if hasattr(item, "question"):
            total_length += len(item.question)
        processed_count += 1
        
        # Show progress every 25 items
        if processed_count % 25 == 0:
            logging.info(f"Processed {processed_count} items...")
    
    logging.info(f"Processed {processed_count} items with average question length: {total_length/processed_count:.1f}")


def demonstrate_thread_safety():
    """Demonstrate that the data API is thread-safe.

    Shows how the API can be used safely in multi-threaded
    environments without explicit context management.
    """
    logging.info("\nDemonstrating thread safety...")
    
    def thread_function(thread_id: int):
        """Function run by each thread."""
        # Each thread can use the data API independently
        items = list(data("mmlu", streaming=True, limit=3))
        logging.info(f"Thread {thread_id} loaded {len(items)} items")
        
        # Also test builder pattern in threads
        builder_items = list(
            data.builder()
            .from_registry("mmlu")
            .sample(2)
            .build()
        )
        logging.info(f"Thread {thread_id} built {len(builder_items)} items")

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


def register_and_use_custom_dataset():
    """Demonstrate registering a custom dataset.
    
    Note: This would typically require actual data source implementation.
    This is shown for API demonstration purposes.
    """
    try:
        # Register a custom dataset
        data.register(
            name="my_custom_dataset",
            source="path/to/custom/data",
            task_type=TaskType.QUESTION_ANSWERING,
            description="Custom QA dataset for demo"
        )
        logging.info("Custom dataset registered successfully")
        
        # Check if it appears in the list
        if "my_custom_dataset" in data.list():
            logging.info("Custom dataset is now available")
    except Exception as e:
        logging.warning(f"Custom dataset registration failed (expected in demo): {e}")


def run_example():
    """Run all data API examples."""
    # Set up logging
    setup_logging()
    
    logging.info("=== EMBER DATA API EXAMPLES ===\n")
    
    # Basic loading
    logging.info("1. Basic Data Loading")
    demonstrate_basic_loading()
    
    # Builder pattern
    logging.info("\n2. Builder Pattern")
    demonstrate_builder_pattern()
    
    # Dataset information
    logging.info("\n3. Dataset Information")
    demonstrate_dataset_info()
    
    # Streaming efficiency
    logging.info("\n4. Streaming Efficiency")
    demonstrate_streaming_efficiency()
    
    # Thread safety
    logging.info("\n5. Thread Safety")
    demonstrate_thread_safety()
    
    # Custom datasets
    logging.info("\n6. Custom Dataset Registration")
    register_and_use_custom_dataset()
    
    logging.info("\n=== Examples completed successfully ===")


if __name__ == "__main__":
    run_example()