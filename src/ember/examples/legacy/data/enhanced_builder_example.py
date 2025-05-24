"""Example demonstrating the enhanced DatasetBuilder pattern in Ember.

This example shows how to use the DatasetBuilder to load, transform, and work with
datasets using a fluent interface. It demonstrates the optimized performance features:

1. Thread-safe DataContext for concurrent access
2. Memory-efficient streaming datasets
3. Optimized lookup patterns for high-performance execution

To run:
    uv run python src/ember/examples/data/enhanced_builder_example.py
"""

import time
from typing import Any, Dict

from ember.api import data


def uppercase_transformer(item: Dict[str, Any]) -> Dict[str, Any]:
    """Transform dataset items by uppercasing text fields.

    Args:
        item: Dataset item to transform

    Returns:
        Transformed item with uppercase text fields
    """
    result = item.copy()

    # Convert question to uppercase if present
    if "question" in result:
        result["question"] = result["question"].upper()

    # Convert choices to uppercase if present
    if "choices" in result:
        if isinstance(result["choices"], dict):
            # Handle dictionary-style choices
            result["choices"] = {
                k: v.upper() if isinstance(v, str) else v
                for k, v in result["choices"].items()
            }
        elif isinstance(result["choices"], list):
            # Handle list-style choices (MMLU format)
            result["choices"] = [
                c.upper() if isinstance(c, str) else c for c in result["choices"]
            ]

    return result


def prompt_formatter(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format dataset items into LLM-ready prompts.

    Args:
        item: Dataset item to format

    Returns:
        Item with added formatted prompt
    """
    result = item.copy()

    if "question" in result and "choices" in result:
        choices_text = ""

        if isinstance(result["choices"], dict):
            # Handle dictionary-style choices
            for key, value in result["choices"].items():
                choices_text += f"{key}. {value}\n"
        elif isinstance(result["choices"], list):
            # Handle list-style choices (MMLU format)
            options = ["A", "B", "C", "D"]
            for i, choice in enumerate(result["choices"]):
                if i < len(options):
                    choices_text += f"{options[i]}. {choice}\n"

        result["formatted_prompt"] = (
            f"Question: {result['question']}\n\n"
            f"Answer choices:\n{choices_text}\n"
            f"Please select the correct answer choice."
        )

    return result


def demonstrate_performance() -> None:
    """Demonstrates the optimized performance features."""
    print("\nDemonstrating performance optimizations:")
    print("======================================")

    # The simplified API handles context management internally
    print("Using internally managed data context")

    # Measure data API access performance
    iterations = 10000
    start = time.time()
    for _ in range(iterations):
        # Access the data API which internally manages context
        # This demonstrates the API's performance characteristics
        _ = data.list()  # Fast metadata access
    duration = time.time() - start

    print(
        f"API access performance: {duration * 1e6 / iterations:.2f}ns per operation"
    )
    print(f"  - Fast access to metadata in {iterations} iterations")
    print("  - Thread-safe internal context management")

    # Demonstrate streaming dataset access
    print("\nDemonstrating streaming dataset access:")
    try:
        # Create a streaming dataset (memory-efficient, O(1) space)
        start = time.time()
        stream = (
            data.builder()
            .from_registry("mmlu")  # Use a registered dataset
            .streaming(True)  # Enable streaming mode
            .batch_size(32)  # Configure processing batch size
            .transform(prompt_formatter)  # Add transformations (applied on the fly)
            .limit(5)  # Limit items for example
            .build()
        )
        build_time = time.time() - start

        print(f"  - Streaming dataset built in {build_time:.4f}s")
        print("  - Memory usage: O(1) regardless of dataset size")

        # Process the stream with minimal memory
        count = 0
        start = time.time()
        for item in stream:
            count += 1
            # Stop after 3 items for the example
            if count >= 3:
                break

        process_time = time.time() - start
        print(f"  - Processed {count} items in {process_time:.4f}s")

    except Exception as e:
        print(f"Error demonstrating streaming: {e}")


def main() -> None:
    """Run the enhanced DatasetBuilder example."""
    print("Ember Enhanced DatasetBuilder Example")
    print("=====================================")

    # List available datasets
    print("\nAvailable datasets in registry:")
    datasets = data.list()
    for dataset in datasets[:10]:  # Show first 10
        print(f"  • {dataset}")
    if len(datasets) > 10:
        print(f"  ... and {len(datasets) - 10} more")

    # Use the enhanced builder pattern
    print("\nLoading dataset with builder pattern...")

    try:
        # Time the operation to demonstrate performance
        start_time = time.time()

        # Chain multiple configuration and transformation steps
        dataset = (
            data.builder()
            .from_registry("mmlu")  # Use a registered dataset
            .subset("high_school_mathematics")  # Select a specific subset
            .split("test")  # Choose the test split
            .sample(3)  # Random sample of 3 items
            .transform(uppercase_transformer)  # Transform to uppercase
            .transform(prompt_formatter)  # Format as prompts
            .streaming(False)  # Use non-streaming for this example
            .build()
        )

        end_time = time.time()
        print(f"Dataset loaded in {end_time - start_time:.4f} seconds")

        # Display the loaded dataset
        print(f"\nLoaded dataset with {len(dataset)} entries")

        for i, entry in enumerate(dataset):
            print(f"\nEntry {i+1}:")

            # Display query from entry
            print(f"Query: {entry.query}")

            # Display formatted prompt if available
            if hasattr(entry, "metadata") and entry.metadata:
                if "formatted_prompt" in entry.metadata:
                    print(f"\n{'-'*40}")
                    print(f"{entry.metadata['formatted_prompt']}")
                    print(f"{'-'*40}")

                # Display other metadata
                print("\nMetadata:")
                for key, value in entry.metadata.items():
                    if key != "formatted_prompt":
                        print(f"  {key}: {value}")

            # Display choices if available
            if hasattr(entry, "choices") and entry.choices:
                print("\nChoices:")
                for key, value in entry.choices.items():
                    print(f"  {key}: {value}")

        # Demonstrate performance optimizations
        demonstrate_performance()

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
