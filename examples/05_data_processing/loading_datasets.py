"""Loading Datasets - Simple data loading with Ember's new API.

Difficulty: Basic
Time: ~5 minutes

Learning Objectives:
- Load datasets with the simple data API
- Use streaming for memory efficiency
- Apply filters and transformations
- Understand the progressive disclosure design

Example:
    >>> from ember.api import data
    >>> # Stream data (memory efficient)
    >>> for item in data.stream("mmlu"):
    ...     process(item)
    >>>
    >>> # Or load into memory
    >>> dataset = data.load("mmlu", split="test")
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import data, operators
from ember.api.xcs import jit
from ember.xcs import vmap


def main():
    """Learn to work with datasets using Ember's simple API."""
    print_section_header("Loading and Processing Datasets")

    # Part 1: Simple Data Loading
    print("Part 1: Basic Data Loading")
    print("=" * 50 + "\n")

    # For demo, create a simple data source
    demo_data = [
        {"id": 1, "text": "Machine learning is fascinating", "category": "tech"},
        {"id": 2, "text": "I love cooking pasta", "category": "food"},
        {"id": 3, "text": "The weather is beautiful today", "category": "general"},
        {"id": 4, "text": "Python is a great language", "category": "tech"},
        {"id": 5, "text": "Pizza is my favorite food", "category": "food"},
        {"id": 6, "text": "Neural networks are powerful", "category": "tech"},
        {"id": 7, "text": "Coffee keeps me productive", "category": "food"},
        {"id": 8, "text": "The sunset was amazing", "category": "general"},
    ]

    print("Demo: Working with in-memory data")
    print(f"Total items: {len(demo_data)}")
    print_example_output("First item", demo_data[0])

    # Process with simple functions
    def add_text_length(item: dict) -> dict:
        """Add text length to each item."""
        return {**item, "text_length": len(item["text"])}

    # Process all items
    processed = [add_text_length(item) for item in demo_data]
    print_example_output("Processed item", processed[0])

    # Part 2: Streaming Pattern
    print("\n" + "=" * 50)
    print("Part 2: Streaming Data Pattern")
    print("=" * 50 + "\n")

    # Simulate streaming from a data source
    def stream_demo_data():
        """Simulate streaming data source."""
        for item in demo_data:
            yield item

    print("Processing data as a stream:")
    count = 0
    for item in stream_demo_data():
        if item["category"] == "tech":
            count += 1
            print(f"  Tech item {count}: {item['text'][:30]}...")

    print_example_output("Tech items found", count)

    # Part 3: Filter and Transform
    print("\n" + "=" * 50)
    print("Part 3: Filter and Transform")
    print("=" * 50 + "\n")

    # Simple filter function
    def is_tech_item(item: dict) -> bool:
        """Filter for tech items."""
        return item["category"] == "tech"

    # Simple transform function
    def enrich_item(item: dict) -> dict:
        """Add metadata to item."""
        return {
            **item,
            "word_count": len(item["text"].split()),
            "uppercase_text": item["text"].upper(),
            "is_long": len(item["text"]) > 30,
        }

    # Apply filter and transform
    tech_items = [item for item in demo_data if is_tech_item(item)]
    enriched_items = [enrich_item(item) for item in tech_items]

    print("Filtered and transformed items:")
    for item in enriched_items:
        print(f"  ID {item['id']}: {item['word_count']} words, long={item['is_long']}")

    # Part 4: Batch Processing
    print("\n" + "=" * 50)
    print("Part 4: Batch Processing")
    print("=" * 50 + "\n")

    def process_batch(items: List[dict]) -> List[dict]:
        """Process a batch of items."""
        # Use vmap for efficient batch processing
        batch_enrich = vmap(enrich_item)
        return batch_enrich(items)

    # Process in batches
    batch_size = 3
    for i in range(0, len(demo_data), batch_size):
        batch = demo_data[i : i + batch_size]
        processed_batch = process_batch(batch)
        print(f"Processed batch {i//batch_size + 1}: {len(processed_batch)} items")

    # Part 5: Using the Data API (Conceptual)
    print("\n" + "=" * 50)
    print("Part 5: Using Ember's Data API")
    print("=" * 50 + "\n")

    print("With real datasets, you would use:")
    print("\n1. Streaming (memory efficient):")
    print("   for item in data.stream('dataset_name'):")
    print("       process(item)")

    print("\n2. Loading (when you need all data):")
    print("   dataset = data.load('dataset_name', split='train')")

    print("\n3. Chaining operations:")
    print("   results = (data.stream('dataset')")
    print("             .filter(lambda x: x['score'] > 0.5)")
    print("             .transform(add_metadata)")
    print("             .first(100))")

    print("\n4. Custom data sources:")
    print("   data.register('my_data', MyDataSource())")
    print("   for item in data.stream('my_data'):")
    print("       process(item)")

    # Part 6: Performance with JIT
    print("\n" + "=" * 50)
    print("Part 6: Optimizing with @jit")
    print("=" * 50 + "\n")

    @jit
    def analyze_text(item: dict) -> dict:
        """Analyze text with JIT optimization."""
        text = item["text"]
        words = text.split()

        # Simulate complex analysis
        features = {
            "length": len(text),
            "words": len(words),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "has_ml_terms": any(
                term in text.lower() for term in ["ml", "ai", "learning"]
            ),
            "sentiment_hint": (
                "positive"
                if any(w in text.lower() for w in ["love", "great", "amazing"])
                else "neutral"
            ),
        }

        return {**item, "analysis": features}

    # Process with JIT
    print("Analyzing items with JIT optimization:")
    analyzed = [analyze_text(item) for item in demo_data[:3]]

    for item in analyzed:
        print(f"\nID {item['id']}: {item['text'][:30]}...")
        print(f"  Analysis: {item['analysis']}")

    # Part 7: Real-World Pattern
    print("\n" + "=" * 50)
    print("Part 7: Real-World Data Pipeline")
    print("=" * 50 + "\n")

    def create_data_pipeline():
        """Create a complete data processing pipeline."""

        @jit
        def preprocess(item: dict) -> dict:
            """Preprocess text data."""
            text = item.get("text", "").strip().lower()
            return {**item, "processed_text": text, "tokens": text.split()}

        @jit
        def extract_features(item: dict) -> dict:
            """Extract features from preprocessed data."""
            tokens = item.get("tokens", [])
            return {
                **item,
                "features": {
                    "token_count": len(tokens),
                    "unique_tokens": len(set(tokens)),
                    "avg_token_length": (
                        sum(len(t) for t in tokens) / len(tokens) if tokens else 0
                    ),
                },
            }

        def pipeline(data_source):
            """Complete pipeline."""
            results = []
            for item in data_source:
                # Step 1: Preprocess
                preprocessed = preprocess(item)

                # Step 2: Extract features
                with_features = extract_features(preprocessed)

                # Step 3: Filter (only items with enough tokens)
                if with_features["features"]["token_count"] >= 3:
                    results.append(with_features)

            return results

        return pipeline

    # Use the pipeline
    pipeline = create_data_pipeline()
    results = pipeline(demo_data)

    print(f"Pipeline processed {len(results)} items")
    print("\nSample result:")
    if results:
        sample = results[0]
        print_example_output("Original", sample["text"])
        print_example_output("Tokens", sample["tokens"][:5])
        print_example_output("Features", sample["features"])

    # Summary
    print("\n" + "=" * 50)
    print("✅ Key Takeaways")
    print("=" * 50)

    print("\n1. Use streaming for large datasets:")
    print("   - data.stream() for memory efficiency")
    print("   - Process items one at a time")

    print("\n2. Simple functions for data processing:")
    print("   - No complex classes needed")
    print("   - Compose functions naturally")

    print("\n3. Optimize with @jit and vmap:")
    print("   - @jit for single-item processing")
    print("   - vmap for batch operations")

    print("\n4. Chain operations fluently:")
    print("   - Filter → Transform → Aggregate")
    print("   - Clean, readable pipelines")

    print("\n5. Register custom data sources:")
    print("   - Integrate any data format")
    print("   - Consistent API for all sources")

    print("\nNext: See streaming_data.py for advanced streaming patterns!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
