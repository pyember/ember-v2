"""Streaming Data - Processing large datasets efficiently.

Learn how to handle large datasets with streaming, batching, and
efficient memory management for AI applications.

Example:
    >>> from ember.api import data
    >>> dataset = data.load_dataset("large_corpus", streaming=True)
    >>> for batch in dataset.batch(32):
    ...     process_batch(batch)
"""

import sys
from pathlib import Path
from typing import Iterator, List, Dict, Any
import time

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def example_basic_streaming():
    """Show basic streaming concepts."""
    print("\n=== Basic Streaming ===\n")

    print("Why streaming matters:")
    print("  • Process data that doesn't fit in memory")
    print("  • Start processing before full download")
    print("  • Reduce memory footprint")
    print("  • Enable real-time processing\n")

    # Simulate streaming
    def data_stream(size: int) -> Iterator[str]:
        """Simulate a data stream."""
        for i in range(size):
            yield f"Record {i}: Sample data entry"
            if i < 5:  # Show first few
                print(f"  Streaming: Record {i}")

    print("Streaming 1000 records:")
    stream = data_stream(1000)

    # Process first few items
    for _ in range(3):
        next(stream)

    print("  ... (continues streaming)")


def example_batch_processing():
    """Demonstrate batch processing patterns."""
    print("\n\n=== Batch Processing ===\n")

    def batch_iterator(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
        """Create batches from items."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    # Simulate data
    data = [f"Item_{i}" for i in range(100)]

    print("Processing 100 items in batches of 16:")
    print()

    batch_count = 0
    total_processed = 0

    for batch in batch_iterator(data, 16):
        batch_count += 1
        total_processed += len(batch)

        if batch_count <= 3:
            print(f"  Batch {batch_count}: {len(batch)} items")
        elif batch_count == 4:
            print("  ...")

    print(f"\nTotal: {batch_count} batches, {total_processed} items processed")


def example_memory_efficient_loading():
    """Show memory-efficient loading techniques."""
    print("\n\n=== Memory-Efficient Loading ===\n")

    print("Techniques for large datasets:\n")

    print("1. Generator-based loading:")
    print("   def load_data():")
    print("       with open('large_file.txt') as f:")
    print("           for line in f:")
    print("               yield process_line(line)\n")

    print("2. Chunked reading:")
    print("   for chunk in pd.read_csv('data.csv', chunksize=10000):")
    print("       process_chunk(chunk)\n")

    print("3. Memory mapping:")
    print("   data = np.memmap('large_array.dat', dtype='float32', mode='r')")
    print("   # Access data without loading all into RAM\n")

    print("4. Lazy loading:")
    print("   dataset = Dataset('path/to/data', lazy=True)")
    print("   # Data loaded only when accessed")


def example_stream_transformations():
    """Demonstrate stream transformation patterns."""
    print("\n\n=== Stream Transformations ===\n")

    # Transformation pipeline
    def number_stream():
        """Generate numbers."""
        for i in range(10):
            yield i

    def square_transform(stream):
        """Square each number."""
        for num in stream:
            yield num**2

    def filter_even(stream):
        """Keep only even numbers."""
        for num in stream:
            if num % 2 == 0:
                yield num

    print("Stream transformation pipeline:")
    print("  numbers → square → filter_even\n")

    # Apply transformations
    stream = number_stream()
    stream = square_transform(stream)
    stream = filter_even(stream)

    results = list(stream)
    print(f"Input: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
    print(f"After square: 0, 1, 4, 9, 16, 25, 36, 49, 64, 81")
    print(f"After filter: {results}")


def example_parallel_streaming():
    """Show parallel streaming patterns."""
    print("\n\n=== Parallel Streaming ===\n")

    print("Parallel processing strategies:\n")

    print("1. Thread pool for I/O-bound:")
    print("   with ThreadPoolExecutor(max_workers=4) as pool:")
    print("       results = pool.map(fetch_data, urls)")
    print("       for result in results:")
    print("           process(result)\n")

    print("2. Process pool for CPU-bound:")
    print("   with ProcessPoolExecutor() as pool:")
    print("       for result in pool.map(cpu_intensive, data_stream()):")
    print("           save(result)\n")

    print("3. Async streaming:")
    print("   async for item in async_data_stream():")
    print("       result = await process_async(item)")
    print("       await save_async(result)")


def example_windowed_processing():
    """Demonstrate windowed/sliding window processing."""
    print("\n\n=== Windowed Processing ===\n")

    def sliding_window(data: List[Any], window_size: int, stride: int = 1):
        """Create sliding windows over data."""
        for i in range(0, len(data) - window_size + 1, stride):
            yield data[i : i + window_size]

    # Example time series
    time_series = list(range(20))

    print("Sliding window (size=5, stride=2):")
    print(f"Data: {time_series}\n")

    windows = list(sliding_window(time_series, 5, 2))
    for i, window in enumerate(windows[:4]):
        print(f"  Window {i}: {window}")
    print("  ...")

    print(f"\nTotal windows: {len(windows)}")
    print("Use cases: Time series analysis, sequence modeling")


def example_streaming_aggregation():
    """Show streaming aggregation patterns."""
    print("\n\n=== Streaming Aggregation ===\n")

    class StreamingStats:
        """Calculate statistics on streaming data."""

        def __init__(self):
            self.count = 0
            self.sum = 0
            self.sum_sq = 0
            self.min = float("inf")
            self.max = float("-inf")

        def update(self, value):
            self.count += 1
            self.sum += value
            self.sum_sq += value**2
            self.min = min(self.min, value)
            self.max = max(self.max, value)

        @property
        def mean(self):
            return self.sum / self.count if self.count > 0 else 0

        @property
        def variance(self):
            if self.count < 2:
                return 0
            return (self.sum_sq - self.sum**2 / self.count) / (self.count - 1)

    print("Online statistics calculation:")
    stats = StreamingStats()

    # Simulate streaming data
    import random

    random.seed(42)

    print("\nProcessing stream...")
    for i in range(1000):
        value = random.gauss(100, 15)
        stats.update(value)

        if i + 1 in [10, 100, 1000]:
            print(f"  After {i+1} items:")
            print(f"    Mean: {stats.mean:.2f}")
            print(f"    Variance: {stats.variance:.2f}")
            print(f"    Range: [{stats.min:.2f}, {stats.max:.2f}]")


def example_backpressure_handling():
    """Demonstrate backpressure handling in streams."""
    print("\n\n=== Backpressure Handling ===\n")

    print("Managing flow control in streams:\n")

    print("1. Buffer with limits:")
    print("   buffer = collections.deque(maxlen=1000)")
    print("   # Automatically drops oldest when full\n")

    print("2. Rate limiting:")
    print("   rate_limiter = RateLimiter(100)  # 100 items/sec")
    print("   for item in stream:")
    print("       rate_limiter.acquire()")
    print("       process(item)\n")

    print("3. Adaptive batching:")
    print("   batch = []")
    print("   for item in stream:")
    print("       batch.append(item)")
    print("       if len(batch) >= batch_size or timeout_reached:")
    print("           process_batch(batch)")
    print("           batch = []")


def example_streaming_pipeline():
    """Show complete streaming pipeline example."""
    print("\n\n=== Complete Streaming Pipeline ===\n")

    print("End-to-end streaming pipeline:\n")

    # Pipeline stages
    stages = [
        "1. Source: Read from files/API/database",
        "2. Parse: Extract structured data",
        "3. Filter: Remove invalid records",
        "4. Transform: Normalize and enrich",
        "5. Batch: Group for efficient processing",
        "6. Process: Apply ML models",
        "7. Aggregate: Compute statistics",
        "8. Sink: Write results",
    ]

    for stage in stages:
        print(f"  {stage}")

    print("\nExample metrics:")
    print("  • Input rate: 10,000 records/sec")
    print("  • Processing latency: 50ms/batch")
    print("  • Memory usage: 500MB constant")
    print("  • Throughput: 9,500 records/sec")


def main():
    """Run all streaming data examples."""
    print_section_header("Streaming Data Processing")

    print("🎯 Streaming for AI Applications:\n")
    print("• Handle datasets larger than memory")
    print("• Process data in real-time")
    print("• Reduce latency and memory usage")
    print("• Enable continuous learning")

    example_basic_streaming()
    example_batch_processing()
    example_memory_efficient_loading()
    example_stream_transformations()
    example_parallel_streaming()
    example_windowed_processing()
    example_streaming_aggregation()
    example_backpressure_handling()
    example_streaming_pipeline()

    print("\n" + "=" * 50)
    print("✅ Streaming Best Practices")
    print("=" * 50)
    print("\n1. Use generators for memory efficiency")
    print("2. Process in batches for performance")
    print("3. Implement proper error handling")
    print("4. Monitor memory usage and throughput")
    print("5. Handle backpressure gracefully")
    print("6. Use appropriate buffer sizes")
    print("7. Consider parallelism carefully")

    print("\n🔧 Tools & Libraries:")
    print("• itertools - Built-in iteration tools")
    print("• asyncio - Async streaming")
    print("• pandas chunks - Large CSV processing")
    print("• Apache Beam - Distributed streaming")
    print("• Dask - Parallel computing")

    print("\nNext: Explore practical patterns in '../09_practical_patterns/'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
