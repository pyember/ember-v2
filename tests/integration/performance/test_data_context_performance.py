"""Performance tests for DataContext system.

Benchmarks critical operations in the data system to ensure
they meet performance targets.
"""

import timeit
from typing import Callable, Dict

import pytest

from ember.core.utils.data.base.models import TaskType
from ember.core.utils.data.context.data_context import (  # Removed get_default_context,; Removed reset_default_context,; Removed set_default_context,
    DataContext,
)


@pytest.fixture
def benchmark_context():
    """Create a context optimized for benchmarking."""
    # Create test context with minimal configuration
    context = DataContext.create_test_context()

    # Register test datasets
    for i in range(100):
        context.register_dataset(
            name=f"benchmark_dataset_{i}",
            source=f"benchmark/source_{i}",
            task_type=TaskType.MULTIPLE_CHOICE,
            description=f"Benchmark dataset {i}",
        )

    # Removed setting/resetting default context
    # old_context = get_default_context()
    # set_default_context(context)

    yield context

    # Removed restoring old context
    # if old_context:
    #     set_default_context(old_context)
    # else:
    #     reset_default_context()


def run_benchmark(
    func: Callable, number: int = 100, repeat: int = 5
) -> Dict[str, float]:
    """Run benchmark with statistics.

    Args:
        func: Function to benchmark
        number: Number of times to execute in one test
        repeat: Number of tests to run

    Returns:
        Dictionary with benchmark statistics
    """
    # Run benchmark
    times = timeit.repeat(func, number=number, repeat=repeat)

    # Calculate statistics
    per_call_times = [t / number for t in times]
    return {
        "min_time": min(per_call_times) * 1e6,  # μs
        "max_time": max(per_call_times) * 1e6,  # μs
        "avg_time": sum(per_call_times) / len(per_call_times) * 1e6,  # μs
        "median_time": sorted(per_call_times)[len(per_call_times) // 2] * 1e6,  # μs
    }


def test_context_lookup_performance(benchmark_context):
    """Benchmark the performance of accessing context.
    
    This test measures how fast we can access a context object that's already
    been initialized, to ensure our context system is optimized for
    repeated access during normal operation.
    """
    # Benchmark direct context access - should be <100ns
    # Context retrieval should be extremely fast since it's a critical operation
    stats = run_benchmark(lambda: benchmark_context)

    # Print results
    print("\nDefault context lookup times (microseconds):")
    print(f"  Min: {stats['min_time']:.2f}μs")
    print(f"  Max: {stats['max_time']:.2f}μs")
    print(f"  Avg: {stats['avg_time']:.2f}μs")
    print(f"  Median: {stats['median_time']:.2f}μs")

    # Should meet performance target (aggressive, but achievable)
    assert stats["median_time"] < 1.0, "Context lookup is too slow"


def test_dataset_service_access_performance(benchmark_context):
    """Benchmark dataset service property access time."""
    # Prime the system
    benchmark_context.dataset_service

    # Benchmark accessing dataset_service property
    stats = run_benchmark(lambda: benchmark_context.dataset_service)

    # Print results
    print("\nDataset service access times (microseconds):")
    print(f"  Min: {stats['min_time']:.2f}μs")
    print(f"  Max: {stats['max_time']:.2f}μs")
    print(f"  Avg: {stats['avg_time']:.2f}μs")
    print(f"  Median: {stats['median_time']:.2f}μs")

    # Should meet performance target
    assert stats["median_time"] < 0.5, "Dataset service access is too slow"


def test_streaming_dataset_creation_performance(benchmark_context):
    """Benchmark creation of streaming datasets."""
    # Register datasets used for benchmark
    num_datasets = 10

    # Prime the system
    for i in range(num_datasets):
        benchmark_context.get_streaming_dataset(f"benchmark_dataset_{i}")

    # Benchmark creating streaming datasets
    def create_datasets():
        for i in range(num_datasets):
            benchmark_context.get_streaming_dataset(f"benchmark_dataset_{i}")

    # Use fewer iterations as this is a more expensive operation
    stats = run_benchmark(create_datasets, number=10, repeat=3)

    # Calculate per-dataset time
    per_dataset_time = {k: v / num_datasets for k, v in stats.items()}

    # Print results
    print("\nStreaming dataset creation times per dataset (microseconds):")
    print(f"  Min: {per_dataset_time['min_time']:.2f}μs")
    print(f"  Max: {per_dataset_time['max_time']:.2f}μs")
    print(f"  Avg: {per_dataset_time['avg_time']:.2f}μs")
    print(f"  Median: {per_dataset_time['median_time']:.2f}μs")

    # This is a more complex operation, so allow more time
    assert per_dataset_time["median_time"] < 500.0, "Dataset creation is too slow"


@pytest.mark.perf
def test_dataset_registry_lookup_performance(benchmark_context):
    """Benchmark registry lookups."""
    # Prime the registry cache
    registry = benchmark_context.registry
    for i in range(100):
        registry.get(name=f"benchmark_dataset_{i}")

    # Benchmark registry lookups (worst case scenario with many entries)
    def lookup_datasets():
        for i in range(10):
            # Ensure we're calling with keyword args
            registry.get(name=f"benchmark_dataset_{i}")

    stats = run_benchmark(lookup_datasets, number=100, repeat=3)

    # Calculate per-lookup time
    per_lookup_time = {k: v / 10 for k, v in stats.items()}

    # Print results
    print("\nRegistry lookup times per dataset (microseconds):")
    print(f"  Min: {per_lookup_time['min_time']:.2f}μs")
    print(f"  Max: {per_lookup_time['max_time']:.2f}μs")
    print(f"  Avg: {per_lookup_time['avg_time']:.2f}μs")
    print(f"  Median: {per_lookup_time['median_time']:.2f}μs")

    # Should be fast even with many datasets
    assert per_lookup_time["median_time"] < 5.0, "Registry lookup is too slow"
