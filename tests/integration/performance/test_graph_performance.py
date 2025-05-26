"""Performance benchmarks for the new simplified Graph implementation."""

import time
import statistics
from typing import List, Tuple
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ember.xcs.graph.graph import Graph


class BenchmarkResult:
    """Stores benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []
    
    def add(self, time: float):
        self.times.append(time)
    
    def report(self) -> str:
        if not self.times:
            return f"{self.name}: No data"
        
        return (
            f"{self.name}:\n"
            f"  Mean: {statistics.mean(self.times)*1000:.2f}ms\n"
            f"  Median: {statistics.median(self.times)*1000:.2f}ms\n"
            f"  Min: {min(self.times)*1000:.2f}ms\n"
            f"  Max: {max(self.times)*1000:.2f}ms\n"
            f"  Stdev: {statistics.stdev(self.times)*1000:.2f}ms" if len(self.times) > 1 else ""
        )


def benchmark(func, iterations: int = 100) -> BenchmarkResult:
    """Run a benchmark and return results."""
    result = BenchmarkResult(func.__name__)
    
    # Warmup
    for _ in range(10):
        func()
    
    # Actual benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        result.add(time.perf_counter() - start)
    
    return result


class TestGraphPerformance:
    """Performance benchmarks for Graph operations."""
    
    def test_graph_construction_performance(self):
        """Benchmark graph construction speed."""
        
        def construct_small_graph():
            graph = Graph()
            for i in range(10):
                graph.add(lambda x=i: x * 2, name=f"n{i}")
            return graph
        
        def construct_medium_graph():
            graph = Graph()
            nodes = []
            for i in range(100):
                node = graph.add(lambda x=i: x * 2, name=f"n{i}")
                nodes.append(node)
                if i > 0 and i % 10 == 0:
                    # Add aggregation every 10 nodes
                    deps = nodes[-10:]
                    graph.add(lambda *args: sum(args), deps=deps, name=f"agg{i}")
            return graph
        
        def construct_large_graph():
            graph = Graph()
            # Create 1000 nodes with complex dependencies
            layers = []
            
            # Input layer
            layer = []
            for i in range(100):
                node = graph.add(lambda x=i: x, name=f"input_{i}")
                layer.append(node)
            layers.append(layer)
            
            # Hidden layers
            for l in range(5):
                prev_layer = layers[-1]
                layer = []
                for i in range(50):
                    # Each node depends on 5 random nodes from previous layer
                    deps = prev_layer[i*2:(i+1)*2] if i*2 < len(prev_layer) else prev_layer[-2:]
                    node = graph.add(lambda *args: sum(args), deps=deps, name=f"hidden_{l}_{i}")
                    layer.append(node)
                layers.append(layer)
            
            # Output layer
            graph.add(lambda *args: sum(args), deps=layers[-1], name="output")
            return graph
        
        # Run benchmarks
        small_result = benchmark(construct_small_graph, 1000)
        medium_result = benchmark(construct_medium_graph, 100)
        large_result = benchmark(construct_large_graph, 10)
        
        print("\n=== Graph Construction Performance ===")
        print(small_result.report())
        print(medium_result.report())
        print(large_result.report())
        
        # Assert reasonable performance
        assert statistics.mean(small_result.times) < 0.001  # < 1ms
        assert statistics.mean(medium_result.times) < 0.01  # < 10ms
        assert statistics.mean(large_result.times) < 0.5    # < 500ms
    
    def test_pattern_detection_performance(self):
        """Benchmark pattern detection speed."""
        
        def create_map_pattern_graph():
            graph = Graph()
            def transform(x): return x * 2
            
            # Create 100 map operations
            for i in range(100):
                graph.add(transform, args=[i], name=f"map_{i}")
            
            return graph
        
        def create_reduce_pattern_graph():
            graph = Graph()
            
            # Create 100 sources feeding into 10 reducers
            sources = []
            for i in range(100):
                node = graph.add(lambda x=i: x, name=f"source_{i}")
                sources.append(node)
            
            for i in range(10):
                deps = sources[i*10:(i+1)*10]
                graph.add(lambda *args: sum(args), deps=deps, name=f"reduce_{i}")
            
            return graph
        
        def create_ensemble_pattern_graph():
            graph = Graph()
            def model(x): return x * 2
            
            # Create 10 ensembles of 10 models each
            for e in range(10):
                source = graph.add(lambda x=e: x, name=f"input_{e}")
                models = []
                for m in range(10):
                    node = graph.add(model, deps=[source], name=f"model_{e}_{m}")
                    models.append(node)
                graph.add(lambda *args: max(args), deps=models, name=f"judge_{e}")
            
            return graph
        
        # Create graphs
        map_graph = create_map_pattern_graph()
        reduce_graph = create_reduce_pattern_graph()
        ensemble_graph = create_ensemble_pattern_graph()
        
        # Benchmark pattern detection
        def detect_map_patterns():
            map_graph._detect_patterns()
        
        def detect_reduce_patterns():
            reduce_graph._detect_patterns()
        
        def detect_ensemble_patterns():
            ensemble_graph._detect_patterns()
        
        map_result = benchmark(detect_map_patterns, 100)
        reduce_result = benchmark(detect_reduce_patterns, 100)
        ensemble_result = benchmark(detect_ensemble_patterns, 100)
        
        print("\n=== Pattern Detection Performance ===")
        print(map_result.report())
        print(reduce_result.report())
        print(ensemble_result.report())
        
        # Assert reasonable performance
        assert statistics.mean(map_result.times) < 0.01      # < 10ms
        assert statistics.mean(reduce_result.times) < 0.01   # < 10ms
        assert statistics.mean(ensemble_result.times) < 0.02 # < 20ms
    
    def test_wave_computation_performance(self):
        """Benchmark wave computation for parallel scheduling."""
        
        def create_sequential_graph():
            graph = Graph()
            prev = None
            for i in range(100):
                deps = [prev] if prev else []
                prev = graph.add(lambda x=i: x + 1, deps=deps, name=f"seq_{i}")
            return graph
        
        def create_parallel_graph():
            graph = Graph()
            # Create 10 waves of 10 parallel nodes each
            for w in range(10):
                if w == 0:
                    for i in range(10):
                        graph.add(lambda x=i: x, name=f"wave_{w}_{i}")
                else:
                    # Each node in this wave depends on one from previous
                    for i in range(10):
                        dep = f"wave_{w-1}_{i}"
                        graph.add(lambda x: x + 1, deps=[dep], name=f"wave_{w}_{i}")
            return graph
        
        def create_complex_dag():
            graph = Graph()
            # Diamond patterns and reconvergence
            for d in range(20):
                top = graph.add(lambda x=d: x, name=f"top_{d}")
                left = graph.add(lambda x: x * 2, deps=[top], name=f"left_{d}")
                right = graph.add(lambda x: x + 5, deps=[top], name=f"right_{d}")
                graph.add(lambda a, b: a + b, deps=[left, right], name=f"bottom_{d}")
            return graph
        
        seq_graph = create_sequential_graph()
        par_graph = create_parallel_graph()
        dag_graph = create_complex_dag()
        
        def compute_seq_waves():
            seq_graph._compute_waves()
        
        def compute_par_waves():
            par_graph._compute_waves()
        
        def compute_dag_waves():
            dag_graph._compute_waves()
        
        seq_result = benchmark(compute_seq_waves, 100)
        par_result = benchmark(compute_par_waves, 100)
        dag_result = benchmark(compute_dag_waves, 100)
        
        print("\n=== Wave Computation Performance ===")
        print(seq_result.report())
        print(par_result.report())
        print(dag_result.report())
        
        # Verify correctness
        seq_waves = seq_graph._compute_waves()
        assert len(seq_waves) == 100  # Fully sequential
        
        par_waves = par_graph._compute_waves()
        assert len(par_waves) == 10   # 10 parallel waves
        
        dag_waves = dag_graph._compute_waves()
        assert len(dag_waves) == 4    # Diamond pattern creates 4 waves
        
        # Assert performance
        assert statistics.mean(seq_result.times) < 0.01  # < 10ms
        assert statistics.mean(par_result.times) < 0.01  # < 10ms
        assert statistics.mean(dag_result.times) < 0.01  # < 10ms
    
    def test_execution_performance(self):
        """Benchmark actual graph execution."""
        
        def create_compute_graph():
            graph = Graph()
            
            # Simulate data processing pipeline
            # Load phase (parallel)
            load_nodes = []
            for i in range(5):
                node = graph.add(lambda x=i: {"id": x, "data": list(range(10))}, name=f"load_{i}")
                load_nodes.append(node)
            
            # Transform phase (parallel)
            transform_nodes = []
            for i, load_node in enumerate(load_nodes):
                node = graph.add(
                    lambda x: {"id": x["id"], "data": [v * 2 for v in x["data"]]},
                    deps=[load_node],
                    name=f"transform_{i}"
                )
                transform_nodes.append(node)
            
            # Aggregate phase
            graph.add(
                lambda *args: {"total": sum(sum(x["data"]) for x in args)},
                deps=transform_nodes,
                name="aggregate"
            )
            
            return graph
        
        graph = create_compute_graph()
        
        # Benchmark sequential execution
        def run_sequential():
            return graph({}, parallel=False)
        
        # Benchmark parallel execution
        def run_parallel():
            return graph({}, parallel=True)
        
        # Clear cache between runs for fair comparison
        graph._execution_cache.clear()
        seq_result = benchmark(run_sequential, 50)
        
        graph._execution_cache.clear()
        par_result = benchmark(run_parallel, 50)
        
        print("\n=== Execution Performance ===")
        print(seq_result.report())
        print(par_result.report())
        
        # Calculate speedup
        seq_mean = statistics.mean(seq_result.times)
        par_mean = statistics.mean(par_result.times)
        speedup = seq_mean / par_mean
        
        print(f"\nParallel speedup: {speedup:.2f}x")
        
        # Parallel should be faster for this workload
        assert speedup > 1.0, "Parallel execution should be faster"
        
        # Both should complete quickly
        assert seq_mean < 0.01  # < 10ms
        assert par_mean < 0.01  # < 10ms
    
    def test_analysis_caching_impact(self):
        """Test the impact of caching analysis results."""
        
        graph = Graph()
        
        # Build medium complexity graph
        for i in range(50):
            if i == 0:
                graph.add(lambda: 1, name=f"n{i}")
            else:
                graph.add(lambda x: x + 1, deps=[f"n{i-1}"], name=f"n{i}")
        
        # First execution (cold)
        graph._execution_cache.clear()
        start = time.perf_counter()
        result1 = graph({})
        cold_time = time.perf_counter() - start
        
        # Second execution (warm cache)
        start = time.perf_counter()
        result2 = graph({})
        warm_time = time.perf_counter() - start
        
        # With cache=False (forces recomputation)
        start = time.perf_counter()
        result3 = graph({}, cache=False)
        no_cache_time = time.perf_counter() - start
        
        print("\n=== Caching Impact ===")
        print(f"Cold execution: {cold_time*1000:.2f}ms")
        print(f"Warm execution: {warm_time*1000:.2f}ms")
        print(f"No cache execution: {no_cache_time*1000:.2f}ms")
        print(f"Cache speedup: {cold_time/warm_time:.1f}x")
        
        # Verify results are identical
        assert result1 == result2 == result3
        
        # Warm should be significantly faster
        assert warm_time < cold_time * 0.5, "Cache should provide significant speedup"


def test_performance_summary():
    """Run all performance tests and generate summary."""
    suite = TestGraphPerformance()
    
    print("\n" + "="*60)
    print("GRAPH PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    
    suite.test_graph_construction_performance()
    suite.test_pattern_detection_performance()
    suite.test_wave_computation_performance()
    suite.test_execution_performance()
    suite.test_analysis_caching_impact()
    
    print("\n" + "="*60)
    print("All performance benchmarks passed!")
    print("="*60)


if __name__ == "__main__":
    test_performance_summary()