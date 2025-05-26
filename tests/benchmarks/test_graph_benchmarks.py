"""Graph performance benchmarks.

Measure what matters. No vanity metrics.
"""

import time
import statistics
from ember.xcs.graph import Graph


def benchmark(name: str, func, warmup=10, iterations=100):
    """Simple, accurate benchmarking."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    
    # Report
    mean_ms = statistics.mean(times) * 1000
    median_ms = statistics.median(times) * 1000
    min_ms = min(times) * 1000
    
    print(f"{name}:")
    print(f"  Median: {median_ms:.3f}ms")
    print(f"  Mean: {mean_ms:.3f}ms") 
    print(f"  Min: {min_ms:.3f}ms")
    print()


def test_construction_speed():
    """How fast can we build graphs?"""
    print("\n=== Construction Speed ===\n")
    
    def small_graph():
        g = Graph()
        for i in range(10):
            g.add(lambda x=i: x * 2)
    
    def medium_graph():
        g = Graph()
        prev = None
        for i in range(100):
            deps = [prev] if prev else []
            prev = g.add(lambda x=i: x + 1, deps=deps)
    
    def large_graph():
        g = Graph()
        # Fan-out, fan-in pattern
        sources = [g.add(lambda x=i: x) for i in range(100)]
        for i in range(0, 100, 10):
            deps = sources[i:i+10]
            g.add(lambda *args: sum(args), deps=deps)
    
    benchmark("Small (10 nodes)", small_graph, iterations=1000)
    benchmark("Medium (100 nodes)", medium_graph, iterations=100)
    benchmark("Large (110 nodes)", large_graph, iterations=100)


def test_execution_speed():
    """How fast do graphs execute?"""
    print("\n=== Execution Speed ===\n")
    
    # Sequential pipeline
    seq_graph = Graph()
    prev = None
    for i in range(50):
        deps = [prev] if prev else []
        prev = seq_graph.add(lambda x=i: x + 1 if deps else 0, deps=deps)
    
    # Parallel graph
    par_graph = Graph()
    sources = [par_graph.add(lambda x=i: x) for i in range(50)]
    par_graph.add(lambda *args: sum(args), deps=sources)
    
    # Complex DAG
    dag_graph = Graph()
    layer1 = [dag_graph.add(lambda x=i: x) for i in range(20)]
    layer2 = []
    for i in range(10):
        deps = layer1[i*2:(i+1)*2]
        node = dag_graph.add(lambda a, b: a + b, deps=deps)
        layer2.append(node)
    dag_graph.add(lambda *args: sum(args), deps=layer2)
    
    benchmark("Sequential (50 nodes)", lambda: seq_graph.run())
    benchmark("Parallel (51 nodes)", lambda: par_graph.run())
    benchmark("DAG (31 nodes)", lambda: dag_graph.run())


def test_scaling():
    """How does performance scale with size?"""
    print("\n=== Scaling Analysis ===\n")
    
    sizes = [10, 50, 100, 500, 1000]
    construction_times = []
    execution_times = []
    
    for size in sizes:
        # Build graph
        start = time.perf_counter()
        g = Graph()
        nodes = [g.add(lambda x=i: x * 2) for i in range(size)]
        g.add(lambda *args: sum(args), deps=nodes)
        construction_time = time.perf_counter() - start
        construction_times.append(construction_time)
        
        # Execute graph
        times = []
        for _ in range(10):
            start = time.perf_counter()
            g.run()
            times.append(time.perf_counter() - start)
        execution_time = statistics.median(times)
        execution_times.append(execution_time)
        
        print(f"Size {size}:")
        print(f"  Build: {construction_time*1000:.3f}ms")
        print(f"  Run: {execution_time*1000:.3f}ms")
        print(f"  Per-node overhead: {execution_time/size*1000000:.1f}μs")
        print()


def test_parallel_speedup():
    """Measure actual parallelism benefit."""
    print("\n=== Parallel Speedup ===\n")
    
    # CPU-bound work function
    def work(n=100000):
        total = 0
        for i in range(n):
            total += i * i
        return total
    
    # Build graph with parallel work
    g = Graph()
    nodes = [g.add(lambda n=100000: work(n)) for _ in range(8)]
    final = g.add(lambda *args: sum(args), deps=nodes)
    
    # Force sequential
    seq_times = []
    saved_func = g._run_parallel
    g._run_parallel = g._run_sequential
    for _ in range(5):
        start = time.perf_counter()
        g.run()
        seq_times.append(time.perf_counter() - start)
    g._run_parallel = saved_func
    
    # Normal (parallel) execution
    par_times = []
    for _ in range(5):
        start = time.perf_counter()
        g.run()
        par_times.append(time.perf_counter() - start)
    
    seq_median = statistics.median(seq_times)
    par_median = statistics.median(par_times)
    speedup = seq_median / par_median
    
    print(f"Sequential: {seq_median*1000:.1f}ms")
    print(f"Parallel: {par_median*1000:.1f}ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {speedup/8*100:.1f}%")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("GRAPH PERFORMANCE BENCHMARKS")
    print("="*50)
    
    test_construction_speed()
    test_execution_speed()
    test_scaling()
    test_parallel_speedup()
    
    print("\n" + "="*50)
    print("Benchmarks complete.")
    print("="*50)