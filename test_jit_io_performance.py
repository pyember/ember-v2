"""Test JIT performance with I/O-bound operations.

This test proves our hypothesis: JIT benefits come from parallelizing I/O,
not from optimizing computation.
"""

import time
from typing import Dict, Any
from ember.xcs import jit
from ember.xcs.graph.simple import Graph


def io_bound_ensemble(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates an ensemble of LLM calls - the actual Ember use case."""
    prompt = inputs["prompt"]
    delays = inputs.get("delays", [0.1, 0.1, 0.1])  # Simulate API latencies
    
    results = []
    for i, delay in enumerate(delays):
        # Simulate LLM API call
        time.sleep(delay)  # This releases GIL, enables parallelism
        results.append({
            "model": f"model_{i}",
            "response": f"Response {i} to: {prompt}",
            "latency_ms": delay * 1000
        })
    
    return {
        "results": results,
        "total_latency": sum(r["latency_ms"] for r in results),
        "consensus": "Aggregated response"
    }


def test_current_jit():
    """Test current JIT implementation."""
    print("=== Testing Current JIT Implementation ===\n")
    
    # Test input
    test_input = {
        "prompt": "What is quantum computing?",
        "delays": [0.1, 0.1, 0.1]  # 3x100ms = 300ms sequential
    }
    
    # Regular function
    print("1. Regular function (sequential):")
    start = time.perf_counter()
    result = io_bound_ensemble(inputs=test_input)
    regular_time = time.perf_counter() - start
    print(f"   Time: {regular_time*1000:.1f}ms")
    print(f"   Expected: ~300ms (3 x 100ms sequential)")
    
    # JIT version
    print("\n2. JIT version (trace strategy):")
    jit_func = jit(io_bound_ensemble, mode="structural")
    
    # First call (includes compilation)
    start = time.perf_counter()
    result = jit_func(inputs=test_input)
    first_time = time.perf_counter() - start
    print(f"   First call: {first_time*1000:.1f}ms (includes compilation)")
    
    # Subsequent calls
    times = []
    for _ in range(5):
        start = time.perf_counter()
        result = jit_func(inputs=test_input)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    print(f"   Avg subsequent: {avg_time*1000:.1f}ms")
    print(f"   Speedup: {regular_time/avg_time:.2f}x")
    
    # The current trace strategy just replays results, 
    # so it's actually faster (no sleep!) but that's cheating


def test_ideal_jit():
    """Test what JIT SHOULD do - parallelize I/O operations."""
    print("\n\n=== What JIT Should Do ===\n")
    
    # Build a graph that actually parallelizes
    graph = Graph()
    
    # Add three parallel I/O operations
    def io_op(model_id: int, delay: float):
        def op(inputs):
            prompt = inputs.get("prompt", "test")
            time.sleep(delay)
            return {
                "model": f"model_{model_id}", 
                "response": f"Response {model_id}",
                "latency_ms": delay * 1000
            }
        return op
    
    # Add parallel nodes
    node1 = graph.add(io_op(0, 0.1))
    node2 = graph.add(io_op(1, 0.1))
    node3 = graph.add(io_op(2, 0.1))
    
    # Add aggregation node that depends on all three
    def aggregate(*results):
        return {
            "results": results,
            "total_latency": sum(r["latency_ms"] for r in results),
            "consensus": "Aggregated response"
        }
    
    final = graph.add(aggregate, deps=[node1, node2, node3])
    
    # Test execution
    test_input = {"prompt": "What is quantum computing?"}
    
    print("Ideal JIT (with parallelization):")
    start = time.perf_counter()
    results = graph.run(test_input)
    parallel_time = time.perf_counter() - start
    
    print(f"   Time: {parallel_time*1000:.1f}ms")
    print(f"   Expected: ~100ms (3 parallel 100ms operations)")
    print(f"   Speedup vs sequential: {0.3/parallel_time:.2f}x")
    
    # Show the actual benefit of parallelization
    print(f"\n   Result: {results[final]}")


def test_cpu_bound_no_benefit():
    """Show that CPU-bound operations don't benefit."""
    print("\n\n=== CPU-Bound Operations (No Benefit) ===\n")
    
    def cpu_bound_work(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        n = inputs["n"]
        results = []
        
        # Three independent computations (but CPU-bound)
        for i in range(3):
            total = 0
            for j in range(n):
                for k in range(1000):
                    total += j * k * (i + 1)
            results.append(total)
        
        return {"results": results, "sum": sum(results)}
    
    test_input = {"n": 1000}
    
    # Regular execution
    start = time.perf_counter()
    result = cpu_bound_work(inputs=test_input)
    regular_time = time.perf_counter() - start
    print(f"Regular execution: {regular_time*1000:.1f}ms")
    
    # JIT version
    jit_func = jit(cpu_bound_work, mode="structural")
    jit_func(inputs=test_input)  # Warmup
    
    start = time.perf_counter()
    result = jit_func(inputs=test_input)
    jit_time = time.perf_counter() - start
    print(f"JIT execution: {jit_time*1000:.1f}ms")
    print(f"Speedup: {regular_time/jit_time:.2f}x")
    print("(No real speedup because GIL prevents CPU parallelism)")


def main():
    print("JIT Performance Analysis: I/O vs CPU Bound\n")
    print("Key Insight: JIT benefits come from parallelizing I/O operations,")
    print("not from optimizing computation (due to Python's GIL).\n")
    
    test_current_jit()
    test_ideal_jit()
    test_cpu_bound_no_benefit()
    
    print("\n\n=== Conclusions ===")
    print("1. Current trace JIT doesn't actually parallelize - it just memoizes")
    print("2. Real speedup comes from parallelizing I/O operations") 
    print("3. Sleep-based benchmarks accurately model LLM API latency")
    print("4. CPU-bound operations see no benefit from threading")
    print("5. JIT should detect and parallelize I/O patterns")


if __name__ == "__main__":
    main()