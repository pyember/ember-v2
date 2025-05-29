"""Test that JIT compilation actually provides performance benefits."""

import time
from typing import Dict, Any
import pytest

from ember.xcs import jit
from ember.core.registry.operator.base.operator_base import Operator


class SimpleOperator(Operator):
    """A simple operator for testing."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Simple computation
        result = 0
        for i in range(100):
            result += inputs["value"] * i
        return {"result": result}


class ComplexOperator(Operator):
    """A complex operator with nested loops."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # More complex computation with nested loops
        result = 0
        for i in range(50):
            for j in range(50):
                result += inputs["value"] * (i + j)
        return {"result": result}


class EnsembleOperator(Operator):
    """An ensemble operator that should benefit from parallelization."""
    
    def __init__(self, num_models=5):
        super().__init__()
        self.num_models = num_models
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        for i in range(self.num_models):
            # Simulate independent model computations
            model_result = 0
            for j in range(100):
                model_result += inputs["value"] * (i + j)
            results.append(model_result)
        return {"ensemble_results": results}


def benchmark_operator(operator_class, jit_mode=None, num_warmup=5, num_iterations=50):
    """Benchmark an operator with and without JIT."""
    # Create instances
    if jit_mode:
        jit_op = jit(operator_class, mode=jit_mode)()
    else:
        jit_op = jit(operator_class)()
    
    regular_op = operator_class()
    
    # Test input
    test_input = {"value": 42}
    
    # Warmup JIT (trigger compilation)
    for _ in range(num_warmup):
        jit_op(inputs=test_input)
    
    # Benchmark JIT version
    jit_start = time.perf_counter()
    for _ in range(num_iterations):
        jit_result = jit_op(inputs=test_input)
    jit_duration = time.perf_counter() - jit_start
    
    # Benchmark regular version
    regular_start = time.perf_counter()
    for _ in range(num_iterations):
        regular_result = regular_op(inputs=test_input)
    regular_duration = time.perf_counter() - regular_start
    
    # Verify results are the same
    assert jit_result == regular_result, "JIT and regular results should match"
    
    # Calculate speedup
    speedup = regular_duration / jit_duration
    
    return {
        "jit_duration": jit_duration,
        "regular_duration": regular_duration,
        "speedup": speedup,
        "jit_per_call": jit_duration / num_iterations,
        "regular_per_call": regular_duration / num_iterations,
    }


@pytest.mark.parametrize("operator_class,expected_min_speedup", [
    (SimpleOperator, 0.8),  # At least 80% as fast (small overhead acceptable)
    (ComplexOperator, 0.9),  # Should be closer to regular speed
    (EnsembleOperator, 1.0),  # Should match or exceed regular speed
])
def test_jit_performance(operator_class, expected_min_speedup):
    """Test that JIT compilation provides expected performance."""
    results = benchmark_operator(operator_class)
    
    print(f"\n{operator_class.__name__} Performance:")
    print(f"  Regular: {results['regular_per_call']*1000:.3f}ms per call")
    print(f"  JIT:     {results['jit_per_call']*1000:.3f}ms per call")
    print(f"  Speedup: {results['speedup']:.2f}x")
    
    # Assert minimum expected speedup
    assert results['speedup'] >= expected_min_speedup, (
        f"JIT should be at least {expected_min_speedup}x as fast, "
        f"but got {results['speedup']:.2f}x"
    )


def test_jit_strategy_performance():
    """Test performance of different JIT strategies."""
    print("\nStrategy Performance Comparison for EnsembleOperator:")
    
    strategies = ["auto", "structural", "enhanced"]
    results = {}
    
    for strategy in strategies:
        result = benchmark_operator(EnsembleOperator, jit_mode=strategy)
        results[strategy] = result
        print(f"  {strategy:10s}: {result['jit_per_call']*1000:.3f}ms per call "
              f"(speedup: {result['speedup']:.2f}x)")
    
    # Enhanced should be fastest for ensemble
    enhanced_time = results["enhanced"]["jit_per_call"]
    trace_time = results["structural"]["jit_per_call"]
    
    # We expect enhanced to be at least as fast as trace for ensemble operators
    # (allowing 10% margin for measurement noise)
    assert enhanced_time <= trace_time * 1.1, (
        f"Enhanced strategy should be faster than trace for ensemble operators, "
        f"but enhanced={enhanced_time:.6f}s, trace={trace_time:.6f}s"
    )


def test_jit_compilation_overhead():
    """Test that JIT compilation overhead is reasonable."""
    # Create a fresh operator (no compilation cached)
    @jit
    class TestOperator(Operator):
        def forward(self, *, inputs):
            return {"result": inputs["value"] * 2}
    
    # Time first call (includes compilation)
    start = time.perf_counter()
    op = TestOperator()
    first_result = op(inputs={"value": 42})
    first_call_time = time.perf_counter() - start
    
    # Time second call (should use cached compilation)
    start = time.perf_counter()
    second_result = op(inputs={"value": 42})
    second_call_time = time.perf_counter() - start
    
    print(f"\nJIT Compilation Overhead:")
    print(f"  First call (with compilation): {first_call_time*1000:.3f}ms")
    print(f"  Second call (cached):          {second_call_time*1000:.3f}ms")
    print(f"  Overhead ratio:                {first_call_time/second_call_time:.1f}x")
    
    # Second call should be much faster (at least 2x)
    assert second_call_time < first_call_time / 2, (
        "Second call should be much faster than first call with compilation"
    )
    
    # Results should be identical
    assert first_result == second_result


if __name__ == "__main__":
    # Run basic benchmarks
    print("Running JIT Performance Benchmarks...")
    
    for op_class in [SimpleOperator, ComplexOperator, EnsembleOperator]:
        results = benchmark_operator(op_class)
        print(f"\n{op_class.__name__}:")
        print(f"  Regular: {results['regular_duration']:.3f}s total")
        print(f"  JIT:     {results['jit_duration']:.3f}s total")
        print(f"  Speedup: {results['speedup']:.2f}x")
    
    # Run strategy comparison
    test_jit_strategy_performance()
    
    # Test compilation overhead
    test_jit_compilation_overhead()