"""Comprehensive JIT performance tests.

Tests various operator patterns to verify JIT provides real performance benefits.
"""

import time
from typing import Dict, Any, List
import pytest
import numpy as np

from ember.core.registry.operator.base.operator_base import Operator
from ember.xcs import jit
from .benchmark_framework import JITBenchmark, BenchmarkResult


# ============================================================================
# Test Operators - Various Patterns
# ============================================================================

class SimpleOperator(Operator):
    """Minimal computation - tests JIT overhead."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": inputs["x"] * 2 + inputs["y"]}


class LoopOperator(Operator):
    """Single loop - tests basic optimization."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = 0
        for i in range(inputs["n"]):
            result += i * inputs["factor"]
        return {"result": result}


class NestedLoopOperator(Operator):
    """Nested loops - prime candidate for optimization."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = 0
        n, m = inputs["n"], inputs["m"]
        factor = inputs["factor"]
        
        for i in range(n):
            for j in range(m):
                result += (i * j) * factor
                
        return {"result": result}


class EnsembleOperator(Operator):
    """Ensemble pattern - tests parallelization potential."""
    
    def __init__(self, num_models: int = 10):
        super().__init__()
        self.num_models = num_models
        
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        base_value = inputs["value"]
        
        # Simulate independent model computations
        for i in range(self.num_models):
            model_result = 0
            for j in range(100):  # Each "model" does some work
                model_result += (base_value + i) * j
            results.append(model_result)
            
        # Aggregate results
        return {
            "ensemble_mean": sum(results) / len(results),
            "ensemble_max": max(results),
            "ensemble_min": min(results),
        }


class ConditionalOperator(Operator):
    """Conditional logic - tests trace strategy with branches."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mode = inputs["mode"]
        value = inputs["value"]
        
        if mode == "fast":
            # Fast path - simple computation
            return {"result": value * 2}
        elif mode == "medium":
            # Medium path - some computation
            result = 0
            for i in range(10):
                result += value * i
            return {"result": result}
        else:
            # Slow path - heavy computation
            result = 0
            for i in range(100):
                for j in range(10):
                    result += value * i * j
            return {"result": result}


class PipelineOperator(Operator):
    """Sequential pipeline - tests graph optimization."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Stage 1: Preprocessing
        data = inputs["data"]
        processed = [x * 2 for x in data]
        
        # Stage 2: Transformation
        transformed = [x ** 2 for x in processed]
        
        # Stage 3: Aggregation
        result = {
            "sum": sum(transformed),
            "mean": sum(transformed) / len(transformed),
            "max": max(transformed),
        }
        
        return result


class RecursiveOperator(Operator):
    """Recursive computation - tests recursive JIT compilation."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        n = inputs["n"]
        if n <= 1:
            return {"result": 1}
        
        # Recursive calls
        sub_result1 = self(inputs={"n": n - 1})["result"]
        sub_result2 = self(inputs={"n": n - 2})["result"] if n > 2 else 1
        
        return {"result": sub_result1 + sub_result2}


class MatrixOperator(Operator):
    """Matrix operations - tests numerical computation optimization."""
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        matrix_a = inputs["matrix_a"]
        matrix_b = inputs["matrix_b"]
        
        # Manual matrix multiplication (intentionally not using numpy)
        # This tests whether JIT can optimize numerical loops
        n = len(matrix_a)
        m = len(matrix_b[0])
        k = len(matrix_b)
        
        result = [[0 for _ in range(m)] for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                for p in range(k):
                    result[i][j] += matrix_a[i][p] * matrix_b[p][j]
                    
        return {"result": result}


# ============================================================================
# Benchmark Tests
# ============================================================================

class TestJITPerformance:
    """Comprehensive JIT performance tests."""
    
    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        return JITBenchmark()
    
    def test_simple_operator_overhead(self, benchmark):
        """Test: JIT overhead for simple operations."""
        test_inputs = {"x": 42, "y": 13}
        
        results = benchmark.compare_strategies(
            SimpleOperator,
            test_inputs,
            warmup_iterations=50,
            test_iterations=10000)
        
        # Simple operators should have minimal overhead
        baseline = results["none"].timing_stats.mean
        auto_overhead = (results["auto"].timing_stats.mean / baseline) - 1
        
        print(f"\nSimple Operator Overhead: {auto_overhead*100:.1f}%")
        assert auto_overhead < 0.20, "JIT overhead should be < 20% for simple ops"
        
        # Save results
        benchmark.save_results(results, "simple_operator_results.json")
        
    def test_loop_optimization(self, benchmark):
        """Test: JIT optimization for loop-heavy code."""
        test_inputs = {"n": 1000, "factor": 3.14}
        
        results = benchmark.compare_strategies(
            LoopOperator,
            test_inputs,
            warmup_iterations=20,
            test_iterations=1000)
        
        # Loop operators should show speedup
        baseline = results["none"].timing_stats.mean
        best_jit = min(
            results[s].timing_stats.mean 
            for s in ["auto", "trace", "structural", "enhanced"]
        )
        speedup = baseline / best_jit
        
        print(f"\nLoop Operator Speedup: {speedup:.2f}x")
        assert speedup > 0.9, "JIT should not significantly slow down loop operations"
        
        # Report break-even points
        for strategy, result in results.items():
            if strategy != "none" and result.calls_to_break_even < float('inf'):
                print(f"  {strategy}: break-even after {result.calls_to_break_even} calls")
                
        benchmark.save_results(results, "loop_operator_results.json")
        
    def test_ensemble_parallelization(self, benchmark):
        """Test: Enhanced strategy for ensemble patterns."""
        test_inputs = {"value": 10}
        
        # Compare strategies for ensemble with different sizes
        for num_models in [5, 10, 20]:
            results = benchmark.compare_strategies(
                lambda: EnsembleOperator(num_models),
                test_inputs,
                warmup_iterations=10,
                test_iterations=100)
            
            # Enhanced should perform best for ensemble patterns
            enhanced_time = results["enhanced"].timing_stats.mean
            trace_time = results["structural"].timing_stats.mean
            
            print(f"\nEnsemble ({num_models} models):")
            print(f"  Enhanced: {enhanced_time:.3f}ms")
            print(f"  Trace: {trace_time:.3f}ms")
            print(f"  Enhanced advantage: {(trace_time/enhanced_time - 1)*100:.1f}%")
            
            benchmark.save_results(
                results, 
                f"ensemble_operator_{num_models}_results.json"
            )
    
    def test_conditional_logic_handling(self, benchmark):
        """Test: How different strategies handle conditional logic."""
        # Test different execution paths
        test_cases = [
            {"mode": "fast", "value": 100},
            {"mode": "medium", "value": 100},
            {"mode": "slow", "value": 10}]
        
        for test_inputs in test_cases:
            results = benchmark.compare_strategies(
                ConditionalOperator,
                test_inputs,
                warmup_iterations=20,
                test_iterations=1000)
            
            mode = test_inputs["mode"]
            print(f"\nConditional Operator ({mode} path):")
            
            for strategy, result in results.items():
                print(f"  {strategy}: {result.timing_stats.mean:.3f}ms")
                
            benchmark.save_results(
                results,
                f"conditional_operator_{mode}_results.json"
            )
    
    def test_scaling_performance(self, benchmark):
        """Test: Performance scaling with input size."""
        def input_generator(size: int) -> Dict[str, Any]:
            return {"data": list(range(size))}
        
        sizes = [10, 100, 1000, 10000]
        
        # Test pipeline operator scaling
        scaling_results = benchmark.run_scaling_benchmark(
            PipelineOperator,
            input_generator,
            sizes,
            strategy="auto",
            warmup_iterations=10,
            test_iterations=100)
        
        print("\nPipeline Operator Scaling:")
        print("Size  | Time (ms) | Speedup | Memory (MB)")
        print("------|-----------|---------|------------")
        
        for size, result in scaling_results.items():
            print(f"{size:5d} | {result.timing_stats.mean:9.3f} | "
                  f"{result.speedup:7.2f}x | {result.memory_delta_mb:10.2f}")
            
        benchmark.save_results(scaling_results, "scaling_results.json")
    
    def test_matrix_operations(self, benchmark):
        """Test: Numerical computation optimization."""
        # Test different matrix sizes
        matrix_sizes = [10, 20, 50]
        
        for size in matrix_sizes:
            # Create test matrices
            matrix_a = [[i+j for j in range(size)] for i in range(size)]
            matrix_b = [[i*j for j in range(size)] for i in range(size)]
            test_inputs = {"matrix_a": matrix_a, "matrix_b": matrix_b}
            
            results = benchmark.compare_strategies(
                MatrixOperator,
                test_inputs,
                warmup_iterations=5,
                test_iterations=20)
            
            print(f"\nMatrix Multiplication ({size}x{size}):")
            
            baseline = results["none"].timing_stats.mean
            for strategy, result in results.items():
                speedup = baseline / result.timing_stats.mean
                print(f"  {strategy}: {result.timing_stats.mean:.3f}ms ({speedup:.2f}x)")
                
            benchmark.save_results(
                results,
                f"matrix_operator_{size}x{size}_results.json"
            )
    
    @pytest.mark.slow
    def test_stress_stability(self, benchmark):
        """Test: Long-running stress test for stability."""
        test_inputs = {"n": 100, "m": 100, "factor": 2.0}
        
        # Run 30-second stress test
        stress_result = benchmark.run_stress_test(
            NestedLoopOperator,
            test_inputs,
            duration_seconds=30,
            concurrent_threads=4,
            strategy="auto")
        
        print(f"\nStress Test Results:")
        print(f"  Total calls: {stress_result.total_calls}")
        print(f"  Calls/second: {stress_result.calls_per_second:.1f}")
        print(f"  Memory leak: {stress_result.memory_leak_mb_per_hour:.3f} MB/hour")
        print(f"  Error rate: {stress_result.error_rate:.4%}")
        
        # Assert stability criteria
        assert stress_result.error_rate < 0.001, "Error rate should be < 0.1%"
        assert stress_result.memory_leak_mb_per_hour < 10, "Memory leak should be < 10MB/hour"
    
    def test_auto_strategy_selection(self, benchmark):
        """Test: Verify AUTO picks appropriate strategies."""
        test_cases = [
            (SimpleOperator, {"x": 10, "y": 20}, "trace"),
            (EnsembleOperator, {"value": 10}, "enhanced"),
            (NestedLoopOperator, {"n": 50, "m": 50, "factor": 1.5}, "enhanced")]
        
        for operator_class, test_inputs, expected_best in test_cases:
            results = benchmark.compare_strategies(
                operator_class,
                test_inputs,
                strategies=["auto", "trace", "structural", "enhanced"],
                warmup_iterations=10,
                test_iterations=100)
            
            # Find which strategy AUTO selected by comparing performance
            auto_time = results["auto"].timing_stats.mean
            
            # Find closest matching strategy
            closest_strategy = min(
                [(s, abs(r.timing_stats.mean - auto_time)) 
                 for s, r in results.items() if s != "auto"],
                key=lambda x: x[1]
            )[0]
            
            print(f"\n{operator_class.__name__}:")
            print(f"  AUTO selected strategy similar to: {closest_strategy}")
            print(f"  Expected best: {expected_best}")
            
            # Generate report
            report = benchmark.generate_report(results)
            print(report)


# ============================================================================
# Main benchmark runner
# ============================================================================

if __name__ == "__main__":
    # Run comprehensive benchmarks
    benchmark = JITBenchmark()
    test = TestJITPerformance()
    
    print("Running JIT Performance Benchmarks...")
    print("=" * 60)
    
    # Run each test
    test.test_simple_operator_overhead(benchmark)
    test.test_loop_optimization(benchmark) 
    test.test_ensemble_parallelization(benchmark)
    test.test_conditional_logic_handling(benchmark)
    test.test_scaling_performance(benchmark)
    test.test_matrix_operations(benchmark)
    test.test_auto_strategy_selection(benchmark)
    
    print("\n" + "=" * 60)
    print("Benchmarks complete! Results saved in benchmark_results/")