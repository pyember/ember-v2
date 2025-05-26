"""Comprehensive JIT benchmarking framework.

Following Jeff Dean's principle: "Measure everything that matters."
"""

import gc
import time
import psutil
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Type, Any, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from ember.core.registry.operator.base.operator_base import Operator
from ember.xcs import jit


@dataclass
class TimingStats:
    """Detailed timing statistics."""
    samples: List[float] = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0
    
    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0
    
    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0
    
    @property
    def p95(self) -> float:
        return np.percentile(self.samples, 95) if self.samples else 0.0
    
    @property
    def p99(self) -> float:
        return np.percentile(self.samples, 99) if self.samples else 0.0
    
    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0.0
    
    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark results."""
    name: str
    strategy: str
    operator_type: str
    
    # Timing metrics (in milliseconds)
    first_call_ms: float
    compilation_ms: float
    timing_stats: TimingStats
    
    # Performance metrics
    speedup: float  # vs baseline
    calls_to_break_even: int  # when compilation cost is recovered
    
    # Resource metrics
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    cache_size_kb: float
    
    # Metadata
    input_size: int
    num_iterations: int
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "strategy": self.strategy,
            "operator_type": self.operator_type,
            "first_call_ms": self.first_call_ms,
            "compilation_ms": self.compilation_ms,
            "mean_call_ms": self.timing_stats.mean,
            "median_call_ms": self.timing_stats.median,
            "p95_ms": self.timing_stats.p95,
            "p99_ms": self.timing_stats.p99,
            "speedup": self.speedup,
            "calls_to_break_even": self.calls_to_break_even,
            "memory_delta_mb": self.memory_delta_mb,
            "cache_size_kb": self.cache_size_kb,
            "input_size": self.input_size,
            "num_iterations": self.num_iterations,
            "errors": self.errors,
        }


@dataclass 
class StressTestResult:
    """Results from stress testing."""
    operator_name: str
    duration_seconds: float
    total_calls: int
    calls_per_second: float
    memory_leak_mb_per_hour: float
    error_rate: float
    thread_results: Dict[int, TimingStats]


class MemoryTracker:
    """Track memory usage changes."""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def measure_delta(self, func: Callable) -> Tuple[Any, float]:
        """Measure memory change from running a function."""
        gc.collect()  # Clean baseline
        before = self.get_memory_mb()
        
        result = func()
        
        gc.collect()  # Clean after
        after = self.get_memory_mb()
        
        return result, after - before


class JITBenchmark:
    """Comprehensive JIT benchmarking framework."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.memory_tracker = MemoryTracker()
        
    def benchmark_operator(
        self,
        operator_class: Type[Operator],
        test_inputs: Dict[str, Any],
        strategy: str = "auto",
        warmup_iterations: int = 10,
        test_iterations: int = 1000,
        name: Optional[str] = None) -> BenchmarkResult:
        """Benchmark a single operator with a specific strategy."""
        name = name or f"{operator_class.__name__}_{strategy}"
        
        # Create baseline (non-JIT) operator
        baseline_op = operator_class()
        
        # Create JIT operator with specific strategy
        if strategy == "none":
            jit_op = baseline_op  # No JIT for baseline comparison
        else:
            jit_decorator = jit(mode=strategy) if strategy != "auto" else jit
            JITOperator = jit_decorator(operator_class)
            jit_op = JITOperator()
        
        # Measure compilation time (first call)
        gc.collect()
        memory_before = self.memory_tracker.get_memory_mb()
        
        compilation_start = time.perf_counter()
        first_result = jit_op(inputs=test_inputs)
        compilation_end = time.perf_counter()
        
        first_call_ms = (compilation_end - compilation_start) * 1000
        
        # Warmup runs
        for _ in range(warmup_iterations):
            jit_op(inputs=test_inputs)
            
        # Measure memory after compilation and warmup
        gc.collect()
        memory_after = self.memory_tracker.get_memory_mb()
        
        # Timing runs
        timing_stats = TimingStats()
        
        for _ in range(test_iterations):
            start = time.perf_counter()
            result = jit_op(inputs=test_inputs)
            end = time.perf_counter()
            timing_stats.samples.append((end - start) * 1000)
        
        # Baseline timing for comparison
        baseline_stats = TimingStats()
        for _ in range(min(100, test_iterations)):  # Fewer iterations for baseline
            start = time.perf_counter()
            baseline_result = baseline_op(inputs=test_inputs)
            end = time.perf_counter()
            baseline_stats.samples.append((end - start) * 1000)
        
        # Calculate metrics
        speedup = baseline_stats.mean / timing_stats.mean if timing_stats.mean > 0 else 0.0
        
        # Estimate compilation cost (first call - average call)
        compilation_ms = max(0, first_call_ms - timing_stats.mean)
        
        # Calculate break-even point
        if speedup > 1.0 and baseline_stats.mean > timing_stats.mean:
            time_saved_per_call = baseline_stats.mean - timing_stats.mean
            calls_to_break_even = int(compilation_ms / time_saved_per_call) if time_saved_per_call > 0 else float('inf')
        else:
            calls_to_break_even = float('inf')
        
        # Estimate cache size (rough approximation)
        cache_size_kb = max(0, (memory_after - memory_before) * 1024)
        
        return BenchmarkResult(
            name=name,
            strategy=strategy,
            operator_type=operator_class.__name__,
            first_call_ms=first_call_ms,
            compilation_ms=compilation_ms,
            timing_stats=timing_stats,
            speedup=speedup,
            calls_to_break_even=calls_to_break_even,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_after - memory_before,
            cache_size_kb=cache_size_kb,
            input_size=len(str(test_inputs)),
            num_iterations=test_iterations)
    
    def compare_strategies(
        self,
        operator_class: Type[Operator],
        test_inputs: Dict[str, Any],
        strategies: List[str] = ["none", "auto", "trace", "structural", "enhanced"],
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """Compare performance across different JIT strategies."""
        results = {}
        
        for strategy in strategies:
            try:
                result = self.benchmark_operator(
                    operator_class,
                    test_inputs,
                    strategy=strategy,
                    **kwargs
                )
                results[strategy] = result
            except Exception as e:
                print(f"Error benchmarking {operator_class.__name__} with {strategy}: {e}")
                
        return results
    
    def run_scaling_benchmark(
        self,
        operator_class: Type[Operator],
        input_generator: Callable[[int], Dict[str, Any]],
        sizes: List[int],
        strategy: str = "auto",
        **kwargs
    ) -> Dict[int, BenchmarkResult]:
        """Test performance scaling with input size."""
        results = {}
        
        for size in sizes:
            test_inputs = input_generator(size)
            result = self.benchmark_operator(
                operator_class,
                test_inputs,
                strategy=strategy,
                name=f"{operator_class.__name__}_{strategy}_size{size}",
                **kwargs
            )
            results[size] = result
            
        return results
    
    def run_stress_test(
        self,
        operator_class: Type[Operator],
        test_inputs: Dict[str, Any],
        duration_seconds: int = 60,
        concurrent_threads: int = 4,
        strategy: str = "auto") -> StressTestResult:
        """Run stress test to check for stability and memory leaks."""
        # Create JIT operator
        jit_decorator = jit(mode=strategy) if strategy != "auto" else jit
        JITOperator = jit_decorator(operator_class)
        
        # Shared operator instance for thread safety testing
        shared_op = JITOperator()
        
        # Warmup
        for _ in range(10):
            shared_op(inputs=test_inputs)
        
        # Track results per thread
        thread_results = {i: TimingStats() for i in range(concurrent_threads)}
        error_count = 0
        call_count = 0
        
        # Memory tracking
        memory_start = self.memory_tracker.get_memory_mb()
        start_time = time.time()
        
        def worker(thread_id: int):
            nonlocal error_count, call_count
            local_calls = 0
            
            while time.time() - start_time < duration_seconds:
                try:
                    call_start = time.perf_counter()
                    result = shared_op(inputs=test_inputs)
                    call_end = time.perf_counter()
                    
                    thread_results[thread_id].samples.append(
                        (call_end - call_start) * 1000
                    )
                    local_calls += 1
                    
                except Exception as e:
                    error_count += 1
                    
            call_count += local_calls
        
        # Run concurrent stress test
        with ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            futures = [
                executor.submit(worker, i)
                for i in range(concurrent_threads)
            ]
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Calculate results
        actual_duration = time.time() - start_time
        memory_end = self.memory_tracker.get_memory_mb()
        memory_leak_mb = memory_end - memory_start
        memory_leak_rate = (memory_leak_mb / actual_duration) * 3600  # MB per hour
        
        return StressTestResult(
            operator_name=operator_class.__name__,
            duration_seconds=actual_duration,
            total_calls=call_count,
            calls_per_second=call_count / actual_duration,
            memory_leak_mb_per_hour=memory_leak_rate,
            error_rate=error_count / call_count if call_count > 0 else 0,
            thread_results=thread_results)
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable = {}
        for key, value in results.items():
            if hasattr(value, 'to_dict'):
                serializable[key] = value.to_dict()
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                serializable[key] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value
                
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
            
        print(f"Results saved to {output_path}")
    
    def generate_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate a human-readable performance report."""
        report = ["# JIT Performance Benchmark Report\n"]
        
        # Summary table
        report.append("## Summary\n")
        report.append("| Strategy | Mean (ms) | P95 (ms) | Speedup | Break-even | Memory (MB) |")
        report.append("|----------|-----------|----------|---------|------------|-------------|")
        
        for strategy, result in results.items():
            report.append(
                f"| {strategy:8s} | "
                f"{result.timing_stats.mean:9.3f} | "
                f"{result.timing_stats.p95:8.3f} | "
                f"{result.speedup:7.2f}x | "
                f"{result.calls_to_break_even:10.0f} | "
                f"{result.memory_delta_mb:11.2f} |"
            )
        
        # Best strategy
        best_strategy = min(results.items(), key=lambda x: x[1].timing_stats.mean)
        report.append(f"\n**Best Strategy**: {best_strategy[0]} "
                     f"({best_strategy[1].timing_stats.mean:.3f}ms mean)\n")
        
        return "\n".join(report)