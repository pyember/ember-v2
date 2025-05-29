"""Realistic JIT performance tests that model actual Ember use cases.

Following Jeff Dean's principle: "Optimize for the common case."
The common case in Ember is I/O-bound LLM operations, not CPU-bound math.
"""

import time
import asyncio
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import pytest

from ember.core.registry.operator.base.operator_base import Operator
from ember.xcs import jit


def simulate_llm_call(model_name: str, prompt: str, delay: float = 0.1) -> Dict[str, Any]:
    """Simulate an LLM API call with realistic latency."""
    # This models real API latency (100ms is optimistic for most LLMs)
    time.sleep(delay)
    return {
        "model": model_name,
        "response": f"Response from {model_name} to: {prompt[:50]}...",
        "tokens": len(prompt.split()),
        "latency_ms": delay * 1000
    }


class RealisticLLMEnsemble(Operator):
    """Models a real ensemble pattern - multiple LLMs voting on an answer."""
    
    def __init__(self):
        super().__init__()
        self.models = ["gpt-4", "claude-3", "gemini-pro"]
        # Bypass specification for testing
        self.specification = None
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs["prompt"]
        
        # Call each model (these could run in parallel)
        responses = []
        for model in self.models:
            response = simulate_llm_call(model, prompt, delay=0.1)
            responses.append(response)
        
        # Aggregate responses (voting logic)
        return {
            "responses": responses,
            "consensus": self._find_consensus(responses),
            "total_latency_ms": sum(r["latency_ms"] for r in responses)
        }
    
    def _find_consensus(self, responses: List[Dict]) -> str:
        # Simulate consensus logic
        return f"Consensus from {len(responses)} models"


class SequentialLLMChain(Operator):
    """Models a sequential chain - output of one feeds into the next."""
    
    def __init__(self):
        super().__init__()
        self.specification = None
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs["prompt"]
        
        # Step 1: Initial processing
        step1 = simulate_llm_call("gpt-4", f"Analyze: {prompt}", delay=0.1)
        
        # Step 2: Refinement (depends on step 1)
        step2 = simulate_llm_call("claude-3", f"Refine: {step1['response']}", delay=0.1)
        
        # Step 3: Final polish (depends on step 2)
        step3 = simulate_llm_call("gemini-pro", f"Polish: {step2['response']}", delay=0.1)
        
        return {
            "final_response": step3["response"],
            "chain": [step1, step2, step3],
            "total_latency_ms": step1["latency_ms"] + step2["latency_ms"] + step3["latency_ms"]
        }


class MapReduceLLMPattern(Operator):
    """Models map-reduce pattern - parallel processing then aggregation."""
    
    def __init__(self, num_chunks: int = 5):
        super().__init__()
        self.num_chunks = num_chunks
        self.specification = None
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs["text"]
        
        # Split text into chunks
        words = text.split()
        chunk_size = max(1, len(words) // self.num_chunks)
        chunks = [
            " ".join(words[i:i+chunk_size]) 
            for i in range(0, len(words), chunk_size)
        ]
        
        # Process each chunk (these are independent)
        chunk_results = []
        for i, chunk in enumerate(chunks):
            result = simulate_llm_call(
                "gpt-4", 
                f"Summarize chunk {i}: {chunk[:50]}...",
                delay=0.05  # Smaller chunks = faster processing
            )
            chunk_results.append(result)
        
        # Reduce - combine summaries
        combined = " ".join(r["response"] for r in chunk_results)
        final_summary = simulate_llm_call(
            "claude-3",
            f"Combine summaries: {combined[:100]}...",
            delay=0.1
        )
        
        return {
            "chunk_results": chunk_results,
            "final_summary": final_summary,
            "total_latency_ms": sum(r["latency_ms"] for r in chunk_results) + final_summary["latency_ms"]
        }


def manual_parallel_ensemble(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Manually parallelized ensemble for comparison."""
    prompt = inputs["prompt"]
    models = ["gpt-4", "claude-3", "gemini-pro"]
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(simulate_llm_call, model, prompt, 0.1)
            for model in models
        ]
        responses = [f.result() for f in futures]
    
    return {
        "responses": responses,
        "consensus": f"Consensus from {len(responses)} models",
        "total_latency_ms": sum(r["latency_ms"] for r in responses)
    }


class TestRealisticJITPerformance:
    """Test JIT with realistic I/O-bound workloads."""
    
    def test_ensemble_pattern_speedup(self):
        """Test that ensemble pattern benefits from parallelization."""
        test_input = {"prompt": "What is the meaning of life, the universe, and everything?"}
        
        # Baseline - sequential execution
        ensemble = RealisticLLMEnsemble()
        
        start = time.perf_counter()
        sequential_result = ensemble(inputs=test_input)
        sequential_time = time.perf_counter() - start
        
        # Manual parallel execution
        start = time.perf_counter()
        parallel_result = manual_parallel_ensemble(test_input)
        parallel_time = time.perf_counter() - start
        
        # Calculate potential speedup
        potential_speedup = sequential_time / parallel_time
        
        print(f"\nEnsemble Pattern Performance:")
        print(f"  Sequential: {sequential_time*1000:.1f}ms")
        print(f"  Parallel:   {parallel_time*1000:.1f}ms")
        print(f"  Potential speedup: {potential_speedup:.2f}x")
        
        # With 3 models and 100ms each:
        # Sequential should take ~300ms
        # Parallel should take ~100ms (plus overhead)
        assert sequential_time > 0.25  # At least 250ms
        assert parallel_time < 0.15    # Less than 150ms
        assert potential_speedup > 1.8  # At least 1.8x speedup
    
    def test_sequential_chain_no_speedup(self):
        """Test that sequential chains don't benefit from parallelization."""
        test_input = {"prompt": "Explain quantum computing"}
        
        chain = SequentialLLMChain()
        
        # Run multiple times to get average
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = chain(inputs=test_input)
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        
        print(f"\nSequential Chain Performance:")
        print(f"  Average time: {avg_time*1000:.1f}ms")
        print(f"  Expected: ~300ms (3 x 100ms sequential calls)")
        
        # Should take about 300ms (3 sequential 100ms calls)
        assert 0.25 < avg_time < 0.35
    
    def test_map_reduce_pattern(self):
        """Test map-reduce pattern with partial parallelization."""
        test_input = {
            "text": " ".join(["word" + str(i) for i in range(100)])
        }
        
        map_reduce = MapReduceLLMPattern(num_chunks=5)
        
        start = time.perf_counter()
        result = map_reduce(inputs=test_input)
        execution_time = time.perf_counter() - start
        
        print(f"\nMap-Reduce Pattern Performance:")
        print(f"  Execution time: {execution_time*1000:.1f}ms")
        print(f"  Reported latency: {result['total_latency_ms']}ms")
        
        # With 5 chunks at 50ms each + 100ms reduce:
        # Sequential would be 350ms
        # Parallel map would be 50ms + 100ms = 150ms
        # Actual will be somewhere between due to Python overhead
    
    @pytest.mark.slow
    def test_jit_with_realistic_workload(self):
        """Test that JIT can recognize and optimize I/O-bound patterns."""
        test_input = {"prompt": "Test prompt for JIT optimization"}
        
        # Test with ensemble (should benefit from JIT)
        ensemble = RealisticLLMEnsemble()
        jit_ensemble = jit(RealisticLLMEnsemble, mode="structural")()
        
        # Warmup
        jit_ensemble(inputs=test_input)
        
        # Compare execution times
        regular_times = []
        jit_times = []
        
        for _ in range(10):
            # Regular execution
            start = time.perf_counter()
            ensemble(inputs=test_input)
            regular_times.append(time.perf_counter() - start)
            
            # JIT execution
            start = time.perf_counter()
            jit_ensemble(inputs=test_input)
            jit_times.append(time.perf_counter() - start)
        
        regular_avg = sum(regular_times) / len(regular_times)
        jit_avg = sum(jit_times) / len(jit_times)
        
        print(f"\nJIT Performance on Realistic Workload:")
        print(f"  Regular: {regular_avg*1000:.1f}ms")
        print(f"  JIT:     {jit_avg*1000:.1f}ms")
        print(f"  Speedup: {regular_avg/jit_avg:.2f}x")
        
        # Note: Current JIT doesn't actually parallelize
        # This test documents current behavior
        # Future JIT should show significant speedup


class TestCPUBoundComparison:
    """Compare CPU-bound vs I/O-bound performance characteristics."""
    
    def test_cpu_bound_no_benefit(self):
        """Show that CPU-bound operations don't benefit from threading."""
        
        def cpu_bound_task(n: int) -> int:
            """Pure CPU computation."""
            total = 0
            for i in range(n):
                total += i * i
            return total
        
        # Sequential execution
        start = time.perf_counter()
        sequential_results = [cpu_bound_task(1000000) for _ in range(4)]
        sequential_time = time.perf_counter() - start
        
        # Threaded execution
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_bound_task, 1000000) for _ in range(4)]
            threaded_results = [f.result() for f in futures]
        threaded_time = time.perf_counter() - start
        
        print(f"\nCPU-Bound Performance:")
        print(f"  Sequential: {sequential_time*1000:.1f}ms")
        print(f"  Threaded:   {threaded_time*1000:.1f}ms")
        print(f"  Speedup:    {sequential_time/threaded_time:.2f}x")
        
        # Due to GIL, threaded might even be slower
        assert threaded_time >= sequential_time * 0.8  # No significant speedup
    
    def test_io_bound_clear_benefit(self):
        """Show that I/O-bound operations benefit from threading."""
        
        def io_bound_task(delay: float) -> str:
            """I/O simulation with sleep."""
            time.sleep(delay)
            return f"Completed after {delay}s"
        
        # Sequential execution
        start = time.perf_counter()
        sequential_results = [io_bound_task(0.1) for _ in range(4)]
        sequential_time = time.perf_counter() - start
        
        # Threaded execution
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(io_bound_task, 0.1) for _ in range(4)]
            threaded_results = [f.result() for f in futures]
        threaded_time = time.perf_counter() - start
        
        print(f"\nI/O-Bound Performance:")
        print(f"  Sequential: {sequential_time*1000:.1f}ms")
        print(f"  Threaded:   {threaded_time*1000:.1f}ms")
        print(f"  Speedup:    {sequential_time/threaded_time:.2f}x")
        
        # Should see ~4x speedup
        assert sequential_time > 0.35  # At least 350ms
        assert threaded_time < 0.15    # Less than 150ms
        assert sequential_time / threaded_time > 2.0  # At least 2x speedup


if __name__ == "__main__":
    # Run key tests
    test = TestRealisticJITPerformance()
    test.test_ensemble_pattern_speedup()
    test.test_sequential_chain_no_speedup()
    test.test_map_reduce_pattern()
    
    comparison = TestCPUBoundComparison()
    comparison.test_cpu_bound_no_benefit()
    comparison.test_io_bound_clear_benefit()