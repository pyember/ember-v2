"""Test that XCS parallelism actually works after our fixes.

Clean tests without the tracing noise.
"""

import time
import pytest
from ember.xcs import jit


def slow_work(delay: float = 0.01):
    """Simulate work that takes time."""
    time.sleep(delay)
    return "done"


class TestXCSParallelismFix:
    """Test parallelism after fixes."""
    
    def test_parallel_speedup(self):
        """Test that parallel operations actually speed up."""
        
        # Non-JIT baseline
        def sequential_four():
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            return "done"
        
        # JIT version
        @jit
        def parallel_four():
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            return "done"
        
        # Time sequential
        start = time.time()
        sequential_four()
        seq_time = time.time() - start
        
        # First call to JIT version (includes tracing)
        start = time.time()
        parallel_four()
        first_time = time.time() - start
        
        # Second call to JIT version (uses cached optimization)
        start = time.time()
        parallel_four()
        second_time = time.time() - start
        
        print(f"\nSequential: {seq_time:.3f}s")
        print(f"JIT first call: {first_time:.3f}s")
        print(f"JIT second call: {second_time:.3f}s")
        
        # Second call should be faster than sequential
        speedup = seq_time / second_time
        print(f"Speedup: {speedup:.1f}x")
        
        # Should see speedup on cached execution
        # Not expecting 4x because of overhead, but should be > 1.5x
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.1f}x"
    
    def test_parallel_with_dependencies(self):
        """Test mixed parallel/sequential execution."""
        
        @jit
        def diamond_pattern():
            # Phase 1: single operation
            a = slow_work(0.01)
            
            # Phase 2: three parallel operations  
            b = slow_work(0.01)
            c = slow_work(0.01)
            d = slow_work(0.01)
            
            # Phase 3: merge (depends on b, c, d)
            result = f"{b}{c}{d}"
            e = slow_work(0.01)
            
            return e
        
        # First execution (with tracing)
        start = time.time()
        diamond_pattern()
        first_time = time.time() - start
        
        # Second execution (optimized)
        start = time.time()
        diamond_pattern()
        second_time = time.time() - start
        
        print(f"\nDiamond pattern:")
        print(f"First call: {first_time:.3f}s")
        print(f"Second call: {second_time:.3f}s")
        
        # Should take ~0.03s (3 phases) not 0.05s (5 sequential)
        assert second_time < 0.04, f"Expected < 0.04s, got {second_time:.3f}s"
    
    def test_no_parallelism_fallback(self):
        """Test that code without parallelism falls back to original."""
        
        @jit
        def no_function_calls():
            # Simple operations without function calls
            a = 1
            b = a + 1
            c = b + 1
            d = c + 1
            return d
        
        # Execute once to trigger optimization decision
        result = no_function_calls()
        assert result == 4
        
        # Check stats - should not be optimized
        stats = no_function_calls.stats()
        print(f"\nNo parallelism stats: {stats}")
        assert stats['optimized'] == False
        assert stats['status'] == 'fallback'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])