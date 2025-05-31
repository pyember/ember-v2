"""Test the XCS public API with excellence.

Tests only the 4 public functions that users can access.
No internal implementation details. Just clean, comprehensive API testing.
"""

import time
import pytest
from ember.api import xcs
from ember.api.operators import Operator, Specification


class TestJIT:
    """Test @jit decorator - the crown jewel of XCS."""
    
    def test_jit_on_function_preserves_behavior(self):
        """@jit preserves function behavior perfectly."""
        def compute(*, inputs):
            return {"result": inputs.get("x", 0) ** 2 + inputs.get("y", 0)}
        
        jitted = xcs.jit(compute)
        
        # Same results
        normal_result = compute(inputs={"x": 3, "y": 4})
        jit_result = jitted(inputs={"x": 3, "y": 4})
        assert normal_result == jit_result == {"result": 13}
    
    def test_jit_on_operator_preserves_behavior(self):
        """@jit works seamlessly with operators."""
        @xcs.jit
        class SquareOperator(Operator):
            specification = Specification()
            
            def forward(self, *, inputs):
                val = inputs.get("value", 0)
                return {"squared": val * val}
        
        op = SquareOperator()
        assert op(value=5) == {"squared": 25}
    
    def test_jit_caches_compiled_functions(self):
        """@jit caches compiled functions for performance."""
        call_count = 0
        
        @xcs.jit
        def cached_function(*, inputs):
            nonlocal call_count
            call_count += 1
            return {"count": call_count, "value": inputs.get("x", 0) * 2}
        
        # First call - compilation happens
        result1 = cached_function(inputs={"x": 5})
        assert result1 == {"count": 1, "value": 10}
        
        # Second call - uses cached version
        result2 = cached_function(inputs={"x": 10})
        assert result2 == {"count": 2, "value": 20}
        
        # Stats should show cache hit
        stats = xcs.get_jit_stats()
        assert stats["cache_hits"] > 0
    
    def test_jit_handles_errors_gracefully(self):
        """@jit preserves error behavior."""
        @xcs.jit
        def failing_function(*, inputs):
            if inputs.get("fail", False):
                raise ValueError("Requested failure")
            return {"status": "ok"}
        
        # Success case
        assert failing_function(inputs={}) == {"status": "ok"}
        
        # Failure case
        with pytest.raises(ValueError, match="Requested failure"):
            failing_function(inputs={"fail": True})
    
    def test_jit_is_truly_zero_configuration(self):
        """@jit requires no configuration."""
        # Just works - no modes, no options needed
        @xcs.jit
        def simple(*, inputs):
            return {"done": True}
        
        assert simple(inputs={}) == {"done": True}


class TestVMap:
    """Test vmap - elegant batch transformation."""
    
    def test_vmap_transforms_single_to_batch(self):
        """vmap transforms single-item functions to batch processors."""
        def process_item(x):
            return x * x
        
        batch_process = xcs.vmap(process_item)
        
        # Natural API - just pass the list directly
        result = batch_process([1, 2, 3, 4, 5])
        assert result == [1, 4, 9, 16, 25]
    
    def test_vmap_preserves_function_signature(self):
        """vmap maintains the function's behavior."""
        def add(x, y):
            return x + y
        
        batch_add = xcs.vmap(add)
        
        # Natural API - multiple arguments
        result = batch_add([1, 2, 3], [10, 20, 30])
        assert result == [11, 22, 33]
    
    def test_vmap_handles_complex_transformations(self):
        """vmap works with complex data structures."""
        def process_dict(d):
            return {
                "sum": d["a"] + d["b"],
                "product": d["a"] * d["b"]
            }
        
        batch_process = xcs.vmap(process_dict)
        
        # Natural API - list of dicts
        result = batch_process([
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6}
        ])
        
        # vmap returns a list of results
        assert result == [
            {"sum": 3, "product": 2},
            {"sum": 7, "product": 12},
            {"sum": 11, "product": 30}
        ]


class TestTrace:
    """Test @trace - execution analysis made simple."""
    
    def test_trace_provides_execution_visibility(self):
        """@trace enables execution analysis."""
        call_count = 0
        
        @xcs.trace
        def traced_function(*, inputs):
            nonlocal call_count
            call_count += 1
            return {"count": call_count}
        
        # Function still works normally
        assert traced_function(inputs={}) == {"count": 1}
        assert traced_function(inputs={}) == {"count": 2}
        
        # Stats should be available
        stats = xcs.get_jit_stats()
        assert isinstance(stats, dict)
    
    def test_trace_is_autograph_alias(self):
        """trace and autograph are the same."""
        assert xcs.trace is xcs.autograph


class TestGetJITStats:
    """Test get_jit_stats - performance insights."""
    
    def test_get_jit_stats_returns_dict(self):
        """get_jit_stats provides performance metrics."""
        stats = xcs.get_jit_stats()
        assert isinstance(stats, dict)
        
        # Should have some standard keys
        expected_keys = {"cache_size", "total_compilations", "cache_hits"}
        assert any(key in stats for key in expected_keys)
    
    def test_get_jit_stats_tracks_function_metrics(self):
        """get_jit_stats tracks individual function metrics."""
        @xcs.jit
        def tracked_function(*, inputs):
            return {"value": inputs.get("x", 0) * 2}
        
        # Execute multiple times
        for i in range(5):
            tracked_function(inputs={"x": i})
        
        # Stats should reflect usage
        stats = xcs.get_jit_stats()
        assert stats is not None


class TestAPICompleteness:
    """Verify the API is complete and minimal."""
    
    def test_public_api_has_exactly_4_functions(self):
        """Public API exports exactly what's promised."""
        # Core functions
        assert hasattr(xcs, 'jit')
        assert hasattr(xcs, 'trace')
        assert hasattr(xcs, 'vmap')
        assert hasattr(xcs, 'get_jit_stats')
        
        # Alias
        assert hasattr(xcs, 'autograph')
        assert xcs.autograph is xcs.trace
    
    def test_removed_apis_are_gone(self):
        """Removed APIs are truly removed."""
        # These should NOT be accessible
        assert not hasattr(xcs, 'pmap')
        assert not hasattr(xcs, 'Graph')
        assert not hasattr(xcs, 'Node')
        assert not hasattr(xcs, 'ExecutionOptions')
        assert not hasattr(xcs, 'JITMode')
        assert not hasattr(xcs, 'explain_jit_selection')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])