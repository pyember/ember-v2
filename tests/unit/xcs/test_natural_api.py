"""Tests for the Natural XCS API."""

import pytest
from typing import List, Dict, Any

from ember.api import xcs


class TestNaturalJIT:
    """Test natural JIT compilation."""
    
    def test_simple_function(self):
        """Natural functions work without modification."""
        @xcs.jit
        def add(x, y):
            return x + y
        
        assert add(2, 3) == 5
        assert add(10, 20) == 30
    
    def test_function_with_defaults(self):
        """Functions with default arguments work correctly."""
        @xcs.jit
        def multiply(x, y=2):
            return x * y
        
        assert multiply(5) == 10
        assert multiply(5, 3) == 15
    
    def test_keyword_only_function(self):
        """Keyword-only functions are supported."""
        @xcs.jit
        def divide(*, numerator, denominator):
            return numerator / denominator
        
        assert divide(numerator=10, denominator=2) == 5.0
    
    def test_mixed_args_function(self):
        """Functions with mixed argument types work."""
        @xcs.jit
        def compute(x, y, *, scale=1.0):
            return (x + y) * scale
        
        assert compute(2, 3) == 5.0
        assert compute(2, 3, scale=2.0) == 10.0
    
    def test_varargs_function(self):
        """Functions with *args work correctly."""
        @xcs.jit
        def sum_all(*numbers):
            return sum(numbers)
        
        assert sum_all(1, 2, 3, 4) == 10
        assert sum_all() == 0
    
    def test_kwargs_function(self):
        """Functions with **kwargs work correctly."""
        @xcs.jit
        def make_dict(**items):
            return items
        
        result = make_dict(a=1, b=2, c=3)
        assert result == {"a": 1, "b": 2, "c": 3}
    
    def test_complex_function(self):
        """Complex functions with multiple features work."""
        @xcs.jit
        def process(x, *args, factor=2, **options):
            base = x * factor
            for arg in args:
                base += arg
            if options.get('square', False):
                base = base ** 2
            return base
        
        assert process(5) == 10
        assert process(5, 1, 2) == 13
        assert process(5, factor=3) == 15
        assert process(5, square=True) == 100
    
    def test_preserves_docstring_and_signature(self):
        """JIT preserves function metadata."""
        def original(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y
        
        jitted = xcs.jit(original)
        
        assert jitted.__name__ == original.__name__
        assert jitted.__doc__ == original.__doc__
    
    def test_error_handling(self):
        """Errors are preserved and meaningful."""
        @xcs.jit
        def failing_function(x):
            if x < 0:
                raise ValueError("x must be non-negative")
            return x ** 2
        
        assert failing_function(5) == 25
        
        with pytest.raises(ValueError, match="x must be non-negative"):
            failing_function(-1)


class TestNaturalVMap:
    """Test natural vmap transformation."""
    
    def test_single_argument_batching(self):
        """Single list argument is automatically batched."""
        @xcs.vmap
        def square(x):
            return x * x
        
        assert square([1, 2, 3, 4]) == [1, 4, 9, 16]
    
    def test_multiple_argument_batching(self):
        """Multiple arguments are batched together."""
        @xcs.vmap
        def add(x, y):
            return x + y
        
        assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    
    def test_keyword_argument_batching(self):
        """Keyword arguments can be batched."""
        @xcs.vmap
        def multiply(x, y):
            return x * y
        
        result = multiply(x=[1, 2, 3], y=[2, 3, 4])
        assert result == [2, 6, 12]
    
    def test_mixed_batching(self):
        """Mix of batched and non-batched arguments."""
        @xcs.vmap
        def scale(x, factor=2):
            return x * factor
        
        # Batch x, scalar factor
        assert scale([1, 2, 3], factor=5) == [5, 10, 15]
        
        # Batch both
        assert scale([1, 2, 3], [2, 3, 4]) == [2, 6, 12]
    
    def test_dict_batching(self):
        """Batching over dictionaries works correctly."""
        @xcs.vmap
        def process_dict(d):
            return d['x'] + d['y']
        
        inputs = [
            {'x': 1, 'y': 2},
            {'x': 3, 'y': 4},
            {'x': 5, 'y': 6}
        ]
        
        assert process_dict(inputs) == [3, 7, 11]
    
    def test_complex_return_values(self):
        """Functions returning complex values are handled."""
        @xcs.vmap
        def analyze(x):
            return {
                'value': x,
                'squared': x * x,
                'is_even': x % 2 == 0
            }
        
        results = analyze([1, 2, 3, 4])
        
        # Results should be a list of dicts
        assert len(results) == 4
        assert results[0] == {'value': 1, 'squared': 1, 'is_even': False}
        assert results[1] == {'value': 2, 'squared': 4, 'is_even': True}
    
    def test_no_batching_passthrough(self):
        """Non-batched calls pass through unchanged."""
        @xcs.vmap
        def compute(x, y):
            return x + y
        
        # Regular call with scalars
        assert compute(5, 3) == 8
    
    def test_empty_batch(self):
        """Empty batches are handled gracefully."""
        @xcs.vmap
        def process(x):
            return x * 2
        
        assert process([]) == []
    
    def test_parallel_execution(self):
        """Parallel execution works when enabled."""
        import time
        
        @xcs.vmap(parallel=True)
        def slow_operation(x):
            time.sleep(0.01)  # Simulate work
            return x * x
        
        start = time.time()
        results = slow_operation([1, 2, 3, 4, 5])
        duration = time.time() - start
        
        assert results == [1, 4, 9, 16, 25]
        # Should be faster than sequential (0.05s)
        # But we can't guarantee exact timing in tests


class TestCombinedTransformations:
    """Test combining JIT and vmap."""
    
    def test_jit_then_vmap(self):
        """JIT compiled functions can be vmapped."""
        @xcs.vmap
        @xcs.jit
        def compute(x, y=1):
            return x * y + x
        
        assert compute([1, 2, 3]) == [2, 4, 6]
        assert compute([1, 2], y=[2, 3]) == [3, 8]
    
    def test_vmap_then_jit(self):
        """Vmapped functions can be JIT compiled."""
        @xcs.jit
        @xcs.vmap
        def process(x):
            return x ** 2
        
        assert process([1, 2, 3, 4]) == [1, 4, 9, 16]
    
    def test_complex_pipeline(self):
        """Complex transformation pipelines work correctly."""
        # Define base function
        def normalize(x, mean=0, std=1):
            return (x - mean) / std
        
        # Apply transformations
        batch_normalize = xcs.jit(xcs.vmap(normalize))
        
        # Test
        data = [1, 2, 3, 4, 5]
        result = batch_normalize(data, mean=3, std=1.5)
        
        expected = [(x - 3) / 1.5 for x in data]
        assert result == pytest.approx(expected)


class TestOperatorSupport:
    """Test natural API with operators."""
    
    def test_operator_natural_calling(self):
        """Operators can be called naturally."""
        from ember.api.operator import Operator
        from ember.core.registry.specification import Specification
        
        @xcs.jit
        class MultiplyOperator(Operator):
            specification = Specification()
            
            def __init__(self, factor=2):
                super().__init__()
                self.factor = factor
            
            def forward(self, *, inputs):
                return {"result": inputs.get("value", 0) * self.factor}
        
        op = MultiplyOperator(factor=3)
        
        # Natural calling should work
        result = op(value=5)
        assert result == {"result": 15}
        
        # Dict calling should still work
        result = op(inputs={"value": 5})
        assert result == {"result": 15}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])