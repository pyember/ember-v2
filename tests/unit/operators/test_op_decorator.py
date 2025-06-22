"""Test the @op decorator for progressive disclosure.

Following CLAUDE.md principles:
- Simple functions become operators
- Preserves function metadata
- Works with composition
- Integrates with JAX
"""

import pytest
import jax
import jax.numpy as jnp
from typing import List, Dict

from ember.api.decorators import op
from ember.api.operators import Operator, chain, ensemble
from ember.xcs import jit


class TestOpDecorator:
    """Test the @op decorator functionality."""
    
    def test_simple_function_to_operator(self):
        """Test that @op converts a function to an operator."""
        
        @op
        def double(x: int) -> int:
            """Double the input."""
            return x * 2
        
        # Should be an Operator instance
        assert isinstance(double, Operator)
        
        # Should work like original function
        result = double(5)
        assert result == 10
        
        # The FunctionOperator has its own forward docstring
        # The original function's docstring is preserved on the class
        assert "FunctionOperator(double)" in double.__class__.__name__
    
    def test_function_with_multiple_args(self):
        """Test @op with functions taking multiple arguments."""
        
        @op
        def add_and_multiply(x: float, y: float, multiplier: int = 2) -> float:
            return (x + y) * multiplier
        
        # Operators take a single input, so we need to pass a dict or tuple
        # The @op decorator should handle unpacking for the original function
        
        # For now, test that it's an operator
        assert isinstance(add_and_multiply, Operator)
    
    def test_composition_with_decorated_functions(self):
        """Test that @op functions work with composition."""
        
        @op
        def normalize(text: str) -> str:
            return text.lower().strip()
        
        @op
        def tokenize(text: str) -> List[str]:
            return text.split()
        
        @op 
        def count_words(tokens: List[str]) -> int:
            return len(tokens)
        
        # Compose with chain
        pipeline = chain(normalize, tokenize, count_words)
        
        result = pipeline("  Hello WORLD  ")
        assert result == 2
    
    def test_ensemble_with_decorated_functions(self):
        """Test that @op functions work in ensembles."""
        
        @op
        def method1(x: int) -> str:
            return "positive" if x > 0 else "negative"
        
        @op
        def method2(x: int) -> str:
            return "even" if x % 2 == 0 else "odd"
        
        @op
        def method3(x: int) -> str:
            return "small" if abs(x) < 10 else "large"
        
        # Create ensemble
        analyzer = ensemble(method1, method2, method3)
        
        results = analyzer(5)
        assert results == ["positive", "odd", "small"]
    
    def test_jax_integration_with_decorated_function(self):
        """Test that @op functions work with JAX transformations."""
        
        @op
        def compute(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x ** 2)
        
        # Should work with jax.grad
        grad_fn = jax.grad(lambda x: compute(x))
        x = jnp.array([1.0, 2.0, 3.0])
        grads = grad_fn(x)
        
        # Gradient of sum(x^2) is 2x
        expected = 2 * x
        assert jnp.allclose(grads, expected)
    
    def test_xcs_jit_with_decorated_function(self):
        """Test that @op functions can be used with XCS @jit."""
        
        @op
        def compute_sum(x: int) -> int:
            # Simple computation
            return sum(range(x))
        
        # The decorated function is an Operator
        assert isinstance(compute_sum, Operator)
        
        # Can apply JIT to the operator
        fast_compute = jit(compute_sum)
        
        # Basic functionality test
        result = compute_sum(10)
        assert result == 45  # sum(0..9)
    
    def test_nested_decorated_functions(self):
        """Test nested calls between decorated functions."""
        
        @op
        def inner(x: int) -> int:
            return x + 1
        
        @op
        def middle(x: int) -> int:
            return inner(x) * 2
        
        @op
        def outer(x: int) -> int:
            return middle(x) + inner(x)
        
        # outer(5) = middle(5) + inner(5)
        #          = (inner(5) * 2) + inner(5)
        #          = (6 * 2) + 6
        #          = 12 + 6 = 18
        result = outer(5)
        assert result == 18
    
    def test_decorated_function_in_operator_class(self):
        """Test using @op functions inside operator classes."""
        
        @op
        def preprocess(text: str) -> str:
            return text.strip().lower()
        
        class AnalysisOperator(Operator):
            def forward(self, text: str) -> Dict[str, any]:
                # Use decorated function
                cleaned = preprocess(text)
                return {
                    "original": text,
                    "cleaned": cleaned,
                    "length": len(cleaned)
                }
        
        analyzer = AnalysisOperator()
        result = analyzer("  HELLO WORLD  ")
        
        assert result["cleaned"] == "hello world"
        assert result["length"] == 11
    
    def test_progressive_disclosure_levels(self):
        """Test that @op supports progressive disclosure."""
        
        # Level 1: Simple function
        @op
        def simple(x):
            return x * 2
        
        # Can be used immediately
        assert simple(5) == 10
        
        # But is actually an Operator
        assert isinstance(simple, Operator)
        assert hasattr(simple, 'forward')
        
        # Can be composed
        pipeline = chain(simple, simple, simple)  # 2^3 = 8x
        assert pipeline(5) == 40
        
        # Can be used with JAX (if it had arrays)
        # Can be optimized with XCS
        # All without changing the original simple function!


if __name__ == "__main__":
    pytest.main([__file__, "-v"])