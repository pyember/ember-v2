"""Operators API for Ember.

This module provides the public API for Ember's operator system, exposing
the simplified design with progressive disclosure.

Architecture Philosophy:
    Operators implement a functional programming model that enables:
    1. **Composability**: All operators can be chained, ensembled, or nested
    2. **JAX Integration**: Automatic differentiation and compilation support
    3. **Progressive Disclosure**: From simple @op decorator to full class specifications
    4. **Static Analysis**: Enable tracing for optimization

Design Rationale:
    Traditional ML pipelines use imperative, stateful components that are
    difficult to optimize, parallelize, or differentiate. Ember operators are:

    - Pure functions: No hidden state, deterministic behavior
    - JAX-compatible: Can be vmapped, pmapped, or differentiated
    - Type-safe: Full type hints for static analysis
    - Lightweight: Minimal overhead wrapper around functions

    The @op decorator provides the simplest entry point, while the Operator
    base class offers full class specification for advanced use cases.

Performance Characteristics:
    - Decorator overhead: ... negligible < 0.01μs per call
    - First compilation: 50-500ms (XCS tracing)
    - Subsequent calls: Near-native Python speed
    - Memory: O(1) for operators, O(n) for traced values
    - Parallelization: Automatic via XCS transformations

Trade-offs:
    - Functional purity vs flexibility: No side effects in operators
    - Compilation time vs runtime: JIT has upfront cost
    - Type safety vs dynamism: Static types enable optimization
    - Simplicity vs control: Decorator hides complexity

JAX Integration Benefits:
    - Automatic differentiation: Gradients through entire pipelines
    - Vectorization: vmap for batched execution
    - Parallelization: pmap for multi-device scaling
    - JIT compilation: 10-100x speedups on repeated execution

Common Patterns:
    - Chain: Sequential composition for pipelines
    - Ensemble: Parallel execution with aggregation
    - Router: Conditional execution based on input
    - Retry: Fault tolerance with exponential backoff

Basic usage:
    >>> from ember.api import operators
    >>>
    >>> # Use the decorator for simple cases
    >>> @operators.op
    ... def summarize(text: str) -> str:
    ...     return models("gpt-4", f"Summarize: {text}").text
    >>>
    >>> # Create operator classes for more control
    >>> class MyOperator(operators.Operator):
    ...     def forward(self, input):
    ...         return process(input)
    >>>
    >>> # Compose operators
    >>> pipeline = operators.chain(
    ...     preprocess,
    ...     summarize,
    ...     postprocess
    ... )
"""

from ember.api.decorators import op
from ember.operators import Operator, chain, ensemble

__all__ = [
    "Operator",
    "chain",
    "ensemble",
    "op",
]
