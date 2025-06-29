"""Ember operators module.

This module provides the core operator system for Ember, based on the
simplified design with automatic JAX integration.

Key components:
- Operator: Base class for all operators with optional validation
- Common operators: Ensemble, Chain, Router, etc.
- Progressive disclosure from simple functions to complex systems

Architecture Philosophy:
    Operators implement a functional programming model inspired by JAX's
    transformation system. Key principles:

    1. **Composability**: All operators can be composed with others
    2. **Static Analysis**: JAX can trace through operator pipelines
    3. **Automatic Differentiation**: Learnable parameters work seamlessly
    4. **Pure Functions**: Operators are stateless transformations

Design Rationale:
    Traditional ML pipelines use imperative, stateful components that are
    difficult to optimize, parallelize, or differentiate. Ember's operators
    are pure functions that enable:

    - JIT compilation of entire pipelines
    - Automatic batching via vmap
    - Gradient flow through mixed symbolic/neural components
    - Deterministic behavior and easy testing

Performance Characteristics:
    - First call: Tracing overhead for JIT compilation
    - Subsequent calls: Near-native performance
    - Memory: Operators are lightweight wrappers
    - Scaling: Automatic parallelization via JAX

Trade-offs:
    - Purity over flexibility: No side effects in operators
    - Compilation time vs runtime: JIT has upfront cost
    - Type safety vs dynamism: Static types enable optimization
"""

# Import base operator
from ember.operators.base import Operator

# Import common operators
from ember.operators.common import (
    Cache,
    Chain,
    Ensemble,
    LearnableRouter,
    Retry,
    Router,
    chain,
    ensemble,
    router,
)

__all__ = [
    # Base class
    "Operator",
    # Common operator classes
    "Ensemble",
    "Chain",
    "Router",
    "LearnableRouter",
    "Retry",
    "Cache",
    # Convenience functions
    "ensemble",
    "chain",
    "router",
]
