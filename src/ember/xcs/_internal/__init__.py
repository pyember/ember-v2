"""Internal XCS implementation.

This module is private. Users should not import from here.
All public API is in ember.xcs.

Architecture Notes:
    The _internal package contains the sophisticated analysis and compilation
    machinery that powers XCS's intelligent transformations. Key components:

    - analysis.py: Static analysis to identify pure vs impure code paths
    - ir.py: Intermediate representation for hybrid computation graphs
    - lowering.py: Conversion from IR to executable JAX/Python code
    - pytree_registration.py: JAX pytree support for Ember types

Design Philosophy:
    The internal implementation follows a classic compiler architecture:
    1. Analysis: Understand code structure and dependencies
    2. IR Construction: Build computation graph with mixed nodes
    3. Optimization: Apply transformations (batching, caching)
    4. Lowering: Generate optimized executable code

    This separation enables sophisticated optimizations while maintaining
    a simple user interface (@jit decorator).

Implementation Trade-offs:
    - Complexity hidden here vs simple public API
    - Analysis overhead vs optimization benefits
    - Memory for IR vs execution speed
    - Generality vs specific optimizations
"""

# Auto-register pytrees on import to fix JAX warnings
from ember.xcs._internal.pytree_registration import register_ember_pytrees

register_ember_pytrees()

# Private module - not part of public API
__all__ = []
