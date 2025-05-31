"""Ember Operators v2: Simple, Powerful, Composable.

This module implements the redesigned operator system following principles of
radical simplicity and functional composition.

The operator system defines WHAT operators are (callables that transform inputs).
The XCS system defines HOW to transform them (jit, vmap, pmap).
"""

from ember.core.operators_v2.protocols import Operator

__all__ = [
    "Operator",
]