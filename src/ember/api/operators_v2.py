"""Simplified operators API for Ember v2.

This module provides the new, simplified operators API. Operators are just
callables - functions or objects with __call__ methods. No base classes required.

Example:
    >>> from ember.api import models
    >>> from ember.xcs import jit, vmap  # Transformations come from XCS
    >>> 
    >>> # Any function is an operator
    >>> def classify(text: str) -> str:
    ...     response = models("gpt-4", f"Classify sentiment: {text}")
    ...     return response.text
    >>> 
    >>> # XCS can transform any operator
    >>> fast_classify = jit(classify)
    >>> batch_classify = vmap(classify)
"""

from ember.core.operators_v2 import Operator

__all__ = [
    "Operator",
]