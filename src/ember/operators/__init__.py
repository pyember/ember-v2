"""Ember operators module.

This module provides the core operator system for Ember, based on the
simplified design with automatic JAX integration.

Key components:
- Operator: Base class for all operators with optional validation
- Common operators: Ensemble, Chain, Router, etc.
- Progressive disclosure from simple functions to complex systems
"""

# Import base operator
from ember.operators.base import Operator

# Import common operators
from ember.operators.common import (
    Ensemble,
    Chain,
    Router,
    LearnableRouter,
    Retry,
    Cache,
    ensemble,
    chain,
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