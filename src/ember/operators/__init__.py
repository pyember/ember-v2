"""Ember operators module.

This module provides the core operator system for Ember, based on the
simplified design with automatic JAX integration.

Key components:
- Operator: Base class for all operators with optional validation
- Common operators: Ensemble, ModelCall, Chain, Router, etc.
- Core metadata-containing objects: EmberData, EmberEmbedding
- Progressive disclosure from simple functions to complex systems
"""

# Import base operator
from ember.operators.base import Operator

# Import common operators
from ember.operators.common import (
    ModelCall,
    Ensemble,
    Chain,
    Router,
    LearnableRouter,
    Retry,
    Cache,
    ensemble,
    chain,
    router,
    EmberEmbedding
)

from ember.operators.ember_data import EmberData

__all__ = [
    # Base class
    "Operator",
    
    # Common operator classes
    "ModelCall",
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
    
    # Context operators
    "EmberData",
    "EmberEmbedding"
]