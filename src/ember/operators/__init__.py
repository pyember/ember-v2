"""Ember operators module.

This module provides the core operator system for Ember, based on the
simplified design with automatic JAX integration.

Key components:
- Operator: Base class for all operators with optional validation
- Common operators: Ensemble, ModelCall, Chain, Router, etc.
- Context operators: ContextAgnostic, ContextAware, ContextualInput, ContextualOutput
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
    ExtractText,
    ModelText,
    ensemble,
    chain,
    router,
)

# Import context operators
from ember.operators.common_context import (
    ContextAgnostic,
    ContextAware,
    ContextualData,
    InitialContext,
)

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
    "ExtractText",
    "ModelText",

    # Convenience functions
    "ensemble",
    "chain",
    "router",
    
    # Context operators
    "ContextAgnostic",
    "ContextAware",
    "ContextualData",
    "InitialContext",
]