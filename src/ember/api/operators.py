"""Operators API for Ember.

This module provides the public API for Ember's operator system, exposing
the simplified design with progressive disclosure.

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
from ember.operators import (
    Operator, ModelCall, Ensemble, Chain, Router, LearnableRouter, 
    ModelText, ExtractText, ContextAgnostic, ContextAware, ContextualData, InitialContext,
    chain, ensemble, router
)

__all__ = [
    "Operator",
    "op",
    # Common operators
    "ModelCall",
    "Ensemble", 
    "Chain",
    "Router",
    "LearnableRouter", 
    "ModelText",
    "ExtractText",
    # Context operators
    "ContextAgnostic",
    "ContextAware", 
    "ContextualData",
    "InitialContext",
    # Convenience functions
    "chain",
    "ensemble", 
    "router",
]