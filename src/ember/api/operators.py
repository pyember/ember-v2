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
from ember.operators import Operator, chain, ensemble

__all__ = [
    "Operator",
    "chain",
    "ensemble",
    "op",
]