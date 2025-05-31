"""Ensemble operators for v2 system.

Provides simple ensemble patterns for combining multiple operations.
"""

from typing import List, Callable, Any


class Ensemble:
    """Simple ensemble that calls multiple functions."""
    
    def __init__(self, functions: List[Callable]):
        self.functions = functions
    
    def __call__(self, *args, **kwargs) -> List[Any]:
        """Call all functions and return results."""
        return [f(*args, **kwargs) for f in self.functions]


def ensemble(*functions: Callable) -> Callable:
    """Create an ensemble from multiple functions.
    
    Args:
        *functions: Functions to ensemble
        
    Returns:
        Function that returns list of all results
    """
    def ensemble_wrapper(*args, **kwargs):
        return [f(*args, **kwargs) for f in functions]
    return ensemble_wrapper