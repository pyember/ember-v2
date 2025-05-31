"""Simplified module system for stateful operators - without JAX dependency.

Provides a minimal base class for operators that require state management.
This is a simplified version that doesn't require JAX.
"""

from dataclasses import dataclass, field, replace
from typing import Any, TypeVar, Dict, Callable


T = TypeVar('T')


def static_field(default=None, default_factory=None):
    """Mark a field as static (not part of transformations)."""
    if default is not None and default_factory is not None:
        raise ValueError("cannot specify both default and default_factory")
    
    if default_factory is not None:
        return field(default_factory=default_factory, metadata={"static": True})
    else:
        return field(default=default, metadata={"static": True})


class EmberModule:
    """Base class for stateful operators.
    
    Subclasses are automatically converted to frozen dataclasses.
    """
    
    def __init_subclass__(cls, **kwargs):
        """Automatically make subclasses into frozen dataclasses."""
        super().__init_subclass__(**kwargs)
        
        # Convert to dataclass if not already
        if not hasattr(cls, '__dataclass_fields__'):
            cls = dataclass(frozen=True)(cls)
        
        # Add replace method
        cls.replace = lambda self, **kwargs: replace(self, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Subclasses should override this."""
        raise NotImplementedError("Subclasses must implement __call__")


class SignatureOperator(EmberModule):
    """Base for operators with typed signatures.
    
    Simplified version that just inherits from EmberModule.
    """
    pass