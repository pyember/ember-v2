"""Core type definitions for Ember.

This module provides the foundational types used throughout the Ember framework,
abstracting away the underlying validation library to maintain a clean API
surface.

The key abstractions are:
- EmberModel: Base class for all structured data with validation
- Field: Validation constraints for model fields

This abstraction layer serves several purposes:
1. Hides implementation details from users (they don't need to know about
   Pydantic)
2. Allows us to change the underlying validation library without breaking
   user code
3. Provides a consistent API surface across all Ember components
4. Enables future optimizations and enhancements

Example:
    from ember._internal.types import EmberModel, Field

    class UserInput(EmberModel):
        text: str = Field(min_length=1, max_length=1000)
        temperature: float = Field(ge=0.0, le=2.0, default=1.0)
"""

from pydantic import BaseModel
from pydantic import Field as PydanticField

# EmberModel is a simple alias for Pydantic's BaseModel, providing
# full validation features with zero overhead.
EmberModel = BaseModel

# Field provides validation constraints without exposing pydantic directly
# This abstraction allows us to change the underlying implementation later
Field = PydanticField

__all__ = ["EmberModel", "Field"]
