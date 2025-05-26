"""Exception definitions for specification module.

This module provides a compatibility layer that re-exports exceptions from the
core exceptions module while maintaining backward compatibility with existing code.
Prefer using the exceptions directly from ember.core.exceptions in new code.
"""

from ember.core.exceptions import InvalidPromptError as PromptSpecificationError
from ember.core.exceptions import SpecificationValidationError

# Re-export specification exceptions for backward compatibility
__all__ = [
    "PromptSpecificationError",
    "SpecificationValidationError"]
