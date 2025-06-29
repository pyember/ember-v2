"""Ember public exceptions API.

This module exposes the exception types that users need for error handling.
All exceptions are re-exported from internal modules to maintain clean abstraction boundaries.

Architecture Philosophy:
    Exceptions are part of the public API contract. Users need to catch and handle
    specific exceptions, so we expose only the exceptions they should interact with.

    This module acts as a facade over the internal exception hierarchy, exposing
    only what's necessary while keeping implementation details hidden.

Common usage patterns:
    >>> from ember.api import exceptions
    >>>
    >>> try:
    >>>     result = models("gpt-4", prompt)
    >>> except exceptions.ModelNotFoundError:
    >>>     # Handle missing model
    >>>     pass
    >>> except exceptions.ProviderAPIError as e:
    >>>     # Handle API errors
    >>>     print(f"API error: {e}")
"""

# Import from internal modules
from ember._internal.exceptions import (
    # Configuration exceptions
    ConfigError,
    ConfigValidationError,
    # Data exceptions
    DataError,
    DataLoadError,
    DatasetNotFoundError,
    DataValidationError,
    # Base exceptions
    EmberError,
    EmberException,  # Legacy alias
    ErrorGroup,
    InitializationError,
    InvalidArgumentError,
    InvalidPromptError,
    MissingConfigError,
    # Model exceptions
    ModelError,
    ModelNotFoundError,
    # Operator exceptions
    OperatorError,
    OperatorExecutionError,
    ProviderAPIError,
    ProviderConfigError,
    SpecificationValidationError,
    # Core exceptions
    ValidationError,
)

# Create legacy aliases for backward compatibility
OperatorException = OperatorError
ModelException = ModelError
ValidationException = ValidationError

__all__ = [
    # Base exceptions
    "EmberError",
    "EmberException",
    "ErrorGroup",
    # Core exceptions
    "ValidationError",
    "ValidationException",  # Legacy alias
    "InvalidArgumentError",
    "InitializationError",
    # Model exceptions
    "ModelError",
    "ModelException",  # Legacy alias
    "ModelNotFoundError",
    "ProviderAPIError",
    "ProviderConfigError",
    "InvalidPromptError",
    # Operator exceptions
    "OperatorError",
    "OperatorException",  # Legacy alias
    "OperatorExecutionError",
    "SpecificationValidationError",
    # Data exceptions
    "DataError",
    "DataValidationError",
    "DataLoadError",
    "DatasetNotFoundError",
    # Configuration exceptions
    "ConfigError",
    "ConfigValidationError",
    "MissingConfigError",
]
