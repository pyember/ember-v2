"""Core components of the Ember framework."""

from __future__ import annotations

# Absolute imports
try:
    # Try the normal import path first
    # from ember._internal import registry  # Deprecated
    # from ember._internal.context import EmberContext, current_context  # Not used by new API
    from ember._internal.exceptions import ConfigError, EmberError, ValidationError
except ImportError:
    # Fall back to src.ember if the regular imports fail
    # from ember._internal import registry  # Deprecated
    # from ember._internal.context import EmberContext, current_context  # Not used by new API
    from ember._internal.exceptions import ConfigError, EmberError, ValidationError

# Legacy alias for backward compatibility
ConfigurationError = ConfigError

__all__ = [
    # "EmberContext",  # Not used by new API
    "EmberError",
    "ValidationError",
    "ConfigError",
    "ConfigurationError",  # Legacy alias
    # "registry"  # Deprecated
]
