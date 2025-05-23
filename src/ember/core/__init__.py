"""Core components of the Ember framework."""

from __future__ import annotations

# Absolute imports
try:
    # Try the normal import path first
    from ember.core import registry
    from ember.core.context import EmberContext, current_context
    from ember.core.exceptions import ConfigError, EmberError, ValidationError
except ImportError:
    # Fall back to src.ember if the regular imports fail
    from ember.core import registry
    from ember.core.context import EmberContext, current_context
    from ember.core.exceptions import ConfigError, EmberError, ValidationError

# Legacy alias for backward compatibility
ConfigurationError = ConfigError

__all__ = [
    "EmberContext",
    "EmberError",
    "ValidationError",
    "ConfigError",
    "ConfigurationError",  # Legacy alias
    "registry",
]
