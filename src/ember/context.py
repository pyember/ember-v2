"""Public API for Ember context management.

This module provides the public interface to Ember's context system,
avoiding the need for users or tests to import from _internal packages.
"""

from typing import Any, Dict, Optional

from ember._internal.context import EmberContext as _EmberContext
from ember._internal.context import current_context, with_context


# Re-export the main class with public alias
EmberContext = _EmberContext


def get_context() -> _EmberContext:
    """Get the current Ember context.
    
    Returns:
        The current thread/async context instance.
        
    Examples:
        >>> ctx = get_context()
        >>> api_key = ctx.get_credential("openai", "OPENAI_API_KEY")
    """
    return current_context()


def create_context(**config_overrides: Any) -> _EmberContext:
    """Create a new isolated context with configuration overrides.
    
    Args:
        **config_overrides: Configuration values to override.
        
    Returns:
        New isolated EmberContext instance.
        
    Examples:
        >>> ctx = create_context(models={"default": "gpt-4"})
        >>> with ctx:
        ...     # Use gpt-4 as default model in this context
        ...     model = ctx.get_model()
    """
    current = current_context()
    return current.create_child(**config_overrides)


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value from current context.
    
    Args:
        key: Configuration key using dot notation.
        default: Default value if key not found.
        
    Returns:
        Configuration value or default.
        
    Examples:
        >>> get_config("models.default")
        'gpt-3.5-turbo'
    """
    return current_context().get_config(key, default)


def set_config(key: str, value: Any) -> None:
    """Set configuration value in current context.
    
    Args:
        key: Configuration key using dot notation.
        value: Value to set.
        
    Examples:
        >>> set_config("models.temperature", 0.7)
    """
    current_context().set_config(key, value)


# Export commonly used functions
__all__ = [
    'EmberContext',
    'get_context',
    'create_context', 
    'get_config',
    'set_config',
    'current_context',
    'with_context',
]