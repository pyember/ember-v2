"""Public API for Ember context management.

This module provides the public interface to Ember's context system,
implementing a simplified API that reduces cognitive load while maintaining
full functionality.
"""

from contextlib import contextmanager
from typing import Any

from ember._internal.context import EmberContext as _EmberContext
from ember._internal.context import current_context as _current_context

EmberContext = _EmberContext


class ContextAPI:
    """Simplified context API following the principle of one obvious way.

    This class provides a clean, namespace-based API for context management,
    reducing the number of top-level functions while maintaining clarity.
    """

    @staticmethod
    def get() -> _EmberContext:
        """Get the current Ember context.

        Returns:
            The current thread/async context instance. Creates one if needed.

        Examples:
            >>> ctx = context.get()
            >>> api_key = ctx.get_credential("openai", "OPENAI_API_KEY")

        Note:
            This is the primary way to access the current context. The context
            is thread-safe and async-safe, propagating correctly across boundaries.
        """
        return _current_context()

    @staticmethod
    @contextmanager
    def manager(**config_overrides: Any):
        """Create a context manager for temporary configuration overrides.

        This unified API provides a single, clear way to manage context scopes.

        Args:
            **config_overrides: Configuration values to override in this scope.
                Supports nested dictionaries for deep configuration.

        Yields:
            EmberContext: Context with applied overrides.

        Examples:
            >>> with context.manager(models={"default": "gpt-4"}) as ctx:
            ...     response = models("Hello")  # Uses gpt-4

            >>> with context.manager(models={"default": "gpt-4", "temperature": 0.9}) as ctx:
            ...     # All operations in this block use these settings
            ...     pass

        Note:
            The context manager ensures proper cleanup and restoration of the
            previous context when exiting the scope.
        """
        current = _current_context()
        child = current.create_child(**config_overrides)
        with child as ctx:
            yield ctx


context = ContextAPI()


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
    return context.get().get_config(key, default)


def set_config(key: str, value: Any) -> None:
    """Set configuration value in current context.

    Args:
        key: Configuration key using dot notation.
        value: Value to set.

    Examples:
        >>> set_config("models.temperature", 0.7)
    """
    context.get().set_config(key, value)


__all__ = [
    "context",
    "EmberContext",
    "get_config",
    "set_config",
]
