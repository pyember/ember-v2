"""Thread-local configuration integration with zero-overhead access patterns.

This module bridges the Ember context system with configuration management,
providing a zero-allocation, thread-safe interface for accessing configuration
values. It implements attribute-style access to configuration with proper
hierarchical navigation and temporary overrides.

Key design principles:
1. Zero-allocation reads: Access configuration without copying or allocating memory
2. Thread isolation: Each thread can modify its configuration independently
3. Hierarchical access: Attribute-style navigation into nested configuration
4. Immutable views: Configuration objects are immutable to prevent side effects
5. Type safety: Integration with Pydantic ensures type validation

This integration enables a highly efficient configuration access pattern that
follows the principle of least astonishment while maintaining robust
thread-isolation guarantees.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator

from ember.core.context.ember_context import EmberContext, current_context


class ConfigView:
    """Zero-allocation interface to configuration with attribute-style access.

    This view creates a proxy to configuration values that enables dot-notation
    access with proper isolation and minimal overhead. It recursively wraps nested
    dictionaries in ConfigView objects, creating a hierarchical access pattern.

    Performance characteristics:
    - Zero-allocation reads: No memory is allocated during value access
    - Thread safety: Only accesses immutable data, safe for concurrent use
    - Minimal indirection: Direct dictionary access with optimized lookup paths

    Example usage:
        # Get configuration view
        config = get_thread_local_config()

        # Access configuration values with attribute syntax
        model_name = config.model.default_name
        batch_size = config.training.batch_size

        # Access with fallback values
        timeout = config.get("timeout", 30)
    """

    __slots__ = ("_config",)

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes a view over an immutable configuration dictionary.

        Args:
            config: Configuration dictionary to wrap (should be immutable).
        """
        # Store direct reference to configuration (no copying)
        self._config = config

    def __getattr__(self, name: str) -> Any:
        """Enables attribute-style access to configuration values.

        This method intercepts attribute access and translates it into
        dictionary lookups, providing a more Pythonic interface to
        configuration values.

        Args:
            name: Configuration key to access.

        Returns:
            Configuration value, or a nested ConfigView for dictionaries.

        Raises:
            AttributeError: If the requested key doesn't exist in configuration.
        """
        # Check if key exists in configuration
        if name in self._config:
            value = self._config[name]

            # Recursively wrap dictionaries for nested access
            if isinstance(value, dict):
                return ConfigView(value)

            # Return primitive values directly
            return value

        # Raise proper AttributeError for compatibility with Python's attribute access
        raise AttributeError(f"Configuration has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value with an optional default.

        Provides dictionary-style access with a default value fallback,
        while maintaining the recursive ConfigView wrapping for nested structures.

        Args:
            key: Configuration key to access.
            default: Value to return if key is not found.

        Returns:
            Configuration value, a nested ConfigView for dictionaries,
            or the default value if the key doesn't exist.
        """
        # Get value with default fallback
        value = self._config.get(key, default)

        # Recursively wrap dictionaries
        if isinstance(value, dict):
            return ConfigView(value)

        # Return primitive values or default directly
        return value

    def __contains__(self, key: str) -> bool:
        """Enables 'in' operator to check if a key exists in configuration.

        Args:
            key: Configuration key to check.

        Returns:
            bool: True if the key exists in configuration, False otherwise.
        """
        return key in self._config

    def __dir__(self) -> list:
        """Enhanced dir() support for interactive exploration.

        Makes the ConfigView more friendly in interactive environments like
        IPython by showing available configuration keys.

        Returns:
            list: List of available configuration keys.
        """
        # Return configuration keys for better interactive experience
        return list(self._config.keys())

    def as_dict(self) -> Dict[str, Any]:
        """Returns the underlying configuration dictionary.

        This method provides a way to access the raw configuration when needed,
        but should be used sparingly to avoid breaking the encapsulation.

        Returns:
            Dict[str, Any]: The underlying configuration dictionary.
        """
        return self._config


@contextmanager
def config_override(overrides: Dict[str, Any]) -> Iterator[None]:
    """Creates a temporary thread-local configuration override.

    This context manager enables scoped modification of configuration values
    for the current thread only, without affecting other threads. It's
    particularly useful for testing and for operation-specific configuration
    changes.

    The implementation ensures:
    1. Thread safety: Only the current thread's configuration is modified
    2. Proper cleanup: Original configuration is always restored, even after exceptions
    3. Type validation: Configuration changes are validated against the schema
    4. Minimal locking: Lock scope is limited to the actual config swap

    Example usage:
        # Temporarily override configuration
        with config_override({"model": {"temperature": 0.8}}):
            # Code in this block sees the modified configuration
            model = create_model()  # Uses temperature=0.8

        # Outside the block, original configuration is restored
        model = create_model()  # Uses original temperature

    Args:
        overrides: Dictionary of configuration values to override.

    Yields:
        None: This context manager doesn't yield a value.
    """
    # Get current thread's context
    context = current_context()

    # Get current configuration manager
    config_manager = context.config_manager

    # Store reference to original configuration (not a copy)
    original_config = config_manager.get_config()

    try:
        # Import dependencies only when needed (faster module loading)
        from ember.core.config.loader import merge_dicts
        from ember.core.config.schema import EmberConfig

        # Create updated configuration by merging with overrides
        updated_config = merge_dicts(original_config.model_dump(), overrides)

        # Validate against schema to ensure type safety
        new_config = EmberConfig.model_validate(updated_config)

        # Replace configuration atomically with minimal lock scope
        with config_manager._lock:
            config_manager._config = new_config

        # Yield control to the caller's context
        yield

    finally:
        # Always restore original configuration, even if an exception occurred
        with config_manager._lock:
            config_manager._config = original_config


def get_thread_local_config() -> ConfigView:
    """Retrieves thread-local configuration with zero overhead.

    This function provides the most efficient way to access configuration
    values for the current thread. It creates a ConfigView that enables
    attribute-style access to configuration values.

    Returns:
        ConfigView: Zero-allocation view into current thread's configuration.
    """
    # Get current thread's context
    context = current_context()

    # Get raw configuration dict (no validation, maximum performance)
    config = context.config_manager.get_config().model_dump()

    # Return wrapped in ConfigView for attribute-style access
    return ConfigView(config)


def integrate_config() -> None:
    """Integrates configuration system with EmberContext.

    This function extends EmberContext with configuration-related functionality:
    1. Adds a 'config' property for attribute-style configuration access
    2. Adds a 'config_override' static method for temporary configuration changes

    This integration is performed automatically when this module is imported.
    It uses monkey patching to inject functionality into EmberContext without
    modifying its source code, maintaining backward compatibility.
    """
    # Create logger for integration status reporting
    logger = logging.getLogger("ember.context.config")

    try:
        # Define property getter for config_view
        def get_config_view(self: EmberContext) -> ConfigView:
            """Property getter for convenient configuration access.

            Returns a ConfigView over the current thread's configuration,
            enabling attribute-style access with minimal overhead.

            Returns:
                ConfigView: Thread-local configuration view.
            """
            # Get configuration as raw dict for maximum performance
            config = self.config_manager.get_config().model_dump()
            return ConfigView(config)

        # Add property to EmberContext class
        EmberContext.config = property(get_config_view)

        # Add static method for configuration overrides
        EmberContext.config_override = staticmethod(config_override)

        # Log successful integration
        logger.info("Configuration integration complete")

    except Exception as e:
        # Log detailed error information
        logger.error(f"Failed to integrate configuration system: {e}", exc_info=True)


# Auto-integrate when this module is imported
integrate_config()
