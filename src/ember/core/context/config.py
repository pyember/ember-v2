"""Configuration management component.

This module provides a component for managing configuration with
efficient access patterns and lazy loading.
"""

import os
from typing import Any, Dict, Optional, cast

import yaml

from .component import Component
from .registry import Registry


class ConfigComponent(Component):
    """Configuration management component.

    This component is responsible for loading and providing access to
    configuration from files and environment variables with minimal overhead.

    Features:
    - Lazy loading: Configuration is only loaded when first accessed
    - Multiple sources: File and environment variables
    - Thread safety: Safe for concurrent access
    - Zero overhead: Cached access paths for fast repeated lookups
    """

    def __init__(
        self,
        registry: Optional[Registry] = None,
        config_path: Optional[str] = None,
        config_data: Optional[Dict[str, Any]] = None):
        """Initialize with registry and optional config source.

        Args:
            registry: Registry to use (current thread's if None)
            config_path: Path to config file (environment if None)
            config_data: Direct config data (overrides file)
        """
        super().__init__(registry)
        self._config_path = config_path or os.environ.get("EMBER_CONFIG")
        self._config_data = config_data
        self._config: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}

    def _register(self) -> None:
        """Register in registry as 'config'."""
        self._registry.register("config", self)

    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration section.

        Args:
            section: Section name or None for entire config

        Returns:
            Configuration dictionary for section or empty dict
        """
        self._ensure_initialized()

        if section is None:
            return self._config.copy()

        # Cache hit optimization
        cache_key = f"section:{section}"
        if cache_key in self._cache:
            return cast(Dict[str, Any], self._cache[cache_key])

        # Get section
        result = self._config.get(section, {})

        # Cache result
        self._cache[cache_key] = result
        return result

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get specific configuration value.

        Args:
            section: Section name
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        # Cache hit optimization
        cache_key = f"value:{section}.{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get value
        section_data = self.get_config(section)
        result = section_data.get(key, default)

        # Cache result
        self._cache[cache_key] = result
        return result

    def _initialize(self) -> None:
        """Load configuration from file or environment.

        The loading order is:
        1. Provided config_data (highest priority)
        2. Configuration file
        3. Environment variables with EMBER_ prefix
        """
        # Start with empty config
        self._config = {}

        # Use provided config data if available
        if self._config_data is not None:
            self._config = self._config_data.copy()
            return

        # Load from file if available
        if self._config_path and os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r") as f:
                    file_config = yaml.safe_load(f) or {}
                    self._config.update(file_config)
            except Exception as e:
                # Log error but continue
                import logging

                logging.error(f"Error loading config from {self._config_path}: {e}")

        # Add environment variables with EMBER_ prefix
        env_config = self._load_from_env()
        self._config.update(env_config)

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Converts environment variables with EMBER_ prefix into
        nested configuration structure.

        Returns:
            Configuration dictionary from environment
        """
        config: Dict[str, Any] = {}
        prefix = "EMBER_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                parts = key[len(prefix) :].lower().split("_")
                current = config

                # Navigate to the correct nesting level
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the value at the leaf node
                current[parts[-1]] = value

        return config

    def clear_cache(self) -> None:
        """Clear the configuration cache.

        This forces re-fetching values from the config dictionary.
        """
        self._cache.clear()
