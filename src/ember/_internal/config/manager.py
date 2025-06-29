"""Configuration manager for Ember.

Single source of truth for configuration with clear precedence rules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from ember.core.config.loader import load_config
from ember.core.credentials import CredentialManager


class ConfigManager:
    """Manages configuration with precedence: overrides > env > file > defaults.

    Example:
        manager = create_config_manager()
        api_key = manager.get_api_key("openai")
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        credential_manager: Optional[CredentialManager] = None,
    ):
        """Initialize manager.

        Args:
            config_path: Optional config file path.
            credential_manager: Optional credential manager.
        """
        self._config = {}
        self._overrides = {}
        self._credential_manager = credential_manager or CredentialManager()

        # Load configuration file if provided
        if config_path:
            self._config = load_config(str(config_path))

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with precedence.

        Args:
            key: Dot-notation key.
            default: Default if not found.

        Returns:
            Value or default.
        """
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]

        # Check environment with EMBER_ prefix
        env_key = f"EMBER_{key.upper().replace('.', '_')}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value

        # Check config file
        parts = key.split(".")
        value = self._config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration override.

        Args:
            key: Configuration key
            value: Value to set
        """
        self._overrides[key] = value

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key with precedence: env > credentials > config.

        Args:
            provider: Provider name.

        Returns:
            API key or None.
        """
        # Map provider names to env vars
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }

        # Check provider-specific env var
        env_var = env_vars.get(provider.lower())
        if env_var:
            value = os.environ.get(env_var)
            if value:
                return value

        # Check Ember-specific env var
        ember_env = f"EMBER_{provider.upper()}_API_KEY"
        value = os.environ.get(ember_env)
        if value:
            return value

        # Check credentials file
        value = self._credential_manager.get(provider)
        if value:
            return value

        # Check config file
        return self.get(f"providers.{provider}.api_key")

    def set_provider_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider.

        Args:
            provider: Provider name
            api_key: API key to set
        """
        # Store in credential manager for persistence
        self._credential_manager.store(provider, api_key)

        # Also set as override for immediate use
        self.set(f"providers.{provider}.api_key", api_key)


# Global instance
_global_manager: Optional[ConfigManager] = None


def create_config_manager(config_path: Optional[Path] = None, reset: bool = False) -> ConfigManager:
    """Get or create global configuration manager.

    Args:
        config_path: Optional config file path.
        reset: Force new instance.

    Returns:
        ConfigManager instance.
    """
    global _global_manager

    if reset or _global_manager is None:
        _global_manager = ConfigManager(config_path=config_path)

    return _global_manager
