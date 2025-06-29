"""Configuration management for Ember."""

from ember._internal.config.manager import (
    ConfigManager,
    create_config_manager,
)

__all__ = [
    "ConfigManager",
    "create_config_manager",
]
