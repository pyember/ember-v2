"""Compatibility layer for existing code.

This module provides backward compatibility with the old EmberContext system,
allowing existing code to continue working while the new system is adopted.

Note: This is a transitional layer and should not be used for new code.
"""

from typing import Any, Dict, Optional

from .config import ConfigComponent
from .data import DataComponent
from .model import ModelComponent
from .registry import Registry


class EmberContext:
    """Legacy compatibility layer for EmberContext.

    This class provides the same interface as the old EmberContext,
    but uses the new component-based implementation internally.
    """

    def __init__(self, registry: Optional[Registry] = None):
        """Initialize with registry.

        Args:
            registry: Registry to use (current thread's if None)
        """
        self._registry = registry or Registry.current()

        # Ensure core components exist
        if not self._registry.get("config"):
            ConfigComponent(self._registry)

        if not self._registry.get("model"):
            ModelComponent(self._registry)

        if not self._registry.get("data"):
            DataComponent(self._registry)

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from model component.

        Args:
            model_id: Model identifier

        Returns:
            Model instance or None if not found
        """
        model_component = self._registry.get("model")
        if model_component:
            return model_component.get_model(model_id)
        return None

    def get_dataset(self, dataset_id: str) -> Optional[Any]:
        """Get dataset from data component.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset or None if not found
        """
        data_component = self._registry.get("data")
        if data_component:
            return data_component.get_dataset(dataset_id)
        return None

    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration from config component.

        Args:
            section: Section name or None for entire config

        Returns:
            Configuration dictionary
        """
        config_component = self._registry.get("config")
        if config_component:
            return config_component.get_config(section)
        return {}


# Thread-local for current context
_thread_local_context = Registry._thread_local


def current_context() -> EmberContext:
    """Get current thread's context.

    Returns:
        Current EmberContext instance
    """
    if not hasattr(_thread_local_context, "ember_context"):
        _thread_local_context.ember_context = EmberContext()
    return _thread_local_context.ember_context
