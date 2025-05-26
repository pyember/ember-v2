"""
Model Context System

This module defines the ModelContext class which serves as a dependency container
for all model-related components. It follows the dependency injection pattern to
eliminate global state, improve testability, and enhance modularity.

The ModelContext class encapsulates:
1. Model Registry: Manages model registration and lookup
2. Model Service: Provides a high-level API for invoking models
3. Usage Service: Tracks model usage statistics

Usage:
    # Create a custom context
    config = ModelConfig(auto_discover=True, api_keys={"openai": "sk-..."})
    custom_context = ModelContext(config=config)

    # Use with the model API
    from ember.api.models import ModelAPI
    model = ModelAPI(model_id="anthropic:claude-3-5-sonnet", context=custom_context)
    response = model.generate("Explain quantum computing to me.")
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

from ember.core.config.manager import ConfigManager, create_config_manager
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.initialization import initialize_registry

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Immutable configuration for the model system.

    This class defines the configuration options for the model system components
    like the registry, services, and discovery behavior.

    Attributes:
        auto_discover (bool): Whether to automatically discover models.
        api_keys (Dict[str, str]): API keys for providers.
        default_timeout (int): Default timeout for model invocations.
        registry_class (Type[ModelRegistry]): Class to use for the registry.
        service_class (Type[ModelService]): Class to use for the model service.
    """

    auto_discover: bool = True
    api_keys: Optional[Dict[str, str]] = None
    default_timeout: int = 30
    registry_class: Type[ModelRegistry] = ModelRegistry
    service_class: Type[ModelService] = ModelService
    config_path: Optional[str] = None
    config_manager: Optional[ConfigManager] = None


class ModelContext:
    """Container for model system dependencies.

    The ModelContext class serves as a central container for all model-related
    components, including the registry, services, and configuration. It follows
    the dependency injection pattern to eliminate global state and improve
    testability.

    Lazy initialization is used to defer expensive operations until needed.

    Thread safety is ensured through proper locking mechanisms.

    Attributes:
        _config (ModelConfig): Configuration for the model system.
        _registry (ModelRegistry): Registry for model management.
        _model_service (ModelService): Service for model invocation.
        _usage_service (UsageService): Service for tracking usage.
        _initialized (bool): Whether the context has been initialized.
        _lock (threading.RLock): Lock for thread safety.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """Initialize the model context.

        Args:
            config (Optional[ModelConfig]): Configuration for the model system.
                If None, default configuration is used.
        """
        self._config = config or ModelConfig()
        self._registry: Optional[ModelRegistry] = None
        self._model_service: Optional[ModelService] = None
        self._usage_service: Optional[UsageService] = None
        self._initialized = False
        self._lock = threading.RLock()  # Use RLock to allow re-entrant locking

    def initialize(self) -> None:
        """Initialize all components of the context.

        This method initializes the registry, model service, and usage service
        according to the configuration. It is thread-safe and idempotent.
        """
        with self._lock:
            if self._initialized:
                return

            # Initialize the registry
            self._registry = initialize_registry(
                config_path=self._config.config_path,
                config_manager=self._config.config_manager,
                auto_discover=self._config.auto_discover)

            # Initialize the usage service
            self._usage_service = UsageService()

            # Initialize the model service with the registry and usage service
            service_class = self._config.service_class
            self._model_service = service_class(
                registry=self._registry, usage_service=self._usage_service
            )

            self._initialized = True

    @property
    def registry(self) -> ModelRegistry:
        """Get the model registry.

        Returns:
            ModelRegistry: The model registry instance.
        """
        if not self._initialized:
            self.initialize()
        return self._registry

    @property
    def model_service(self) -> ModelService:
        """Get the model service.

        Returns:
            ModelService: The model service instance.
        """
        if not self._initialized:
            self.initialize()
        return self._model_service

    @property
    def usage_service(self) -> UsageService:
        """Get the usage service.

        Returns:
            UsageService: The usage service instance.
        """
        if not self._initialized:
            self.initialize()
        return self._usage_service

    @property
    def config(self) -> ModelConfig:
        """Get the model configuration.

        Returns:
            ModelConfig: The model configuration instance.
        """
        return self._config


# Default global context for backward compatibility
_default_context = None
_default_context_lock = threading.Lock()


def get_default_context() -> ModelContext:
    """Get the default model context singleton.

    This function returns the default model context instance,
    creating it if necessary. This provides backward compatibility
    with code that doesn't explicitly manage context.

    Returns:
        ModelContext: The default model context instance.
    """
    global _default_context

    # Fast path if already initialized
    if _default_context is not None:
        return _default_context

    with _default_context_lock:
        # Check again in case another thread initialized it
        if _default_context is None:
            _default_context = ModelContext()

    return _default_context


def create_context(config: Optional[ModelConfig] = None) -> ModelContext:
    """Create a new model context with the specified configuration.

    This factory function creates a new model context with the
    specified configuration.

    Args:
        config (Optional[ModelConfig]): Configuration for the model system.
            If None, default configuration is used.

    Returns:
        ModelContext: A new model context instance.
    """
    return ModelContext(config=config)
