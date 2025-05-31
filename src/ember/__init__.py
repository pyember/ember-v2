"""Ember: Framework for Compound AI Systems.

Example:
    >>> import ember
    >>> from ember.api import models, data, non, xcs
    
    >>> # Direct model invocation
    >>> response = models("gpt-4", "What is the capital of France?")
    >>> print(response.text)
    
    >>> # Load and process data
    >>> dataset = data("mmlu", streaming=True, limit=100)
    
    >>> # Create operator ensemble
    >>> ensemble = non.UniformEnsemble(num_units=3, model_name="gpt-4")
    
    >>> # Optimize execution
    >>> @xcs.jit
    ... def process(x):
    ...     return ensemble(x)
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Early logging configuration to suppress HTTP library noise
import logging
import os

# Suppress HTTP library logs before they're even imported
_http_log_level = os.environ.get("EMBER_HTTP_LOG_LEVEL", "ERROR")
try:
    _http_level = getattr(logging, _http_log_level.upper())
except AttributeError:
    _http_level = logging.ERROR

# Set levels for HTTP libraries that might be imported early
for _lib in ["httpcore", "httpx", "urllib3", "openai", "anthropic", "requests"]:
    _logger = logging.getLogger(_lib)
    _logger.setLevel(_http_level)
    if not _logger.handlers:
        _logger.addHandler(logging.NullHandler())

# Forward reference for return type
ModelRegistryType = TypeVar("ModelRegistryType")

# Import primary API components - these are the only public interfaces
from ember.api import models  # Language model access (models.openai.gpt4, etc.)
from ember.api import non  # Network of Networks patterns (non.UniformEnsemble, etc.)
from ember.api import operators  # Operator registry (operators.get_operator(), etc.)
from ember.api import xcs  # Execution optimization (xcs.jit, etc.)

# Core imports are done lazily in functions to avoid circular dependencies
# This follows the thread-local design pattern established in the EmberContext system
# and maintains the separation between API and implementation layers

# Version detection
try:
    __version__ = importlib.metadata.version("ember-ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

# Package metadata
_PACKAGE_METADATA = {
    "name": "ember-ai",
    "version": __version__,
    "description": "Compositional framework for building and orchestrating "
    "Compound AI Systems and Networks of Networks (NONs).",
}


def initialize_ember(
    config_path: Optional[str] = None,
    auto_discover: bool = True,
    force_discovery: bool = False,
    api_keys: Optional[Dict[str, str]] = None,
    env_prefix: str = "EMBER_",
    verbose_logging: bool = False) -> ModelRegistryType:
    """Initialize Ember with configuration.

    Args:
        config_path: Configuration file path.
        auto_discover: Auto-discover models.
        force_discovery: Force re-discovery.
        api_keys: API keys by provider (e.g., {"openai": "sk-..."}).
        env_prefix: Environment variable prefix.
        verbose_logging: Enable debug logging.

    Returns:
        Initialized model registry.
    """
    # Import modules where needed to avoid circular dependencies
    from ember.core.config.manager import create_config_manager
    from ember.core.registry.model.initialization import initialize_registry
    from ember.core.utils.logging import configure_logging

    # 0. Configure logging first
    configure_logging(verbose=verbose_logging)

    # 1. Create the configuration manager with the provided config path
    config_manager = create_config_manager(config_path=config_path)

    # 2. Apply API keys if provided (highest precedence)
    if api_keys:
        for provider, api_key in api_keys.items():
            config_manager.set_provider_api_key(provider, api_key)

    # 3. Initialize the model registry
    registry = initialize_registry(
        config_manager=config_manager,
        auto_discover=auto_discover,
        force_discovery=force_discovery)

    # Context is initialized lazily via current_context()

    # Return the registry
    return registry


def init(
    config: Optional[Union[Dict[str, Any], Dict[str, Any]]] = None,
    usage_tracking: bool = False) -> Callable:
    """Initialize Ember with simplified API.

    Args:
        config: Configuration overrides.
        usage_tracking: Enable token/cost tracking.

    Returns:
        Callable model service.

    Example:
        >>> service = init()
        >>> response = service("gpt-4", "Explain quantum computing")
    """
    from ember.api.models import ModelService, UsageService
    from ember.core.config.manager import create_config_manager
    from ember.core.registry.model.initialization import initialize_registry

    # Initialize configuration if needed
    config_manager = None
    if isinstance(config, dict):
        config_manager = create_config_manager()
        for key, value in config.items():
            config_manager.set(key, value)
    elif config is not None and hasattr(config, "set") and hasattr(config, "get"):
        config_manager = config

    # Initialize the registry with auto-discovery
    registry = initialize_registry(auto_discover=True, config_manager=config_manager)

    # Create usage service if tracking is enabled
    usage_service = UsageService() if usage_tracking else None

    # Create a model service that can be called directly
    service = ModelService(registry=registry, usage_service=usage_service)

    # Create a wrapper function that allows direct calling with model ID and prompt
    def service_wrapper(model_id_or_enum, prompt, **kwargs):
        return service.invoke_model(model_id_or_enum, prompt, **kwargs)

    # Add service attributes to the wrapper
    service_wrapper.model_service = service
    service_wrapper.registry = registry
    if usage_service:
        service_wrapper.usage_service = usage_service

    return service_wrapper


# Public interface - only export the main API components
__all__ = [
    "models",  # Language model access
    "operators",  # Operator registry
    "non",  # Network of Networks patterns
    "xcs",  # Execution optimization
    "initialize_ember",  # Global initialization function
    "init",  # Simple initialization function (matches README examples)
    "configure_logging",  # Logging configuration utility
    "set_component_level",  # Fine-grained logging control
    "__version__"]
