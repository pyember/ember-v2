"""
Ember: Compositional Framework for Compound AI Systems
=====================================================

Ember is a powerful, extensible Python framework for building and orchestrating
Compound AI Systems and "Networks of Networks" (NONs).

Core Features:
- Eager Execution by Default
- Parallel Graph Execution
- Composable Operators
- Extensible Registry System
- Enhanced JIT System
- Built-in Evaluation
- Powerful Data Handling
- Intuitive Model Access

For more information, visit https://pyember.org

Examples:
    # Import primary API modules
    import ember

    # Initialize model registry and service
    from ember.api.models import initialize_registry, create_model_service

    registry = initialize_registry(auto_discover=True)
    model_service = create_model_service(registry=registry)

    # Call a model
    response = model_service.invoke_model(
        model_id="openai:gpt-4",
        prompt="What's the capital of France?",
        temperature=0.7
    )

    # Load datasets directly
    from ember.api.data import data
    mmlu_data = data("mmlu")

    # Or use the dataset builder pattern
    from ember.api.data import DatasetBuilder
    dataset = DatasetBuilder().split("test").sample(100).build("mmlu")

    # Create Networks of Networks (NONs)
    from ember.api import non
    ensemble = non.UniformEnsemble(
        num_units=3,
        model_name="openai:gpt-4o"
    )

    # Optimize with XCS
    from ember.api import xcs
    @xcs.jit
    def optimized_fn(x):
        return complex_computation(x)
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Callable, Dict, Optional, TypeVar, Union

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
    verbose_logging: bool = False,
) -> ModelRegistryType:
    """Initialize core Ember components.

    This function configures logging, API keys, and the model registry.
    The global Ember context is initialized lazily on first use via
    `ember.core.context.current_context()`.

    Args:
        config_path: Path to the main configuration file.
        auto_discover: Automatically discover models and plugins.
        force_discovery: Force re-discovery even if cache exists.
        api_keys: Dictionary of API keys (e.g., {"openai": "sk-..."}).
        env_prefix: Prefix for environment variables (e.g., EMBER_OPENAI_API_KEY).
        verbose_logging: Enable verbose debug logging.

    Returns:
        The initialized ModelRegistry.
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
        force_discovery=force_discovery,
    )

    # Context is initialized lazily via current_context()

    # Return the registry
    return registry


def init(
    config: Optional[Union[Dict[str, Any], Dict[str, Any]]] = None,
    usage_tracking: bool = False,
) -> Callable:
    """Initialize Ember and return a unified model service.

    This function provides a simple entry point for initializing Ember and accessing
    models directly through a callable service object, as shown in the README examples.

    Args:
        config: Optional configuration to override defaults. Can be a dictionary or
            a ConfigManager
        usage_tracking: Whether to enable cost/token tracking

    Returns:
        A model service that can be called directly with models and prompts

    Examples:
        # Simple usage
        service = init()
        response = service("openai:gpt-4o", "What is the capital of France?")

        # With usage tracking
        service = init(usage_tracking=True)
        response = service(models.ModelEnum.gpt_4o, "What is quantum computing?")
        usage = service.usage_service.get_total_usage()
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
    "__version__",
]
