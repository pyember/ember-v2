"""Model registry for language model providers.

Central registry for discovering, loading, and managing language models
from various providers (OpenAI, Anthropic, etc).

Example:
    >>> from ember.core.registry.model import ModelRegistry
    >>> registry = ModelRegistry()
    >>> model = registry.get_model("gpt-4")
    >>> response = model.generate("What is AI?")
"""

from __future__ import annotations

from typing import List

# Configuration and initialization - import from core config to avoid circular imports
from ember.core.config.schema import EmberSettings

# Import submodules
from ember.core.registry.model import examples
from ember.core.registry.model.base.registry.factory import ModelFactory

# Registry components
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse)
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit

# Absolute imports for core schemas
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.usage import (
    UsageRecord,
    UsageStats,
    UsageSummary)

# Services
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService

# Absolute imports for exceptions
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ModelDiscoveryError,
    ModelRegistrationError)
from ember.core.registry.model.config.model_enum import ModelEnum, parse_model_str

# Base provider classes
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel)

# These are already imported above using absolute imports, so we can remove these relative imports


# Add load_model function
def load_model(model_id: str, registry: ModelRegistry) -> BaseProviderModel:
    """Load a model instance from the registry.

    Args:
        model_id: Model identifier (e.g., "gpt-4", "claude-3")
        registry: ModelRegistry instance to query

    Returns:
        Instantiated provider model ready for use
    """
    return registry.get_model(model_id)


__all__: List[str] = [
    # Schemas
    "ModelInfo",
    "ProviderInfo",
    "ModelCost",
    "RateLimit",
    "ChatRequest",
    "ChatResponse",
    "UsageStats",
    "UsageRecord",
    "UsageSummary",
    # Base classes
    "BaseChatParameters",
    "BaseProviderModel",
    # Registry
    "ModelRegistry",
    "ModelFactory",
    "ModelEnum",
    "parse_model_str",
    # Services
    "ModelService",
    "UsageService",
    # Settings
    "EmberSettings",
    "initialize_ember",
    "ModelRegistrationError",
    "ModelDiscoveryError",
    "load_model",
    # Submodules
    "examples"]


# Initialization function - defined here to avoid circular imports
def initialize_ember(
    config_path: str | None = None,
    auto_register: bool = True,
    auto_discover: bool = True,
    force_discovery: bool = False) -> ModelRegistry:
    """Initialize the Ember model registry.

    DEPRECATED: Use initialize_registry from ember.core.registry.model.initialization.

    Args:
        config_path: Optional path to config file
        auto_register: Automatically register models from config
        auto_discover: Enable provider model discovery
        force_discovery: Force discovery even if auto_discover is False

    Returns:
        Initialized ModelRegistry instance
    """
    import warnings

    warnings.warn(
        "initialize_ember() is deprecated. Use initialize_registry() from "
        "ember.core.registry.model.initialization instead.",
        DeprecationWarning,
        stacklevel=2)

    from ember.core.config.manager import create_config_manager
    from ember.core.registry.model.initialization import initialize_registry

    config_manager = create_config_manager(config_path=config_path)
    return initialize_registry(
        config_manager=config_manager,
        auto_discover=auto_discover,
        force_discovery=force_discovery)
