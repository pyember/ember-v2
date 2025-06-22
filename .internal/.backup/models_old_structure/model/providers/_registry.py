"""Simplified provider registry with explicit mapping.

This module provides a clean, explicit mapping of provider names to their
implementation classes, replacing the complex dynamic discovery system.

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

from typing import Type, Dict, List, Optional

from ember.core.registry.model.providers.base_provider import BaseProviderModel

# Import core providers
from ember.core.registry.model.providers.openai.openai_provider import OpenAIModel
from ember.core.registry.model.providers.anthropic.anthropic_provider import AnthropicModel
from ember.core.registry.model.providers.deepmind.deepmind_provider import GeminiModel


# Explicit mapping of provider names to their implementation classes
# This replaces the complex filesystem scanning with clear, explicit dependencies
CORE_PROVIDERS: Dict[str, Type[BaseProviderModel]] = {
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
    "deepmind": GeminiModel,
    "google": GeminiModel,  # Alias for backward compatibility
}

# Custom providers can be registered at runtime
# This allows advanced users to add their own providers without modifying core
_custom_providers: Dict[str, Type[BaseProviderModel]] = {}


def register_provider(name: str, provider_class: Type[BaseProviderModel]) -> None:
    """Register a custom provider implementation.
    
    This function allows advanced users to register their own provider
    implementations without modifying the core codebase.
    
    Args:
        name: The provider name (e.g., "custom", "enterprise").
        provider_class: The provider implementation class.
        
    Raises:
        TypeError: If provider_class doesn't inherit from BaseProviderModel.
        ValueError: If name is empty or already registered as custom.
        
    Examples:
        >>> from my_company import EnterpriseProvider
        >>> register_provider("enterprise", EnterpriseProvider)
        >>> 
        >>> # Now can use: models("enterprise/gpt-4", "Hello")
        
    Note:
        Custom providers can override core providers. This is intentional
        to allow testing and enterprise customization.
    """
    if not name:
        raise ValueError("Provider name cannot be empty")
        
    if not isinstance(provider_class, type):
        raise TypeError("provider_class must be a class, not an instance")
        
    if not issubclass(provider_class, BaseProviderModel):
        raise TypeError(
            f"Provider class must inherit from BaseProviderModel, "
            f"got {provider_class.__name__}"
        )
    
    _custom_providers[name] = provider_class


def unregister_provider(name: str) -> bool:
    """Unregister a custom provider.
    
    Args:
        name: The provider name to unregister.
        
    Returns:
        True if provider was unregistered, False if not found.
        
    Note:
        Only custom providers can be unregistered. Core providers
        cannot be removed.
        
    Examples:
        >>> unregister_provider("enterprise")
        True
        >>> unregister_provider("openai")  # Core provider
        False
    """
    if name in _custom_providers:
        del _custom_providers[name]
        return True
    return False


def get_provider_class(name: str) -> Type[BaseProviderModel]:
    """Get provider implementation class by name.
    
    Args:
        name: The provider name (e.g., "openai", "anthropic").
        
    Returns:
        The provider implementation class.
        
    Raises:
        ValueError: If provider name is not found.
        
    Examples:
        >>> provider_class = get_provider_class("openai")
        >>> provider = provider_class(api_key="...")
        
    Note:
        Custom providers take precedence over core providers,
        allowing override for testing or customization.
    """
    # Check custom providers first (allows overriding)
    if name in _custom_providers:
        return _custom_providers[name]
    
    # Then check core providers
    if name in CORE_PROVIDERS:
        return CORE_PROVIDERS[name]
    
    # Provide helpful error message
    available = list_providers()
    raise ValueError(
        f"Unknown provider '{name}'. "
        f"Available providers: {', '.join(sorted(available))}"
    )


def list_providers() -> List[str]:
    """List all available provider names.
    
    Returns:
        List of provider names, with custom providers marked.
        
    Examples:
        >>> providers = list_providers()
        >>> print(providers)
        ['anthropic', 'google', 'openai']
    """
    # Combine core and custom, with custom taking precedence
    all_providers = set(CORE_PROVIDERS.keys()) | set(_custom_providers.keys())
    return sorted(all_providers)


def is_provider_available(name: str) -> bool:
    """Check if a provider is available.
    
    Args:
        name: The provider name to check.
        
    Returns:
        True if provider is available, False otherwise.
        
    Examples:
        >>> is_provider_available("openai")
        True
        >>> is_provider_available("nonexistent")
        False
    """
    return name in _custom_providers or name in CORE_PROVIDERS


def get_provider_info(name: str) -> Dict[str, str]:
    """Get information about a provider.
    
    Args:
        name: The provider name.
        
    Returns:
        Dictionary with provider information.
        
    Raises:
        ValueError: If provider not found.
        
    Examples:
        >>> info = get_provider_info("openai")
        >>> print(info["type"])  # "core"
    """
    if name in _custom_providers:
        provider_class = _custom_providers[name]
        return {
            "name": name,
            "type": "custom",
            "class": provider_class.__name__,
            "module": provider_class.__module__
        }
    elif name in CORE_PROVIDERS:
        provider_class = CORE_PROVIDERS[name]
        return {
            "name": name,
            "type": "core",
            "class": provider_class.__name__,
            "module": provider_class.__module__
        }
    else:
        raise ValueError(f"Provider '{name}' not found")


def resolve_model_id(model: str) -> tuple[str, str]:
    """Resolve a model string to (provider, model_name).
    
    This function handles both explicit provider notation (provider/model)
    and implicit notation where the provider is inferred from the model name.
    
    Args:
        model: Model identifier (e.g., "gpt-4", "openai/gpt-4", "claude-3").
        
    Returns:
        Tuple of (provider_name, model_name).
        
    Examples:
        >>> resolve_model_id("gpt-4")
        ('openai', 'gpt-4')
        
        >>> resolve_model_id("anthropic/claude-3")
        ('anthropic', 'claude-3')
        
        >>> resolve_model_id("claude-3-opus")
        ('anthropic', 'claude-3-opus')
        
    Note:
        For unknown models without explicit provider, returns
        ('unknown', model) to let the registry handle the error
        with proper context.
    """
    # Check for explicit provider notation
    if "/" in model:
        parts = model.split("/", 1)
        return parts[0], parts[1]
    
    # Well-known model prefixes
    model_lower = model.lower()
    
    # OpenAI models
    if (model_lower.startswith("gpt-") or 
        model_lower.startswith("davinci") or
        model_lower.startswith("babbage") or
        model_lower.startswith("ada") or
        model_lower.startswith("text-") or
        model_lower in ["gpt-4o", "gpt-4o-mini"]):
        return "openai", model
    
    # Anthropic models
    elif model_lower.startswith("claude"):
        return "anthropic", model
    
    # Google models
    elif model_lower.startswith("gemini"):
        return "google", model
    
    # Meta/Llama models (could be served by various providers)
    elif model_lower.startswith("llama"):
        # Default to a common provider or return unknown
        return "unknown", model
    
    # Unknown model - let registry handle with full context
    return "unknown", model