"""Provider registry with explicit mapping.

This module implements a principled approach to provider management that favors
explicit registration over magic discovery. The design follows the philosophy
that explicit is better than implicit.

Key Design Decisions:
    1. Static registry - Core providers are hardcoded, not discovered
    2. Runtime extension - Custom providers can be registered dynamically
    3. Clear precedence - Custom providers can override core providers
    4. Simple resolution - Model-to-provider mapping uses clear rules

The architecture eliminates the complexity of plugin discovery systems in
favor of a simple dictionary that any developer can understand at a glance.

Examples:
    Using core providers:
    
    >>> from ember.api import models
    >>> response = models("gpt-4", "Hello")
    >>> response = models("claude-3-opus", "Hello")
    
    Registering custom provider:
    
    >>> from ember.models.providers import register_provider
    >>> from my_company import ProprietaryProvider
    >>> 
    >>> register_provider("proprietary", ProprietaryProvider)
    >>> response = models("proprietary/custom-model", "Hello")
"""

from typing import Type, Dict, List, Optional

from ember.models.providers.base import BaseProvider

# Import core providers
from ember.models.providers.openai import OpenAIProvider
from ember.models.providers.anthropic import AnthropicProvider
from ember.models.providers.google import GoogleProvider


# Explicit mapping of provider names to their implementation classes.
# This is intentionally a simple dictionary rather than a complex registry.
# The philosophy: if you can't grep for it, it's too magical.
PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "deepmind": GoogleProvider,  # Alias for backward compatibility
}

# Custom providers can be registered at runtime.
# This separation allows extending without modifying core code.
_custom_providers: Dict[str, Type[BaseProvider]] = {}


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """Register a custom provider implementation.
    
    This function enables runtime extension of the provider system without
    modifying core code. It's designed for enterprise users who need to
    integrate proprietary models or custom endpoints.
    
    The registration is global and persists for the process lifetime.
    Custom providers take precedence over core providers with the same name.
    
    Args:
        name: Provider identifier. Should be lowercase, no spaces.
            Examples: "azure", "bedrock", "enterprise"
        provider_class: Class inheriting from BaseProvider.
            Must implement the complete provider interface.
        
    Raises:
        TypeError: If provider_class doesn't inherit from BaseProvider.
        ValueError: If name is empty.
        
    Examples:
        Basic registration:
        
        >>> from my_company import EnterpriseProvider
        >>> register_provider("enterprise", EnterpriseProvider)
        >>> response = models("enterprise/gpt-4", "Hello")
        
        Override core provider:
        
        >>> from my_azure import AzureOpenAIProvider  
        >>> register_provider("openai", AzureOpenAIProvider)
        >>> # Now all OpenAI calls route through Azure
        
        With validation:
        
        >>> class CustomProvider(BaseProvider):
        ...     def complete(self, prompt, model, **kwargs):
        ...         # Custom implementation
        ...         pass
        >>> register_provider("custom", CustomProvider)
    """
    if not name:
        raise ValueError("Provider name cannot be empty")
        
    if not isinstance(provider_class, type):
        raise TypeError("provider_class must be a class, not an instance")
        
    if not issubclass(provider_class, BaseProvider):
        raise TypeError(
            f"Provider class must inherit from BaseProvider, "
            f"got {provider_class.__name__}"
        )
    
    _custom_providers[name] = provider_class


def unregister_provider(name: str) -> bool:
    """Unregister a custom provider.
    
    Removes a previously registered custom provider. Core providers cannot
    be unregistered, ensuring system stability.
    
    Args:
        name: The provider name to unregister.
        
    Returns:
        True if provider was unregistered, False if not found or core.
        
    Examples:
        >>> register_provider("test", TestProvider)
        >>> success = unregister_provider("test")
        >>> print(success)  # True
        >>> 
        >>> # Core providers cannot be removed
        >>> success = unregister_provider("openai")
        >>> print(success)  # False
    """
    if name in _custom_providers:
        del _custom_providers[name]
        return True
    return False


def get_provider_class(name: str) -> Type[BaseProvider]:
    """Get provider implementation class by name.
    
    This is the primary lookup function used by the registry. It checks
    custom providers first, allowing overrides of core functionality.
    
    Args:
        name: Provider identifier (e.g., "openai", "anthropic", "custom").
        
    Returns:
        The provider implementation class (not instance).
        
    Raises:
        ValueError: If provider name is not found.
        
    Examples:
        Get and instantiate provider:
        
        >>> provider_class = get_provider_class("openai")
        >>> provider = provider_class(api_key="sk-...")
        >>> response = provider.complete("Hello", "gpt-4")
        
        Check provider type:
        
        >>> from ember.models.providers.openai import OpenAIProvider
        >>> provider_class = get_provider_class("openai")
        >>> print(provider_class is OpenAIProvider)  # True
    """
    # Check custom providers first (allows overriding)
    if name in _custom_providers:
        return _custom_providers[name]
    
    # Then check core providers
    if name in PROVIDERS:
        return PROVIDERS[name]
    
    # Provide helpful error message
    available = list_providers()
    raise ValueError(
        f"Unknown provider '{name}'. "
        f"Available providers: {', '.join(sorted(available))}"
    )


def list_providers() -> List[str]:
    """List all available provider names.
    
    Returns a sorted list of all registered providers, both core and custom.
    This is useful for discovery and validation.
    
    Returns:
        Sorted list of provider names.
        
    Examples:
        Default providers:
        
        >>> providers = list_providers()
        >>> print(providers)
        ['anthropic', 'deepmind', 'google', 'openai']
        
        With custom provider:
        
        >>> register_provider("azure", AzureProvider)
        >>> providers = list_providers()
        >>> print('azure' in providers)  # True
    """
    # Combine core and custom, with custom taking precedence
    all_providers = set(PROVIDERS.keys()) | set(_custom_providers.keys())
    return sorted(all_providers)


def is_provider_available(name: str) -> bool:
    """Check if a provider is available.
    
    Simple boolean check for provider existence. Useful for validation
    before attempting to use a provider.
    
    Args:
        name: The provider name to check.
        
    Returns:
        True if provider is registered, False otherwise.
        
    Examples:
        >>> print(is_provider_available("openai"))  # True
        >>> print(is_provider_available("nonexistent"))  # False
    """
    return name in _custom_providers or name in PROVIDERS


def get_provider_info(name: str) -> Dict[str, str]:
    """Get information about a provider.
    
    Returns metadata about a provider including its type (core/custom)
    and implementation details. Useful for debugging and introspection.
    
    Args:
        name: The provider name.
        
    Returns:
        Dictionary containing:
            - name: Provider identifier
            - type: "core" or "custom"
            - class: Implementation class name
            - module: Module path
        
    Raises:
        ValueError: If provider not found.
        
    Examples:
        >>> info = get_provider_info("openai")
        >>> print(info["type"])  # "core"
        >>> print(info["class"])  # "OpenAIProvider"
    """
    if name in _custom_providers:
        provider_class = _custom_providers[name]
        return {
            "name": name,
            "type": "custom",
            "class": provider_class.__name__,
            "module": provider_class.__module__
        }
    elif name in PROVIDERS:
        provider_class = PROVIDERS[name]
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
    
    This function implements the model resolution logic that makes Ember's
    API intuitive. It handles both explicit notation (provider/model) and
    implicit notation where the provider is inferred from model naming
    conventions.
    
    Resolution Rules:
        1. If "/" present: Split on first "/" as provider/model
        2. Otherwise, infer from prefixes:
           - gpt-*, davinci, etc. → openai
           - claude* → anthropic  
           - gemini* → google
           - Others → "unknown"
    
    Args:
        model: Model identifier in any supported format.
            Examples: "gpt-4", "openai/gpt-4", "claude-3-opus"
        
    Returns:
        Tuple of (provider_name, model_name).
        Provider is "unknown" if inference fails.
        
    Examples:
        Implicit resolution:
        
        >>> resolve_model_id("gpt-4")
        ('openai', 'gpt-4')
        
        >>> resolve_model_id("claude-3-opus")
        ('anthropic', 'claude-3-opus')
        
        >>> resolve_model_id("gemini-pro")
        ('google', 'gemini-pro')
        
        Explicit resolution:
        
        >>> resolve_model_id("anthropic/claude-3")
        ('anthropic', 'claude-3')
        
        >>> resolve_model_id("azure/gpt-4")
        ('azure', 'gpt-4')
        
        Unknown models:
        
        >>> resolve_model_id("llama-2-70b")
        ('unknown', 'llama-2-70b')
    """
    # Check for explicit provider notation
    if "/" in model:
        parts = model.split("/", 1)
        return parts[0], parts[1]
    
    # Infer provider from well-known model naming patterns.
    # This list is intentionally conservative - only models with
    # clear provider association are mapped.
    model_lower = model.lower()
    
    # OpenAI models - comprehensive list of known patterns
    if (model_lower.startswith("gpt-") or 
        model_lower.startswith("davinci") or
        model_lower.startswith("babbage") or
        model_lower.startswith("ada") or
        model_lower.startswith("text-") or
        model_lower.startswith("o1-") or  # New reasoning models
        model_lower in ["gpt-4o", "gpt-4o-mini"]):
        return "openai", model
    
    # Anthropic models - all Claude variants
    elif model_lower.startswith("claude"):
        return "anthropic", model
    
    # Google models - Gemini family
    elif model_lower.startswith("gemini") or model_lower.startswith("models/gemini"):
        return "google", model
    
    # Models without clear provider association return "unknown".
    # This includes Llama, Mistral, etc. which can be served by
    # multiple providers. The registry will provide a better error.
    return "unknown", model


# Export key functions
__all__ = [
    "register_provider",
    "unregister_provider", 
    "get_provider_class",
    "list_providers",
    "is_provider_available",
    "get_provider_info",
    "resolve_model_id",
    "PROVIDERS",
]