"""Models API for Ember.

Clean, intuitive interface for interacting with language models from various providers.
Supports multiple invocation patterns with explicit dependency management.

Examples:
    # Function-based pattern (recommended)
    from ember.api import models
    response = models.model("gpt-4o")("What is the capital of France?")

    # Provider namespaces
    response = models.openai.gpt4o("What is the capital of France?")

    # Reusable models with configuration
    gpt4 = models.model("gpt-4o", temperature=0.7)
    response = gpt4("Tell me a joke")

    # With configuration context manager
    with models.configure(temperature=0.2, max_tokens=100):
        response = models.model("gpt-4o")("Write a haiku")

    # Type-safe ModelEnum references
    from ember.api.models import ModelEnum
    response = models.from_enum(ModelEnum.OPENAI_GPT4O)("Hello")
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from difflib import get_close_matches
from typing import Callable, ClassVar, Dict, List, Optional, Protocol, TypeVar, Union, cast, overload

from ember.core.registry.model.base.context import ModelConfig as ContextConfig
from ember.core.registry.model.base.context import (
    ModelContext,
    create_context,
    get_default_context,
)
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.config.model_enum import ModelEnum
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.types.ember_model import EmberModel

# Define type for raw LLM responses
ResponseType = Union[ChatResponse, Dict[str, str], str]

logger = logging.getLogger(__name__)


# Global configuration state with thread safety
class ModelConfig:
    """Global configuration for model behavior."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelConfig, cls).__new__(cls)
            cls._instance.temperature = 0.7
            cls._instance.max_tokens = None
            cls._instance.timeout = None
            cls._instance.top_p = 1.0
            cls._instance.top_k = None
            cls._instance.stop_sequences = None
            cls._instance.thread_local_overrides = {}
        return cls._instance

    def update(self, **kwargs):
        """Update global configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration option: {key}")
        return self

    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration, including thread-local overrides."""
        import threading

        thread_id = threading.get_ident()
        base_config = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": self.stop_sequences,
        }

        # Apply thread-local overrides if they exist
        overrides = self._instance.thread_local_overrides.get(thread_id, {})
        return {**base_config, **overrides}

    def set_thread_local_override(self, **kwargs):
        """Set thread-local configuration overrides."""
        import threading

        thread_id = threading.get_ident()
        self._instance.thread_local_overrides[thread_id] = kwargs

    def clear_thread_local_override(self):
        """Clear thread-local configuration overrides."""
        import threading

        thread_id = threading.get_ident()
        if thread_id in self._instance.thread_local_overrides:
            del self._instance.thread_local_overrides[thread_id]


# Context manager for temporary configuration
@contextmanager
def configure(**kwargs):
    """Temporarily override model configuration.

    Args:
        **kwargs: Configuration parameters to override.

    Examples:
        ```python
        with configure(temperature=0.2, max_tokens=100):
            response = model("gpt-4o")("Write a haiku")
        ```
    """
    config = ModelConfig()
    config.set_thread_local_override(**kwargs)
    try:
        yield
    finally:
        config.clear_thread_local_override()


class Response:
    """A clean, consistent response object with enhanced functionality.

    This class wraps a raw ChatResponse and provides a more intuitive interface.

    Examples:
        ```python
        response = model("gpt-4o")("What is the capital of France?")

        # Use as a string
        print(response)  # "Paris is the capital of France."

        # Access metadata
        print(f"Tokens used: {response.usage.total_tokens}")
        print(f"Cost: ${response.usage.cost:.6f}")
        ```
    """

    def __init__(self, raw_response: Any):
        """Initialize the response object.

        Args:
            raw_response: The raw response from the model.
        """
        self._raw = raw_response

        # Extract the primary content as text
        if hasattr(raw_response, "data"):
            self.text = raw_response.data
        elif hasattr(raw_response, "content"):
            self.text = raw_response.content
        else:
            self.text = str(raw_response)

        # Extract usage information if available
        self.usage = raw_response.usage if hasattr(raw_response, "usage") else None

        # Extract other metadata
        self.messages = self._extract_messages(raw_response)
        self.model_id = getattr(raw_response, "model_id", None)
        self.finish_reason = getattr(raw_response, "finish_reason", None)

    def __str__(self) -> str:
        """Return the response text when used as a string."""
        return self.text

    def _extract_messages(self, raw_response: Any) -> List[Dict[str, Any]]:
        """Extract conversation messages if present."""
        if hasattr(raw_response, "messages"):
            return raw_response.messages
        elif hasattr(raw_response, "choices") and hasattr(
            raw_response.choices[0], "message"
        ):
            return [choice.message for choice in raw_response.choices]
        else:
            return []

    def visualize(self) -> None:
        """Display a visual representation of this response.
        
        Uses rich formatting in Jupyter notebooks or falls back to logging.
        """
        try:
            import IPython.display
            from rich.console import Console
            from rich.panel import Panel

            console = Console()

            # Create a string representation for display
            display_text = f"[bold]Response:[/bold]\n{self.text}\n\n"

            if self.usage:
                display_text += "[bold]Usage:[/bold]\n"
                display_text += f"Total tokens: {self.usage.total_tokens}\n"
                display_text += f"Input tokens: {self.usage.prompt_tokens}\n"
                display_text += f"Output tokens: {self.usage.completion_tokens}\n"
                if hasattr(self.usage, "cost") and self.usage.cost:
                    display_text += f"Cost: ${self.usage.cost:.6f}\n"

            panel = Panel(display_text, title=f"Model: {self.model_id or 'Unknown'}")

            # Check if we're in a notebook
            try:
                import IPython
                IPython.get_ipython()
                IPython.display.display(
                    IPython.display.HTML(console.export_html(panel))
                )
            except (ImportError, AttributeError):
                # Fallback to console output
                console.print(panel)

        except ImportError:
            # Fallback to simple logging
            logger.info("Response: %s", self.text)
            if self.usage:
                logger.info("Usage: %s tokens", self.usage.total_tokens)


class ModelCallable:
    """A callable model with a clean, fluent interface.

    This class represents a model that can be called with a prompt to generate a response.
    It handles model resolution, configuration, and invocation.

    Examples:
        ```python
        # Create a model
        gpt4 = ModelCallable("gpt-4o", temperature=0.7)

        # Call the model
        response = gpt4("What is the capital of France?")
        ```
    """

    def __init__(self, model_id: str, context: Optional[ModelContext] = None, **config):
        """Initialize the model callable.

        Args:
            model_id: The model identifier (with or without provider prefix)
            context: Optional model context for dependency injection
            **config: Configuration options for the model
        """
        self._context = context or get_default_context()
        self.model_id = self._resolve_model_id(model_id)
        self.config = config

    def __call__(self, prompt: str, **kwargs: object) -> Response:
        """Call the model with a prompt.

        Args:
            prompt: The input prompt to generate from
            **kwargs: Additional overrides for this specific call

        Returns:
            A Response object containing the model's output
            
        Raises:
            ProviderConfigError: When API keys are missing
            ModelNotFoundError: When the specified model cannot be found
            ProviderAPIError: When the provider API call fails
            ModelError: For other model-related errors
        """
        from ember.core.exceptions import (
            ModelError, 
            ModelNotFoundError, 
            ProviderConfigError,
            ProviderAPIError
        )
        
        # Merge configurations in order of precedence
        global_config = ModelConfig().get_effective_config()
        merged_config = {**global_config, **self.config, **kwargs}

        # Access model service from context
        model_service = self._context.model_service

        # Invoke the model
        try:
            raw_response = model_service.invoke_model(
                model_id=self.model_id, prompt=prompt, **merged_config
            )
            return Response(raw_response)
        except Exception as e:
            # Handle missing API key errors
            if "API key" in str(e):
                provider = (
                    self.model_id.split(":")[0] if ":" in self.model_id else "provider"
                )
                recovery_hint = f"Please set the {provider.upper()}_API_KEY environment variable."
                
                raise ProviderConfigError(
                    f"Missing API key for {provider}.",
                    context={"provider": provider, "model_id": self.model_id},
                    recovery_hint=recovery_hint,
                    cause=e
                )
            # Handle model not found errors
            elif "not found" in str(e) or "invalid" in str(e).lower():
                logger.debug("Model %s not found. Looking for suggestions.", self.model_id)
                available_models = list(self._context.registry.list_models())
                suggestion = self._find_closest_model(self.model_id, available_models)
                
                # Create more helpful context with suggestion
                context = {
                    "model_id": self.model_id,
                    "available_models": available_models[:5],
                }
                if suggestion:
                    context["suggestion"] = suggestion
                
                raise ModelNotFoundError.for_model(
                    model_name=self.model_id,
                    provider_name=self.model_id.split(":")[0] if ":" in self.model_id else None
                )
            # Handle provider API errors
            elif hasattr(e, "status_code") or "quota" in str(e).lower() or "rate" in str(e).lower():
                provider = self.model_id.split(":")[0] if ":" in self.model_id else "unknown"
                status_code = getattr(e, "status_code", None)
                
                raise ProviderAPIError.for_provider(
                    provider_name=provider,
                    message=f"Error calling {provider} API: {str(e)}",
                    status_code=status_code,
                    cause=e
                )
            else:
                # Re-raise as a ModelError for other issues
                logger.error("Error invoking model %s: %s", self.model_id, e)
                raise ModelError(
                    f"Error invoking model {self.model_id}: {str(e)}",
                    context={"model_id": self.model_id},
                    cause=e
                )

    def _resolve_model_id(self, model_id: str) -> str:
        """Convert a simple model name to a fully-qualified ID if needed.

        Args:
            model_id: The model identifier (with or without provider prefix)

        Returns:
            A fully-qualified model ID
        """
        # If already has provider prefix, return as is
        if ":" in model_id:
            return model_id

        # Try to guess provider based on model name
        registry = self._context.registry

        # Common model name prefixes mapped to their providers
        provider_hints = {
            "gpt": "openai",
            "claude": "anthropic",
            "gemini": "deepmind",
        }

        # Check for prefix matches
        for prefix, provider in provider_hints.items():
            if model_id.lower().startswith(prefix):
                candidate = f"{provider}:{model_id}"
                if registry.is_registered(candidate):
                    return candidate

        # Try all providers as fallback
        for provider in ["openai", "anthropic", "deepmind"]:
            candidate = f"{provider}:{model_id}"
            if registry.is_registered(candidate):
                return candidate

        # If we got here, just return as-is and let the registry handle errors
        return model_id

    def _find_closest_model(
        self, model_id: str, available_models: List[str]
    ) -> Optional[str]:
        """Find the closest matching model name for suggestions.

        Args:
            model_id: The requested model ID that wasn't found
            available_models: List of available model IDs

        Returns:
            The closest matching model ID, if any
        """
        try:
            from difflib import get_close_matches

            # Extract base name without provider
            base_name = model_id.split(":")[-1] if ":" in model_id else model_id

            # Look for close matches among all available models
            base_names = [m.split(":")[-1] for m in available_models]
            matches = get_close_matches(base_name, base_names, n=1, cutoff=0.6)

            if matches:
                # Find the full model ID for the matched base name
                for m in available_models:
                    if m.split(":")[-1] == matches[0]:
                        return m

            return None
        except Exception:
            # If anything goes wrong with this nice-to-have feature, just return None
            return None

    def info(self) -> None:
        """Display information about this model."""
        try:
            registry = self._context.registry
            model_info = registry.get_model_info(self.model_id)

            # Log basic info
            logger.info("Model: %s", self.model_id)
            if model_info:
                logger.info("Provider: %s", model_info.provider.name)
                if hasattr(model_info, "description") and model_info.description:
                    logger.info("Description: %s", model_info.description)

                # Log pricing if available
                if model_info.cost:
                    logger.info("Pricing:")
                    logger.info(
                        "  Input: $%.6f per 1K tokens",
                        model_info.cost.input_cost_per_thousand
                    )
                    logger.info(
                        "  Output: $%.6f per 1K tokens",
                        model_info.cost.output_cost_per_thousand
                    )

                # Log rate limits if available
                if model_info.rate_limit:
                    logger.info("Rate Limits:")
                    if model_info.rate_limit.tokens_per_minute:
                        logger.info(
                            "  Tokens: %s per minute",
                            model_info.rate_limit.tokens_per_minute
                        )
                    if model_info.rate_limit.requests_per_minute:
                        logger.info(
                            "  Requests: %s per minute",
                            model_info.rate_limit.requests_per_minute
                        )
        except Exception as e:
            logger.error("Could not retrieve model information: %s", e)


@overload
def model(model_id: str) -> ModelCallable:
    ...


@overload
def model(
    model_id: str, *, context: Optional[ModelContext] = None, **config
) -> ModelCallable:
    ...


def model(
    model_id: str, *, context: Optional[ModelContext] = None, **config
) -> ModelCallable:
    """Create a callable model instance.

    This is the primary entry point for the model API.

    Args:
        model_id: Name of the model (with or without provider prefix)
        context: Optional model context for dependency injection
        **config: Configuration options like temperature, max_tokens, etc.

    Returns:
        A callable model instance that can generate responses.

    Examples:
        >>> response = model("gpt-4o")("What is the capital of France?")
        >>> print(response)
        "Paris is the capital of France."

        >>> gpt4 = model("gpt-4o", temperature=0.7)
        >>> response = gpt4("Tell me a joke")
        >>> print(response)

        >>> # With custom context
        >>> custom_context = create_context(config=ContextConfig(api_keys={"openai": "my-key"}))
        >>> response = model("gpt-4o", context=custom_context)("Hello")
    """
    return ModelCallable(model_id, context=context, **config)


def complete(
    prompt: str, *, model: str, context: Optional[ModelContext] = None, **kwargs
) -> Response:
    """Complete a prompt using the specified model.

    This is a convenience function for one-off completions.

    Args:
        prompt: The input prompt
        model: The model to use
        context: Optional model context for dependency injection
        **kwargs: Additional parameters for the model

    Returns:
        A Response object containing the model's response

    Examples:
        >>> answer = complete("Explain quantum computing", model="gpt-4o", temperature=0.7)
        >>> print(answer)
    """
    return model(model, context=context, **kwargs)(prompt)


def from_enum(
    enum_value: ModelEnum, *, context: Optional[ModelContext] = None, **config
) -> ModelCallable:
    """Create a model instance from a ModelEnum value.

    This provides a type-safe way to create model instances using the enum.

    Args:
        enum_value: A ModelEnum value representing the model
        context: Optional model context for dependency injection
        **config: Configuration options

    Returns:
        A callable model instance

    Examples:
        >>> from ember.api.models import ModelEnum
        >>> model = from_enum(ModelEnum.OPENAI_GPT4O, temperature=0.7)
        >>> response = model("Hello, world!")
    """
    return model(enum_value.value, context=context, **config)


# Global configuration instance for easy access
config = ModelConfig()


# Create provider-specific entry points for advanced users who prefer them
def create_provider_namespace(
    provider_name: str, context: Optional[ModelContext] = None
) -> object:
    """Create a provider namespace for model access.

    This function creates a namespace that allows direct access to models
    from a specific provider.

    Args:
        provider_name: The name of the provider, e.g., "openai"
        context: Optional model context for dependency injection

    Returns:
        A namespace object with dynamic model access

    Examples:
        >>> openai = create_provider_namespace("openai", context=custom_context)
        >>> response = openai.gpt4("Hello")
    """
    ctx = context or get_default_context()

    class ProviderNamespace:
        """Provider namespace for accessing models by name."""

        def __init__(self, provider: str):
            self._provider = provider
            self._context = ctx

        def __getattr__(self, model_name: str) -> Callable[[str], Response]:
            """Get a callable for the specified model."""
            # Convert underscores to hyphens
            model_name_normalized = model_name.replace("_", "-")
            model_id = f"{self._provider}:{model_name_normalized}"

            # Create and return a callable model
            return ModelCallable(model_id, context=self._context)

    return ProviderNamespace(provider_name)


# Create provider namespaces using the default context
openai = create_provider_namespace("openai")
anthropic = create_provider_namespace("anthropic")
deepmind = create_provider_namespace("deepmind")

# Create top-level aliases for the most common models
gpt4 = model("openai:gpt-4")
gpt4o = model("openai:gpt-4o")
claude = model("anthropic:claude-3-5-sonnet")
gemini = model("deepmind:gemini-1.5-pro")


# Legacy API for backward compatibility
class ModelAPI:
    """High-level API for interacting with a specific model.

    This class provides backward compatibility with the old API style.
    For new code, use the function-based API (`model()`) instead.

    Examples:
        >>> model_api = ModelAPI(model_id="anthropic:claude-3-5-sonnet", context=custom_context)
        >>> response = model_api.generate("Explain quantum computing to me.")
    """

    def __init__(self, model_id: str, context: Optional[ModelContext] = None):
        """Initialize with explicit dependencies.

        Args:
            model_id: The model identifier
            context: Optional model context, uses default if not provided
        """
        self._context = context or get_default_context()
        self.model_id = model_id
        self._callable = ModelCallable(model_id, context=self._context)

    def generate(self, prompt: str, **kwargs: Any) -> Response:
        """Generate a response from the model.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model

        Returns:
            A Response object containing the model's response
        """
        return self._callable(prompt, **kwargs)

    @classmethod
    def from_enum(
        cls, enum_value: ModelEnum, context: Optional[ModelContext] = None
    ) -> "ModelAPI":
        """Create a ModelAPI instance from a ModelEnum value.

        Args:
            enum_value: A ModelEnum value representing the model
            context: Optional model context for dependency injection

        Returns:
            A ModelAPI instance

        Examples:
            >>> from ember.api.models import ModelEnum
            >>> model = ModelAPI.from_enum(ModelEnum.OPENAI_GPT4O)
            >>> response = model.generate("Hello, world!")
        """
        return cls(enum_value.value, context=context)


# Backward compatibility functions for existing code
def get_registry(context: Optional[ModelContext] = None) -> ModelRegistry:
    """Get the model registry from the specified context.

    Args:
        context: Optional model context, uses default if not provided

    Returns:
        The model registry
    """
    ctx = context or get_default_context()
    return ctx.registry


def get_model_service(context: Optional[ModelContext] = None) -> ModelService:
    """Get the model service from the specified context.

    Args:
        context: Optional model context, uses default if not provided

    Returns:
        The model service
    """
    ctx = context or get_default_context()
    return ctx.model_service


def get_usage_service(context: Optional[ModelContext] = None) -> UsageService:
    """Get the usage service from the specified context.

    Args:
        context: Optional model context, uses default if not provided

    Returns:
        The usage service
    """
    ctx = context or get_default_context()
    return ctx.usage_service


# Export builder pattern for backward compatibility
class ModelBuilder:
    """Builder pattern for configuring model parameters.

    This class provides backward compatibility with the old API style.
    For new code, use the function-based API with direct kwargs.

    Examples:
        >>> model = (
        >>>     ModelBuilder()
        >>>     .temperature(0.7)
        >>>     .max_tokens(100)
        >>>     .build("anthropic:claude-3-5-sonnet")
        >>> )
        >>> response = model.generate("Explain quantum computing")
    """

    def __init__(self, context: Optional[ModelContext] = None) -> None:
        """Initialize the model builder.

        Args:
            context: Optional model context for dependency injection
        """
        self._config: Dict[str, Any] = {}
        self._context = context or get_default_context()

    def temperature(self, value: float) -> "ModelBuilder":
        """Set the temperature parameter."""
        self._config["temperature"] = value
        return self

    def max_tokens(self, value: int) -> "ModelBuilder":
        """Set the maximum tokens parameter."""
        self._config["max_tokens"] = value
        return self

    def top_p(self, value: float) -> "ModelBuilder":
        """Set the top_p parameter."""
        self._config["top_p"] = value
        return self

    def timeout(self, value: int) -> "ModelBuilder":
        """Set the timeout parameter."""
        self._config["timeout"] = value
        return self

    def build(self, model_id: Union[str, ModelEnum]) -> ModelAPI:
        """Build a model API instance with the configured parameters."""
        # Convert enum to string if needed
        if hasattr(model_id, "value"):
            model_id = model_id.value

        # Create a ModelAPI with our context
        api = ModelAPI(model_id=str(model_id), context=self._context)

        # Apply configuration
        api._callable.config.update(self._config)

        return api


# Export public API
__all__ = [
    # New primary API
    "model",
    "complete",
    "configure",
    "from_enum",
    "Response",
    "create_provider_namespace",
    "config",
    # Context system
    "ModelContext",
    "ContextConfig",
    "create_context",
    "get_default_context",
    # Legacy API for backward compatibility
    "ModelAPI",
    "ModelBuilder",
    "get_registry",
    "get_model_service",
    "get_usage_service",
    # Provider namespaces
    "openai",
    "anthropic",
    "deepmind",
    # Model aliases
    "gpt4",
    "gpt4o",
    "claude",
    "gemini",
    # Core classes and types for advanced usage
    "ModelRegistry",
    "ModelService",
    "UsageService",
    "BaseProviderModel",
    "LMModule",
    "LMModuleConfig",
    # Model information and configuration
    "ModelInfo",
    "ProviderInfo",
    "ModelCost",
    "RateLimit",
    "UsageStats",
    # Request and response types
    "ChatRequest",
    "ChatResponse",
    # Constants and enums
    "ModelEnum",
]
