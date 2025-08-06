"""Models API for language model interactions.

Provides direct function-based access to language models without requiring
client initialization. Simply import and call.

Basic usage:
    >>> from ember.api import models
    >>> response = models("gpt-4", "Hello world")
    >>> print(response.text)
    Hello! How can I help you today?
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ember._internal.exceptions import (
    ModelError,
    ModelNotFoundError,
    ProviderAPIError,
)
from ember.models import ModelRegistry
from ember.models.catalog import (
    Models,
    get_model_info,
    get_providers,
    list_available_models,
)
from ember.models.schemas import ChatResponse

logger = logging.getLogger(__name__)


class Response:
    """Response from a language model.
    
    Attributes:
        text: Generated text content.
        usage: Token usage and cost statistics.
        model_id: Model that generated the response.
        
    Example:
        >>> response = models("gpt-4", "What is 2+2?")
        >>> print(response.text)
        The answer is 4.
        >>> print(response.usage)
        {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15, 'cost': 0.0006}
    """
    
    def __init__(self, raw_response: ChatResponse):
        """Initialize Response.
        
        Args:
            raw_response: Raw response from the model registry.
        """
        self._raw = raw_response
    
    @property
    def text(self) -> str:
        """Get the generated text content.
        
        Returns:
            The model's text response.
            
        Examples:
            >>> response = models("gpt-4", "Hello")
            >>> print(response.text)
            Hello! How can I help you today?
        """
        return self._raw.data if self._raw.data else ""
    
    @property
    def usage(self) -> Dict[str, Any]:
        """Get token usage and cost information.
        
        Returns:
            Dictionary containing:
                - prompt_tokens: Number of input tokens
                - completion_tokens: Number of output tokens
                - total_tokens: Total tokens used
                - cost: Total cost in USD
                
        Examples:
            >>> response = models("gpt-4", "Write a haiku")
            >>> print(f"Cost: ${response.usage['cost']:.4f}")
            Cost: $0.0015
        """
        if not self._raw.usage:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0
            }
        
        return {
            "prompt_tokens": self._raw.usage.prompt_tokens,
            "completion_tokens": self._raw.usage.completion_tokens,
            "total_tokens": self._raw.usage.total_tokens,
            "cost": getattr(self._raw.usage, "cost_usd", 0.0)
        }
    
    @property
    def model_id(self) -> Optional[str]:
        """Get the ID of the model that generated this response.
        
        Returns:
            Model identifier or None if not available.
            
        Examples:
            >>> response = models("gpt-4", "Hello")
            >>> print(response.model_id)
            gpt-4
        """
        if hasattr(self._raw, "model_id"):
            return self._raw.model_id
        if hasattr(self._raw, "raw_output") and self._raw.raw_output is not None and hasattr(self._raw.raw_output, "model"):
            return self._raw.raw_output.model
        return None
    
    def __str__(self) -> str:
        """Return the generated text."""
        return self.text
    
    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"Response(text='{self.text[:50]}...', tokens={self.usage['total_tokens']})"


class ModelBinding:
    """Reusable model configuration.
    
    Binds a model with preset parameters for efficient reuse.
    
    Example:
        >>> creative = models.instance("gpt-4", temperature=0.9)
        >>> story1 = creative("Write about dragons")
        >>> story2 = creative("Write about robots")
    """
    
    def __init__(self, model_id: str, registry: ModelRegistry, **params):
        """Initialize ModelBinding.
        
        Args:
            model_id: Model identifier (e.g., "gpt-4", "claude-3").
            registry: Model registry for invocations.
            **params: Parameters to bind (temperature, max_tokens, etc.).
        """
        self.model_id = model_id
        self.registry = registry
        self.params = params
        # self._validate_model_id()
    
    def _validate_model_id(self) -> None:
        """Validate that the model exists.
        
        Raises:
            ModelNotFoundError: If model ID is not found.
        """
        try:
            self.registry.get_model(model_id=self.model_id)
        except Exception as e:
            if "not found" in str(e).lower():
                raise ModelNotFoundError(
                    f"Model '{self.model_id}' not found",
                    context={"model_id": self.model_id}
                )
            raise
    
    def __call__(self, prompt: str, **override_params) -> Response:
        """Invoke the bound model.
        
        Args:
            prompt: The prompt to send to the model.
            **override_params: Parameters that override the bound parameters.
            
        Returns:
            Response object with generated text and metadata.
            
        Examples:
            >>> gpt4 = models.instance("gpt-4", temperature=0.7)
            >>> response = gpt4("Hello")
            >>> print(response.text)
        """
        merged_params = {**self.params, **override_params}
        raw_response = self.registry.invoke_model(self.model_id, prompt, **merged_params)
        return Response(raw_response)
    
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"ModelBinding(model_id='{self.model_id}', params={self.params})"


class ModelsAPI:
    """Main interface for language models.
    
    Direct function-based API without client initialization.
    
    Example:
        >>> from ember.api import models
        >>> 
        >>> # Direct call
        >>> response = models("gpt-4", "Hello world")
        >>> 
        >>> # With parameters
        >>> response = models("gpt-4", "Be creative", temperature=0.9)
    """
    
    def __init__(self):
        """Initialize ModelsAPI with context-aware registry."""
        from ember._internal.context import EmberContext
        
        # Get or create context
        self._context = EmberContext.current()
        
        # Get registry from context (lazy initialization)
        self._registry = self._context.model_registry
    
    def __call__(self, model: str, prompt: str, **params) -> Response:
        """Call a model.
        
        Args:
            model: Model ID (e.g., "gpt-4", "claude-3-opus").
            prompt: Text prompt.
            **params: Optional parameters (temperature, max_tokens, etc.).
            
        Returns:
            Response with text and usage.
            
        Raises:
            ModelNotFoundError: Model not found.
            ModelProviderError: Missing API key.
            ProviderAPIError: API errors.
        """
        raw_response = self._registry.invoke_model(model, prompt, **params)
        return Response(raw_response)
    
    def instance(self, model: str, **params) -> ModelBinding:
        """Create reusable model binding.
        
        Args:
            model: Model ID.
            **params: Default parameters.
            
        Returns:
            Callable ModelBinding.
            
        Example:
            >>> creative = models.instance("gpt-4", temperature=0.9)
            >>> story = creative("Write a story")
        """
        return ModelBinding(model, self._registry, **params)
    
    def list(self) -> List[str]:
        """List all available models.
        
        Returns:
            List of all available model IDs.
            
        Examples:
            >>> models_list = models.list()
            >>> print(models_list)
            ['gpt-4', 'gpt-4-turbo', 'claude-3-opus', ...]
        """
        return list_available_models()
    
    def discover(self, provider: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Discover available models with detailed information.
        
        Args:
            provider: Optional provider filter (e.g., "openai", "anthropic")
            
        Returns:
            Dictionary mapping model IDs to their information.
            
        Examples:
            >>> # Show all models
            >>> info = models.discover()
            >>> for model_id, details in info.items():
            ...     print(f"{model_id}: {details['description']}")
            
            >>> # Show only OpenAI models
            >>> openai_models = models.discover("openai")
        """
        model_ids = list_available_models(provider)
        return {
            model_id: {
                "provider": get_model_info(model_id).provider,
                "description": get_model_info(model_id).description,
                "context_window": get_model_info(model_id).context_window,
            }
            for model_id in model_ids
        }
    
    def providers(self) -> List[str]:
        """List all available providers.
        
        Returns:
            List of provider names.
            
        Examples:
            >>> providers = models.providers()
            >>> print(providers)
            ['openai', 'anthropic', 'google']
        """
        return sorted(get_providers())
    
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return "ModelsAPI(simplified=True, direct_init=True)"


# Create global instance
_global_models_api = ModelsAPI()


def models(model: str, prompt: str, **params) -> Response:
    """Invoke a language model directly.
    
    This is the simplest way to use Ember - just import and call.
    
    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3").
        prompt: The prompt to send to the model.
        **params: Optional parameters like temperature, max_tokens.
        
    Returns:
        Response object with text and usage information.
        
    Examples:
        >>> from ember.api import models
        >>> response = models("gpt-4", "Hello world")
        >>> print(response.text)
    """
    return _global_models_api(model, prompt, **params)


# Expose the instance methods for creating bindings
models.instance = _global_models_api.instance
models.list = _global_models_api.list
models.discover = _global_models_api.discover
models.providers = _global_models_api.providers