"""Simplified Models API with direct initialization.

This module provides a streamlined interface for language model interactions,
using direct initialization instead of complex dependency injection.

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import logging
import os
from typing import Union, Optional, Any, Dict, List, TYPE_CHECKING

from ember.models import ModelRegistry
from ember.models.schemas import ChatResponse
from ember._internal.exceptions import (
    ModelError,
    ModelNotFoundError,
    ProviderAPIError
)

logger = logging.getLogger(__name__)


class Response:
    """Response object for model outputs.
    
    Provides clean access to generated text and metadata from language model
    responses. This is one of Ember's key innovations - a simple, consistent
    interface regardless of provider.
    
    Attributes:
        text: The generated text content.
        usage: Token usage statistics with automatic cost calculation.
        model_id: Identifier of the model that generated the response.
        
    Examples:
        >>> response = models("gpt-4", "What is 2+2?")
        >>> print(response.text)
        The answer is 4.
        
        >>> print(response.usage)
        {'prompt_tokens': 10, 'completion_tokens': 5, 
         'total_tokens': 15, 'cost': 0.0006}
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
        if hasattr(self._raw, "raw_output") and hasattr(self._raw.raw_output, "model"):
            return self._raw.raw_output.model
        return None
    
    def __str__(self) -> str:
        """Return the generated text."""
        return self.text
    
    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"Response(text='{self.text[:50]}...', tokens={self.usage['total_tokens']})"


class ModelBinding:
    """Reusable model configuration with preset parameters.
    
    This is a unique Ember innovation that improves performance by validating
    parameters once and reusing the configuration for multiple calls.
    
    Examples:
        >>> # Create a binding with preset temperature
        >>> creative_gpt4 = models.instance("gpt-4", temperature=0.9)
        >>> 
        >>> # Use it multiple times efficiently
        >>> story1 = creative_gpt4("Write a story about a dragon")
        >>> story2 = creative_gpt4("Write a story about a robot")
        >>> 
        >>> # Override parameters when needed
        >>> serious = creative_gpt4("Explain quantum physics", temperature=0.1)
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
        self._validate_model_id()
    
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
    """Simplified interface for language model interactions.
    
    This is the main entry point for using language models in Ember.
    It provides a clean, direct interface without requiring client
    initialization - a key differentiator from other libraries.
    
    Examples:
        >>> from ember.api import models
        >>> 
        >>> # Direct invocation - the simplest way
        >>> response = models("gpt-4", "Hello world")
        >>> print(response.text)
        >>> 
        >>> # With parameters
        >>> response = models("gpt-4", "Be creative", temperature=0.9)
        >>> 
        >>> # Reusable binding for efficiency
        >>> gpt4 = models.instance("gpt-4", temperature=0.7)
        >>> response = gpt4("First prompt")
    """
    
    def __init__(self):
        """Initialize ModelsAPI with direct registry creation."""
        # Create registry directly - no complex context management
        metrics = None
        if os.getenv("EMBER_METRICS_ENABLED", "").lower() == "true":
            metrics = self._create_metrics()
        
        self._registry = ModelRegistry(metrics=metrics)
    
    def _create_metrics(self) -> Dict[str, Any]:
        """Create metrics collectors if enabled.
        
        Returns:
            Dictionary of metric collectors.
        """
        # Simplified metrics - could integrate with Prometheus later
        return {
            "invocation_duration": None,  # Placeholder
            "model_invocations": None,    # Placeholder
        }
    
    def __call__(self, model: str, prompt: str, **params) -> Response:
        """Invoke a language model directly.
        
        This is the primary interface - just call models() with a model
        name and prompt. No client initialization required.
        
        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3", "openai/gpt-4").
            prompt: The prompt to send to the model.
            **params: Optional parameters like temperature, max_tokens, etc.
                     Special parameter 'providers' for provider preferences.
            
        Returns:
            Response object with generated text and metadata.
            
        Raises:
            ModelNotFoundError: Model doesn't exist or provider unavailable.
            ModelProviderError: Missing or invalid API key.
            ProviderAPIError: Provider-specific errors (including rate limits).
            
        Examples:
            >>> # Simple usage
            >>> response = models("gpt-4", "Hello world")
            >>> 
            >>> # With parameters
            >>> response = models("gpt-4", "Be creative", 
            ...                  temperature=0.9, max_tokens=100)
            >>> 
            >>> # With explicit provider
            >>> response = models("openai/gpt-4", "Hello")
            >>> 
            >>> # With provider preferences (advanced)
            >>> response = models("gpt-4", "Hello", 
            ...                  providers=["azure", "openai"])
        """
        # Extract provider preferences if specified
        providers = params.pop("providers", None)
        
        # TODO: Implement provider preferences in registry
        if providers:
            logger.debug(f"Provider preferences specified: {providers}")
        
        raw_response = self._registry.invoke_model(model, prompt, **params)
        return Response(raw_response)
    
    def instance(self, model: str, **params) -> ModelBinding:
        """Create a reusable model binding with preset parameters.
        
        This is an Ember innovation - create a model configuration once
        and reuse it efficiently for multiple calls.
        
        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3").
            **params: Default parameters for all calls (temperature, etc.).
            
        Returns:
            ModelBinding that can be called multiple times.
            
        Examples:
            >>> # Create specialized model configurations
            >>> creative = models.instance("gpt-4", temperature=0.9)
            >>> analytical = models.instance("gpt-4", temperature=0.1)
            >>> 
            >>> # Use them for different purposes
            >>> story = creative("Write a story")
            >>> analysis = analytical("Analyze this data: ...")
        """
        return ModelBinding(model, self._registry, **params)
    
    def list(self) -> List[str]:
        """List all available models.
        
        Returns:
            List of model IDs that have been instantiated.
            
        Examples:
            >>> models_list = models.list()
            >>> print(models_list)
            ['gpt-4', 'claude-3-opus']
        """
        return self._registry.list_models()
    
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return "ModelsAPI(simplified=True, direct_init=True)"


# Create global instance
_global_models_api = ModelsAPI()


# Module-level function for convenience
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


# Expose the instance method for creating bindings
models.instance = _global_models_api.instance
models.list = _global_models_api.list