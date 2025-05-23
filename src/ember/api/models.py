"""Simplified Models API for Ember.

This module provides a clean, opinionated interface for interacting with language models.
It emphasizes simplicity and eliminates confusion by providing one clear way to accomplish
each task.

Examples:
    # Primary pattern: Direct invocation
    from ember.api import models
    
    response = models("gpt-4", "What is the capital of France?")
    print(response.text)  # "The capital of France is Paris."
    
    # With parameters when needed
    response = models("claude-3", "Write a haiku", temperature=0.7, max_tokens=50)
    
    # Explicit binding for reuse (performance optimization)
    gpt4 = models.bind("gpt-4", temperature=0.5)
    response1 = gpt4("Explain quantum computing")
    response2 = gpt4("What is machine learning?")
"""

from __future__ import annotations

import logging
from typing import Union, Optional, Any, Dict, TYPE_CHECKING

from ember.core.registry.model.base.context import (
    ModelContext,
    get_default_context,
)
# Define model-specific errors
class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class AuthenticationError(ModelError):
    """Raised when authentication fails."""
    pass

class RateLimitError(ModelError):
    """Raised when rate limit is exceeded."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when a model is not found."""
    pass

if TYPE_CHECKING:
    from ember.core.registry.model.base.schemas.chat_schemas import ChatResponse
    from ember.core.registry.model.base.services.model_service import ModelService

logger = logging.getLogger(__name__)


class Response:
    """Simplified response object with clear accessors.
    
    This class wraps the raw model response and provides a clean interface
    for accessing the generated text and metadata.
    """
    
    def __init__(self, raw_response: ChatResponse):
        """Initialize with a raw ChatResponse.
        
        Args:
            raw_response: The raw response from the model service
        """
        self._raw = raw_response
    
    @property
    def text(self) -> str:
        """The generated text content."""
        return self._raw.data if self._raw.data else ""
    
    @property
    def usage(self) -> Dict[str, Any]:
        """Token usage and cost information."""
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
            "cost": getattr(self._raw.usage, "cost", 0.0)
        }
    
    @property
    def model_id(self) -> Optional[str]:
        """The ID of the model that generated this response."""
        # Try to get from raw response metadata
        if hasattr(self._raw, "model_id"):
            return self._raw.model_id
        if hasattr(self._raw, "raw_output") and hasattr(self._raw.raw_output, "model"):
            return self._raw.raw_output.model
        return None
    
    def __str__(self) -> str:
        """String representation returns the generated text."""
        return self.text
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"Response(text='{self.text[:50]}...', tokens={self.usage['total_tokens']})"


class ModelBinding:
    """A bound model with preset parameters.
    
    This class represents a reusable model configuration that can be called
    multiple times with different prompts while maintaining the same parameters.
    """
    
    def __init__(self, model_id: str, service: ModelService, **params):
        """Initialize a model binding.
        
        Args:
            model_id: The model identifier
            service: The model service to use for invocations
            **params: Parameters to bind to the model
        """
        self.model_id = model_id
        self.service = service
        self.params = params
        self._validate_model_id()
    
    def _validate_model_id(self) -> None:
        """Validate that the model exists."""
        # This will raise ModelNotFoundError if invalid
        try:
            self.service.get_model(model_id=self.model_id)
        except Exception as e:
            if "not found" in str(e).lower():
                # Get available models for suggestions
                available = self.service.list_models()
                suggestions = self._find_similar_models(self.model_id, available)
                raise ModelNotFoundError(
                    f"Model '{self.model_id}' not found",
                    model_id=self.model_id,
                    suggestions=suggestions,
                    available_models=available[:10]  # Show first 10
                )
            raise
    
    def _find_similar_models(self, model_id: str, available: list[str]) -> list[str]:
        """Find models with similar names for suggestions."""
        model_lower = model_id.lower()
        
        # Exact substring matches first
        exact_matches = [m for m in available if model_lower in m.lower()]
        
        # Then partial matches
        partial_matches = []
        for m in available:
            if any(part in m.lower() for part in model_lower.split("-")):
                if m not in exact_matches:
                    partial_matches.append(m)
        
        return (exact_matches + partial_matches)[:3]
    
    def __call__(self, prompt: str, **override_params) -> Response:
        """Invoke the bound model with merged parameters.
        
        Args:
            prompt: The prompt to send to the model
            **override_params: Parameters that override the bound parameters
            
        Returns:
            Response object with generated text and metadata
        """
        merged_params = {**self.params, **override_params}
        return _invoke_model(self.model_id, prompt, self.service, **merged_params)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ModelBinding(model_id='{self.model_id}', params={self.params})"


class ModelsAPI:
    """The simplified models API.
    
    This class provides the main interface for the simplified models API.
    It supports direct invocation and explicit binding patterns.
    """
    
    def __init__(self, context: Optional[ModelContext] = None):
        """Initialize the models API.
        
        Args:
            context: Optional model context for dependency injection
        """
        self._context = context
        self._service: Optional[ModelService] = None
    
    @property
    def service(self) -> ModelService:
        """Lazy load the model service."""
        if self._service is None:
            ctx = self._context or get_default_context()
            self._service = ctx.model_service
        return self._service
    
    def __call__(self, model: str, prompt: str, **params) -> Response:
        """Primary API: Direct model invocation.
        
        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3")
            prompt: The prompt to send to the model
            **params: Optional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response object with generated text and metadata
            
        Raises:
            AuthenticationError: Missing or invalid API key
            RateLimitError: Rate limit exceeded
            ModelNotFoundError: Model doesn't exist
            ModelError: Other model-related errors
            
        Examples:
            >>> response = models("gpt-4", "Hello world")
            >>> print(response.text)
            
            >>> response = models("claude-3", "Write a poem", temperature=0.9)
            >>> print(f"Used {response.usage['total_tokens']} tokens")
        """
        return _invoke_model(model, prompt, self.service, **params)
    
    def bind(self, model: str, **params) -> ModelBinding:
        """Create a reusable model binding with preset parameters.
        
        Args:
            model: Model identifier
            **params: Parameters to bind to the model
            
        Returns:
            ModelBinding that can be called multiple times
            
        Raises:
            ModelNotFoundError: If the model doesn't exist
            
        Examples:
            >>> gpt4 = models.bind("gpt-4", temperature=0.7)
            >>> response1 = gpt4("First prompt")
            >>> response2 = gpt4("Second prompt")
            
            >>> # Override bound parameters
            >>> response3 = gpt4("Third prompt", temperature=0.9)
        """
        return ModelBinding(model, self.service, **params)
    
    async def async_call(self, model: str, prompt: str, **params) -> Response:
        """Async version of model invocation.
        
        Note: This is a placeholder for future async support.
        
        Raises:
            NotImplementedError: Async support coming in Phase 2
        """
        raise NotImplementedError(
            "Async support is not yet implemented. "
            "Use the synchronous API: models(model, prompt)"
        )
    
    def list(self, provider: Optional[str] = None) -> List[str]:
        """List available models.
        
        Args:
            provider: Optional provider filter (e.g., "openai", "anthropic")
            
        Returns:
            List of model identifiers
        """
        ctx = self._context or get_default_context()
        all_models = ctx.registry.list_models()
        
        if provider:
            return [m for m in all_models if m.startswith(f"{provider}:")]
        return all_models
    
    def info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Dictionary with model information
        """
        ctx = self._context or get_default_context()
        model_info = ctx.registry.get_model_info(model_id)
        
        if not model_info:
            raise ValueError(f"Model not found: {model_id}")
        
        return {
            "id": model_info.id,
            "provider": model_info.provider.name,
            "context_window": model_info.context_window,
            "pricing": {
                "input": model_info.cost.input_cost_per_thousand,
                "output": model_info.cost.output_cost_per_thousand,
            }
        }
    
    def get_registry(self):
        """Get the underlying model registry.
        
        Returns:
            The model registry instance
        """
        ctx = self._context or get_default_context()
        return ctx.registry
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return "ModelsAPI(simplified=True)"


def _invoke_model(
    model_id: str, 
    prompt: str, 
    service: ModelService, 
    **params
) -> Response:
    """Internal function to invoke a model with error handling.
    
    Args:
        model_id: The model identifier
        prompt: The prompt to send
        service: The model service instance
        **params: Additional parameters
        
    Returns:
        Response object with the model output
        
    Raises:
        Various ModelError subclasses based on the error type
    """
    try:
        # The model service now returns proper errors thanks to our updates
        raw_response = service.invoke_model(
            model_id=model_id,
            prompt=prompt,
            **params
        )
        
        # Wrap in simplified response
        return Response(raw_response)
        
    except ModelError:
        # Re-raise our specific errors as-is
        raise
    except Exception as e:
        # This shouldn't happen with our error mapping, but just in case
        logger.error(f"Unexpected error invoking {model_id}: {e}")
        raise ModelError(
            f"Unexpected error invoking {model_id}: {str(e)}",
            context={"model_id": model_id, "error_type": type(e).__name__}
        )


# Create the singleton API instance
models = ModelsAPI()

# Export only what's needed
__all__ = [
    'models',           # Primary API instance
    'Response',         # Response type
    'ModelBinding',     # Binding type
    # Re-export error types for convenience
    'ModelError',
    'AuthenticationError', 
    'RateLimitError',
    'ModelNotFoundError',
]