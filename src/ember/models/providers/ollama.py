"""Ollama provider implementation.

Integrates a local Ollama endpoint with the Ember framework.
"""

import logging
import os
from typing import Any, Dict, Optional

import ollama

from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats
from ember._internal.exceptions import (
    ProviderAPIError,
    ModelProviderError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Provider for Ollama language models.
    
    Supports all Ollama chat models.
    Includes automatic retry logic and standardized error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Ollama provider.
        """
        super().__init__(" " if api_key is None else api_key)
    
    def complete(self, prompt: str, model: str, **kwargs) -> ChatResponse:
        """Complete a prompt using Ollama's API.

        Args:
            prompt: The prompt text.
            model: Model name (e.g., "ollama/granite3.2:8b").
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            ChatResponse with completion and usage information.
            
        Raises:
            ProviderAPIError: For API errors.
            AuthenticationError: For invalid API key.
            RateLimitError: When rate limited.
        """
        if not self.validate_model(model):
            raise ProviderAPIError(
                f"Ollama is not running",
                context={"model": model}
            )
            
        model = model.replace("ollama/", "", 1)
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if provided via context
        context = kwargs.pop("context", None)
        if context:
            messages.insert(0, {"role": "system", "content": context})
        
        # Extract known options
        options = {
        }
        
        # Map common parameters
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            options["num_ctx"] = kwargs.pop("max_tokens")
        if "top_p" in kwargs:
            options["top_p"] = kwargs.pop("top_p")
        if "stop" in kwargs:
            options["stop"] = kwargs.pop("stop")
        
        # Add any remaining provider-specific parameters
        options.update(kwargs)
        
        try:
            # Make API call
            logger.debug(f"Ollama API call: model={model}, messages={len(messages)}")
            response = ollama.chat(model=model,messages=messages,options=options)
            
            # Extract response data
            text = response["message"]["content"] or ""
            
            # Build usage stats
            usage = UsageStats(
                prompt_tokens=response["prompt_eval_count"],
                completion_tokens=response["eval_count"],
                total_tokens=response["prompt_eval_count"] + response["eval_count"]
            )
            
            return ChatResponse(
                data=text,
                usage=usage,
                model_id=model,
                raw_output=response,
            )

        except Exception as e:
            logger.exception(f"Unexpected error calling Ollama API")
            raise ProviderAPIError(
                f"Unexpected error: {str(e)}",
                context={"model": model}
            ) from e
    
    def _get_api_key_from_env(self) -> Optional[str]:
        return None
    
    def validate_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Args:
            model: Model name to validate.
            
        Returns:
            True if model is supported.
        """
        model = model.replace("ollama/", "", 1)
        try:
            models = ollama.list()["models"]
            if any(m['name'] == model for m in models):
                return True
            import sys
            print(f"Pulling model {model}", file=sys.stderr)
            ollama.pull(model)
            return True
        except Exception as e:
            logger.warning(f"Unable to pull desired model {model}")
            return False
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model: Model name.
            
        Returns:
            Dictionary with model information.
        """
        return ollama.show(model)
