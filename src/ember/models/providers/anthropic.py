"""Anthropic provider implementation.

Integrates Anthropic's Claude models with the Ember framework.
"""

import logging
import os
from typing import Any, Dict, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats
from ember._internal.exceptions import (
    ProviderAPIError,
    ModelProviderError,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude models.
    
    Supports all Claude models including Claude 3 Opus, Sonnet, and Haiku.
    Includes automatic retry logic and standardized error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key. If not provided, will check environment.
        """
        super().__init__(api_key)
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _resolve_model_name(self, model: str) -> str:
        """Resolve user-friendly model names to actual API model IDs.
        
        Maps short names like 'claude-3-haiku' to full API names like 
        'claude-3-haiku-20240307'. This provides a better user experience
        while maintaining API compatibility.
        
        Args:
            model: User-provided model name.
            
        Returns:
            Resolved model name for API calls.
        """
        # Model name mappings from friendly names to API names
        model_mappings = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229", 
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
            # Already versioned names pass through unchanged
            "claude-3-opus-20240229": "claude-3-opus-20240229",
            "claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307": "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
            # Legacy models
            "claude-2.1": "claude-2.1",
            "claude-2.0": "claude-2.0",
            "claude-instant-1.2": "claude-instant-1.2",
        }
        
        return model_mappings.get(model, model)
    
    def complete(self, prompt: str, model: str, **kwargs) -> ChatResponse:
        """Complete a prompt using Anthropic's API.
        
        Args:
            prompt: The prompt text.
            model: Model name (e.g., "claude-3-opus", "claude-3-sonnet").
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            ChatResponse with completion and usage information.
            
        Raises:
            ProviderAPIError: For API errors.
            AuthenticationError: For invalid API key.
            RateLimitError: When rate limited.
        """
        # Resolve model name to actual API model ID
        resolved_model = self._resolve_model_name(model)
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Extract system message if provided
        system = kwargs.pop("context", None) or kwargs.pop("system", None)
        
        # Build parameters
        params = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),  # Anthropic requires this
        }
        
        # Add system message if provided
        if system:
            params["system"] = system
        
        # Map common parameters
        if "temperature" in kwargs:
            params["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            params["top_p"] = kwargs.pop("top_p")
        if "stop" in kwargs:
            params["stop_sequences"] = kwargs.pop("stop")
        
        # Add any remaining provider-specific parameters
        params.update(kwargs)
        
        try:
            # Make API call
            logger.debug(f"Anthropic API call: model={resolved_model} (requested: {model}), messages={len(messages)}")
            response = self.client.messages.create(**params)
            
            # Extract response text
            text = ""
            if response.content:
                # Handle different content types
                for content in response.content:
                    if hasattr(content, "text"):
                        text += content.text
                    elif isinstance(content, str):
                        text += content
            
            # Build usage stats
            usage = None
            if hasattr(response, "usage"):
                usage = UsageStats(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                )
            
            return ChatResponse(
                data=text,
                usage=usage,
                model_id=model,
                raw_output=response,
            )
            
        except anthropic.AuthenticationError as e:
            logger.error(f"Anthropic authentication error: {e}")
            raise ProviderAPIError(
                "Invalid Anthropic API key",
                context={
                    "model": model, 
                    "resolved_model": resolved_model,
                    "error_type": "authentication"
                }
            ) from e
            
        except anthropic.RateLimitError as e:
            logger.warning(f"Anthropic rate limit: {e}")
            raise ProviderAPIError(
                "Anthropic rate limit exceeded",
                context={
                    "model": model,
                    "resolved_model": resolved_model,
                    "error_type": "rate_limit"
                }
            ) from e
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise ProviderAPIError(
                f"Anthropic API error: {str(e)}",
                context={
                    "model": model,
                    "resolved_model": resolved_model
                }
            ) from e
            
        except Exception as e:
            logger.exception(f"Unexpected error calling Anthropic API")
            raise ProviderAPIError(
                f"Unexpected error: {str(e)}",
                context={
                    "model": model,
                    "resolved_model": resolved_model
                }
            ) from e
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get Anthropic API key from environment variables.
        
        Checks ANTHROPIC_API_KEY and EMBER_ANTHROPIC_API_KEY.
        
        Returns:
            API key or None if not found.
        """
        return os.getenv("ANTHROPIC_API_KEY") or os.getenv("EMBER_ANTHROPIC_API_KEY")
    
    def validate_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Args:
            model: Model name to validate.
            
        Returns:
            True if model is supported.
        """
        # Check if it's a known model (either short or full name)
        known_models = {
            "claude-3-opus", "claude-3-opus-20240229",
            "claude-3-sonnet", "claude-3-sonnet-20240229",
            "claude-3-5-sonnet", "claude-3-5-sonnet-20241022",
            "claude-3-haiku", "claude-3-haiku-20240307",
            "claude-2.1", "claude-2.0",
            "claude-instant-1.2",
        }
        
        # Also check if it can be resolved
        resolved = self._resolve_model_name(model)
        return model in known_models or resolved != model or model.startswith("claude")
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model: Model name.
            
        Returns:
            Dictionary with model information.
        """
        info = super().get_model_info(model)
        
        # Add Anthropic-specific information
        # Use resolved model name for lookup
        resolved_model = self._resolve_model_name(model)
        
        context_windows = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000,
        }
        
        info.update({
            "context_window": context_windows.get(resolved_model, 100000),
            "supports_vision": resolved_model.startswith("claude-3"),
            "supports_functions": False,  # Claude doesn't have native function calling
        })
        
        return info