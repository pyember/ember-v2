"""OpenAI provider implementation.

Integrates OpenAI models (GPT-3.5, GPT-4, GPT-4o) with the Ember framework.
"""

import logging
import os
from typing import Any, Dict, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ember._internal.exceptions import (
    ProviderAPIError,
)
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI language models.

    Supports all OpenAI chat models including GPT-3.5, GPT-4, and GPT-4o.
    Includes automatic retry logic and standardized error handling.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, will check environment.
        """
        super().__init__(api_key)
        self.client = openai.OpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    def complete(self, prompt: str, model: str, **kwargs) -> ChatResponse:
        """Complete a prompt using OpenAI's API.

        Args:
            prompt: The prompt text.
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo").
            **kwargs: Additional parameters like temperature, max_tokens, etc.

        Returns:
            ChatResponse with completion and usage information.

        Raises:
            ProviderAPIError: For API errors.
            AuthenticationError: For invalid API key.
            RateLimitError: When rate limited.
        """
        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Add system message if provided via context or system
        context = kwargs.pop("context", None)
        system = kwargs.pop("system", None)
        if context or system:
            messages.insert(0, {"role": "system", "content": context or system})

        # Extract known parameters
        params = {
            "model": model,
            "messages": messages,
        }

        # Map common parameters
        if "temperature" in kwargs:
            params["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs.pop("max_tokens")
        if "top_p" in kwargs:
            params["top_p"] = kwargs.pop("top_p")
        if "stop" in kwargs:
            params["stop"] = kwargs.pop("stop")

        # Add any remaining provider-specific parameters
        params.update(kwargs)

        try:
            # Make API call
            logger.debug(f"OpenAI API call: model={model}, messages={len(messages)}")
            response = self.client.chat.completions.create(**params)

            # Extract response data
            text = response.choices[0].message.content or ""

            # Build usage stats
            usage = None
            if response.usage:
                usage = UsageStats(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return ChatResponse(
                data=text,
                usage=usage,
                model_id=model,
                raw_output=response,
            )

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            raise ProviderAPIError(
                "Invalid OpenAI API key",
                context={"model": model, "error_type": "authentication"},
            ) from e

        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit: {e}")
            raise ProviderAPIError(
                "OpenAI rate limit exceeded",
                context={"model": model, "error_type": "rate_limit"},
            ) from e

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ProviderAPIError(f"OpenAI API error: {str(e)}", context={"model": model}) from e

        except Exception as e:
            logger.exception("Unexpected error calling OpenAI API")
            raise ProviderAPIError(f"Unexpected error: {str(e)}", context={"model": model}) from e

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get OpenAI API key from environment variables.

        Checks OPENAI_API_KEY and EMBER_OPENAI_API_KEY.

        Returns:
            API key or None if not found.
        """
        # This method is only called if no key was provided to __init__
        # The registry handles credential lookup through context/credentials
        return os.getenv("OPENAI_API_KEY") or os.getenv("EMBER_OPENAI_API_KEY")

    def validate_model(self, model: str) -> bool:
        """Check if this provider supports the given model.

        Args:
            model: Model name to validate.

        Returns:
            True if model is supported.
        """
        supported = {
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "text-davinci-003",
            "text-davinci-002",
        }
        return model in supported or model.startswith("gpt-")

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model name.

        Returns:
            Dictionary with model information.
        """
        info = super().get_model_info(model)

        # Add OpenAI-specific information
        context_windows = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
        }

        info.update(
            {
                "context_window": context_windows.get(model, 4096),
                "supports_functions": model.startswith("gpt-"),
                "supports_vision": model in ["gpt-4o", "gpt-4o-mini"],
            }
        )

        return info
