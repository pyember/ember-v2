"""Google/DeepMind provider implementation.

Integrates Google's Gemini models with the Ember framework.
"""

import logging
import os
from typing import Any, Dict, Optional, TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats
from ember._internal.exceptions import (
    ProviderAPIError,
    ModelProviderError,
)

if TYPE_CHECKING:
    import google.generativeai as genai
else:
    genai = None

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """Provider for Google's Gemini models.
    
    Supports Gemini Pro, Gemini Pro Vision, and other Gemini variants.
    Uses lazy imports to avoid protobuf issues at module load time.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Google provider.
        
        Args:
            api_key: Google API key. If not provided, will check environment.
        """
        super().__init__(api_key)
        self._ensure_imports()
        genai.configure(api_key=self.api_key)
    
    def _ensure_imports(self):
        """Ensure Google Generative AI imports are loaded.
        
        This uses lazy loading to avoid protobuf issues at module import time.
        """
        global genai
        if genai is None:
            try:
                import google.generativeai as genai_import
                genai = genai_import
            except TypeError as e:
                if "got multiple values for keyword argument '_options'" in str(e):
                    raise ModelProviderError(
                        "Google Generative AI import failed due to protobuf version conflict. "
                        "This is a known issue with protobuf 3.20+. "
                        "Please install protobuf<3.20 or use a different provider."
                    ) from e
                else:
                    raise
            except ImportError as e:
                raise ModelProviderError(
                    "Google Generative AI library not installed. "
                    "Install with: pip install google-generativeai"
                ) from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def complete(self, prompt: str, model: str, **kwargs) -> ChatResponse:
        """Complete a prompt using Google's API.
        
        Args:
            prompt: The prompt text.
            model: Model name (e.g., "gemini-pro", "gemini-pro-vision").
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            ChatResponse with completion and usage information.
            
        Raises:
            ProviderAPIError: For API errors.
            AuthenticationError: For invalid API key.
        """
        # Ensure model name format
        if not model.startswith("models/"):
            model = f"models/{model}"
        
        # Get the model
        try:
            gemini_model = genai.GenerativeModel(model)
        except Exception as e:
            logger.error(f"Failed to create Gemini model: {e}")
            raise ProviderAPIError(
                f"Failed to create Gemini model: {str(e)}",
                context={"model": model}
            ) from e
        
        # Build generation config
        generation_config = {}
        
        # Map common parameters
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs.pop("max_tokens")
        if "top_p" in kwargs:
            generation_config["top_p"] = kwargs.pop("top_p")
        if "stop" in kwargs:
            generation_config["stop_sequences"] = kwargs.pop("stop")
        
        # Handle context/system message
        context = kwargs.pop("context", None)
        if context:
            prompt = f"{context}\n\n{prompt}"
        
        try:
            # Make API call
            logger.debug(f"Google API call: model={model}")
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config or None,
            )
            
            # Extract response text
            text = response.text if hasattr(response, "text") else ""
            
            # Build usage stats (Gemini doesn't provide detailed token counts)
            # We estimate based on response
            usage = UsageStats(
                prompt_tokens=len(prompt.split()) * 2,  # Rough estimate
                completion_tokens=len(text.split()) * 2,  # Rough estimate
                total_tokens=(len(prompt.split()) + len(text.split())) * 2,
            )
            
            return ChatResponse(
                data=text,
                usage=usage,
                model_id=model,
                raw_output=response,
            )
            
        except Exception as e:
            error_str = str(e)
            
            if "API_KEY_INVALID" in error_str or "invalid api key" in error_str.lower():
                logger.error(f"Google authentication error: {e}")
                raise ProviderAPIError(
                    "Invalid Google API key",
                    context={"model": model, "error_type": "authentication"}
                ) from e
                
            elif "RATE_LIMIT_EXCEEDED" in error_str:
                logger.warning(f"Google rate limit: {e}")
                raise ProviderAPIError(
                    "Google rate limit exceeded",
                    context={"model": model, "error_type": "rate_limit"}
                ) from e
                
            else:
                logger.error(f"Google API error: {e}")
                raise ProviderAPIError(
                    f"Google API error: {error_str}",
                    context={"model": model}
                ) from e
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get Google API key from environment variables.
        
        Checks GOOGLE_API_KEY, GEMINI_API_KEY, and EMBER_GOOGLE_API_KEY.
        
        Returns:
            API key or None if not found.
        """
        return (
            os.getenv("GOOGLE_API_KEY") or 
            os.getenv("GEMINI_API_KEY") or
            os.getenv("EMBER_GOOGLE_API_KEY")
        )
    
    def validate_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Args:
            model: Model name to validate.
            
        Returns:
            True if model is supported.
        """
        supported = {
            "gemini-pro", "gemini-pro-vision",
            "gemini-1.5-pro", "gemini-1.5-flash",
            "models/gemini-pro", "models/gemini-pro-vision",
        }
        return model in supported or model.startswith("gemini")
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model: Model name.
            
        Returns:
            Dictionary with model information.
        """
        info = super().get_model_info(model)
        
        # Add Google-specific information
        info.update({
            "context_window": 32768,  # Gemini Pro default
            "supports_vision": "vision" in model,
            "supports_functions": True,  # Gemini supports function calling
        })
        
        return info