"""Base provider interface for language models.

This module defines the minimal contract that all providers must implement.
The design follows the Interface Segregation Principle - we define only the
absolute minimum required methods, allowing providers maximum flexibility.

Design Philosophy:
    1. Minimal interface - Just enough to ensure compatibility
    2. Clear contracts - Abstract methods with explicit documentation
    3. Flexible implementation - Providers decide their own complexity
    4. Error consistency - Common error types across all providers

The base class intentionally avoids helper methods or shared logic that
might constrain provider implementations. Each provider is free to organize
its internals as appropriate for its target API.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ember.models.schemas import ChatResponse


class BaseProvider(ABC):
    """Minimal provider interface following SOLID principles.
    
    This abstract base class defines the contract for all model providers.
    It's intentionally minimal - just two required methods and two optional
    ones. This allows providers to range from simple HTTP wrappers to
    complex systems with caching, retries, and failover.
    
    The design follows these principles:
        - Single Responsibility: Only defines the provider contract
        - Open/Closed: Extensible via subclassing, closed for modification
        - Interface Segregation: Minimal interface, no unused methods
        - Dependency Inversion: Registry depends on this abstraction
    
    Attributes:
        api_key: Authentication credential for the provider's API.
                 Can be None if provider doesn't require auth.
                 
    Examples:
        Implementing a minimal provider:
        
        >>> class MinimalProvider(BaseProvider):
        ...     def complete(self, prompt, model, **kwargs):
        ...         # Make API call and return ChatResponse
        ...         return ChatResponse(data="Generated text")
        ...     
        ...     def _get_api_key_from_env(self):
        ...         return os.getenv("MINIMAL_API_KEY")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the provider.
        
        The initialization follows a clear precedence order for API keys:
        1. Explicitly passed api_key parameter
        2. Environment variable (via _get_api_key_from_env)
        3. Raise ValueError if neither available
        
        This pattern supports both programmatic usage and twelve-factor
        app principles for configuration.
        
        Args:
            api_key: API key for authentication. If not provided,
                    will attempt to get from environment variables.
                    
        Raises:
            ValueError: If no API key is available from any source.
            
        Examples:
            Explicit key:
            
            >>> provider = SomeProvider(api_key="sk-...")
            
            Environment key:
            
            >>> os.environ["OPENAI_API_KEY"] = "sk-..."
            >>> provider = OpenAIProvider()  # Gets from env
        """
        self.api_key = api_key or self._get_api_key_from_env()
        if not self.api_key:
            raise ValueError(
                f"API key required for {self.__class__.__name__}. "
                f"Set via constructor or environment variable."
            )
    
    @abstractmethod
    def complete(self, prompt: str, model: str, **kwargs) -> ChatResponse:
        """Complete a prompt using the specified model.
        
        This is the core method that all providers must implement. It serves
        as the bridge between Ember's unified API and provider-specific APIs.
        
        Implementation Requirements:
            1. Make the API call to the provider
            2. Handle provider-specific errors gracefully
            3. Convert response to ChatResponse format
            4. Include usage statistics if available
            5. Preserve raw response in ChatResponse.raw_output
        
        Args:
            prompt: The input text to complete. Providers should handle
                   any necessary formatting (e.g., chat templates).
            model: Model identifier. Providers may need to map this to
                  their internal model names.
            **kwargs: Provider-specific parameters. Common ones include:
                - temperature (float): Sampling temperature (0.0-2.0)
                - max_tokens (int): Maximum response length
                - top_p (float): Nucleus sampling parameter
                - stop (str|List[str]): Stop sequences
                - stream (bool): Whether to stream response
                - And any provider-specific parameters
        
        Returns:
            ChatResponse containing:
                - data: The generated text
                - usage: Token counts and costs (if available)
                - model_id: The model used
                - raw_output: Original API response for debugging
            
        Raises:
            ProviderAPIError: For general API errors. Use context dict:
                - error_type: "rate_limit" for rate limiting
                - error_type: "authentication" for auth issues
                - error_type: "invalid_request" for bad parameters
                - Include other relevant context
                
        Examples:
            Basic completion:
            
            >>> response = provider.complete("Hello", "gpt-4")
            >>> print(response.data)
            
            With parameters:
            
            >>> response = provider.complete(
            ...     "Write a story",
            ...     "claude-3",
            ...     temperature=0.9,
            ...     max_tokens=1000
            ... )
        """
        pass
    
    @abstractmethod
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables.
        
        Each provider should implement this to check their specific
        environment variables. The method should check multiple possible
        variable names for flexibility.
        
        Recommended Implementation Pattern:
            1. Check PROVIDER_API_KEY (e.g., OPENAI_API_KEY)
            2. Check EMBER_PROVIDER_API_KEY (e.g., EMBER_OPENAI_API_KEY)
            3. Check any legacy or alternative names
            4. Return None if not found (don't raise)
        
        Returns:
            API key string or None if not found in environment.
            
        Examples:
            Typical implementation:
            
            >>> def _get_api_key_from_env(self) -> Optional[str]:
            ...     return (
            ...         os.getenv("OPENAI_API_KEY") or
            ...         os.getenv("EMBER_OPENAI_API_KEY")
            ...     )
        """
        pass
    
    def validate_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Optional method for providers to validate model names before
        attempting API calls. This can prevent unnecessary API requests
        for invalid model names.
        
        The default implementation accepts all models, following the
        principle of late validation - let the API return errors for
        truly invalid models.
        
        Args:
            model: Model identifier to validate.
            
        Returns:
            True if model is supported, False otherwise.
            
        Examples:
            Override for specific models:
            
            >>> def validate_model(self, model: str) -> bool:
            ...     valid_models = {"gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"}
            ...     return model in valid_models
            
            Pattern matching:
            
            >>> def validate_model(self, model: str) -> bool:
            ...     return model.startswith("gpt-") or model == "davinci"
        """
        return True
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Optional method for providers to expose model metadata. This can
        include context windows, pricing, capabilities, and other details
        useful for model selection and validation.
        
        The default implementation returns minimal information. Providers
        should override to add model-specific details.
        
        Args:
            model: Model identifier.
            
        Returns:
            Dictionary with model information. Common fields:
                - model: The model identifier
                - provider: Provider class name
                - context_window: Maximum token limit
                - supports_streaming: Boolean
                - supports_functions: Boolean
                - supports_vision: Boolean
                - Any provider-specific metadata
                
        Examples:
            Rich implementation:
            
            >>> def get_model_info(self, model: str) -> Dict[str, Any]:
            ...     info = super().get_model_info(model)
            ...     if model == "gpt-4":
            ...         info.update({
            ...             "context_window": 8192,
            ...             "supports_functions": True,
            ...             "supports_vision": False,
            ...             "training_cutoff": "2023-09"
            ...         })
            ...     return info
        """
        return {
            "model": model,
            "provider": self.__class__.__name__,
        }