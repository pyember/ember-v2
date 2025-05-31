"""Base classes for language model providers.

Define the contract all providers must implement for consistent
model interactions across different APIs.

Example:
    >>> class MyProvider(BaseProviderModel):
    ...     PROVIDER_NAME = "MyProvider"
    ...     
    ...     def create_client(self) -> Any:
    ...         return MyAPIClient(api_key=self.api_key)
    ...     
    ...     def forward(self, request: ChatRequest) -> ChatResponse:
    ...         response = self.client.chat(request.messages)
    ...         return ChatResponse(content=response.text)
"""

import abc
from typing import Any, Optional

from pydantic import BaseModel, Field

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse)
from ember.core.registry.model.base.schemas.model_info import ModelInfo


class BaseChatParameters(BaseModel):
    """Standard parameters for language model requests.

    Base class for provider-specific parameters. Extend this
    to add custom parameters for your provider.

    Attributes:
        prompt: User input text
        context: Optional system instructions
        temperature: Randomness (0.0=deterministic, 2.0=creative)
        max_tokens: Optional response length limit
        timeout: Request timeout in seconds
    """

    prompt: str
    context: Optional[str] = None
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    timeout: Optional[int] = Field(default=30, ge=1)

    def build_prompt(self) -> str:
        """Combine context and prompt into final text.

        Returns:
            Complete prompt with context prepended if present
        """
        if self.context:
            return "{context}\n\n{prompt}".format(
                context=self.context, prompt=self.prompt
            )
        return self.prompt


class BaseProviderModel(abc.ABC):
    """Abstract base for language model provider implementations.

    All providers must implement this interface to integrate with Ember.
    Handles client creation, request processing, and response normalization.
    """

    def __init__(self, model_info: ModelInfo) -> None:
        """Initialize with model metadata.

        Args:
            model_info: Model configuration and metadata
        """
        self.model_info: ModelInfo = model_info
        self.client: Any = self.create_client()

    @abc.abstractmethod
    def create_client(self) -> Any:
        """Create the provider's API client.

        Returns:
            Configured API client instance
        """
        raise NotImplementedError("Subclasses must implement create_client")

    @abc.abstractmethod
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Process chat request through the provider.

        Args:
            request: Standardized chat request

        Returns:
            Normalized chat response
        """
        raise NotImplementedError("Subclasses must implement forward")

    def get_api_model_name(self) -> str:
        """Get provider-specific model name.

        Override to transform model names for your API.

        By default, returns the model name unchanged.

        Returns:
            str: The properly formatted model name for API requests.
        """
        return self.model_info.name

    def __call__(self, prompt: str, **kwargs: Any) -> ChatResponse:
        """Allow the instance to be called as a function to process a prompt.

        This method constructs a ChatRequest using the prompt and keyword arguments,
        and then delegates the request processing to the forward() method.

        Args:
            prompt (str): The chat prompt to send.
            **kwargs (Any): Additional parameters to pass into the ChatRequest.

        Returns:
            ChatResponse: The response produced by processing the chat request.
        """
        chat_request: ChatRequest = ChatRequest(prompt=prompt, **kwargs)
        return self.forward(request=chat_request)
