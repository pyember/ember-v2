"""Base classes for automated model discovery.

Discovery providers query APIs to fetch available models dynamically,
keeping the registry up-to-date as providers add or remove models.

Example:
    >>> class MyDiscovery(BaseDiscoveryProvider):
    ...     def fetch_models(self) -> Dict[str, Dict[str, Any]]:
    ...         models = self.client.list_models()
    ...         return {f"my:{m.id}": m.to_dict() for m in models}
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ember.core.exceptions import ModelDiscoveryError, NotImplementedFeatureError


class BaseDiscoveryProvider(ABC):
    """Abstract base class for model discovery providers.

    This class defines the interface that all model discovery implementations must follow.
    Discovery providers are responsible for querying their respective provider APIs to
    fetch available models and their capabilities, enabling automatic registration
    without manual configuration.

    Design principles:
    - Decouples model registration from manual configuration
    - Enables dynamic discovery of newly released models
    - Standardizes model metadata format across providers
    - Supports fine-grained discovery customization through provider-specific implementations

    Implementation requirements:
    - Subclasses must implement the fetch_models method
    - Implementations should handle authentication and error cases gracefully
    - Model metadata should be normalized to a consistent format
    - Each model ID should include the provider prefix (e.g., 'openai:gpt-4')

    Raises:
        ModelDiscoveryError: If the provider encounters an error during discovery.
    """

    @abstractmethod
    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve model metadata from the provider's API.

        This method must be overridden by subclasses to query the provider's API
        and return a standardized dictionary of available models and their metadata.
        The method should handle authentication, pagination, rate limiting, and
        error handling specific to the provider.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping where each key is a canonical model ID
                (e.g., 'openai:gpt-4') and its value is a dictionary containing model
                metadata such as:
                - name: The display name of the model
                - capabilities: What the model can do (text generation, embeddings, etc.)
                - model_family: The model family this model belongs to
                - context_window: Maximum context length supported
                - cost: Pricing information
                - Additional provider-specific metadata

        Raises:
            ModelDiscoveryError: On failure to fetch models due to API errors,
                authentication issues, rate limiting, or unexpected response formats.

        Example:
            ```python
            {
                "openai:gpt-4": {
                    "name": "GPT-4",
                    "capabilities": ["chat", "function-calling"],
                    "model_family": "GPT-4",
                    "context_window": 8192,
                    "cost": {
                        "input_cost_per_thousand": 0.03,
                        "output_cost_per_thousand": 0.06
                    }
                },
                # Additional models...
            }
            ```
        """
        raise NotImplementedFeatureError("Subclasses must implement fetch_models.")
