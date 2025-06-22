"""Model registry with direct initialization.

This module provides a streamlined ModelRegistry that uses explicit
provider mapping and environment-based configuration.

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

import logging
import threading
from typing import Dict, Optional, List, Type

from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.cost import ModelCost
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ModelNotFoundError
)
from ember.core.registry.model.providers._registry import (
    get_provider_class,
    resolve_model_id,
    is_provider_available
)
from ember.core.registry.model._costs import get_model_cost


class ModelRegistry:
    """Thread-safe registry for managing model instances.
    
    This registry provides lazy instantiation of models with a cleaner
    implementation that relies on explicit provider mapping and
    environment-based configuration.
    
    Key simplifications:
        - Single lock for all operations (proven sufficient)
        - Direct provider instantiation (no complex factory)
        - Automatic cost loading from _costs module
        - No dynamic discovery (explicit is better)
    
    Attributes:
        _models: Cache of instantiated model instances.
        _lock: Single lock for thread safety.
        _logger: Logger instance.
        
    Examples:
        >>> registry = ModelRegistry()
        >>> model = registry.get_model("gpt-4")
        >>> response = model("Hello, world!")
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the registry.
        
        Args:
            logger: Optional logger instance. If None, creates default.
        """
        self._models: Dict[str, BaseProviderModel] = {}
        self._lock = threading.Lock()
        self._logger = logger or logging.getLogger(self.__class__.__name__)
    
    def get_model(self, model_id: str) -> BaseProviderModel:
        """Get or create a model instance.
        
        This method provides lazy instantiation - models are only created
        when first requested, then cached for subsequent calls.
        
        Args:
            model_id: Model identifier (e.g., "gpt-4", "claude-3").
            
        Returns:
            The model instance ready for use.
            
        Raises:
            ValueError: If model_id is empty.
            ModelNotFoundError: If the model or provider is not available.
            
        Examples:
            >>> model = registry.get_model("gpt-4")
            >>> # First call creates instance
            >>> 
            >>> model2 = registry.get_model("gpt-4")  
            >>> # Second call returns cached instance
            >>> assert model is model2  # Same object
        """
        if not model_id:
            raise ValueError("Model ID cannot be empty")
        
        # Fast path - check if already cached
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]
        
        # Create model instance
        model = self._create_model(model_id)
        
        # Cache it
        with self._lock:
            # Double-check in case another thread created it
            if model_id not in self._models:
                self._models[model_id] = model
                self._logger.info(f"Instantiated and cached model: {model_id}")
            else:
                # Another thread beat us, use their instance
                model = self._models[model_id]
        
        return model
    
    def _create_model(self, model_id: str) -> BaseProviderModel:
        """Create a new model instance.
        
        Args:
            model_id: Model identifier.
            
        Returns:
            New model instance.
            
        Raises:
            ModelNotFoundError: If provider not available.
        """
        # Resolve provider from model ID
        provider_name, model_name = resolve_model_id(model_id)
        
        if provider_name == "unknown":
            raise ModelNotFoundError(
                f"Cannot determine provider for model '{model_id}'. "
                f"Use explicit format: provider/model (e.g., 'openai/gpt-4')"
            )
        
        if not is_provider_available(provider_name):
            raise ModelNotFoundError(
                f"Provider '{provider_name}' not available for model '{model_id}'"
            )
        
        # Get provider class
        try:
            provider_class = get_provider_class(provider_name)
        except ValueError as e:
            raise ModelNotFoundError(str(e))
        
        # Create model info with costs
        model_info = self._create_model_info(model_id, provider_name, model_name)
        
        # Instantiate provider
        try:
            model = provider_class(model_info=model_info)
            self._logger.debug(f"Created {provider_name} provider for {model_id}")
            return model
        except Exception as e:
            self._logger.error(f"Failed to create model {model_id}: {e}")
            raise ModelNotFoundError(
                f"Failed to instantiate model '{model_id}': {str(e)}"
            )
    
    def _create_model_info(self, model_id: str, provider_name: str, 
                          model_name: str) -> ModelInfo:
        """Create ModelInfo with cost data.
        
        Args:
            model_id: Full model identifier.
            provider_name: Provider name.
            model_name: Model name without provider.
            
        Returns:
            ModelInfo instance with costs.
        """
        # Get cost information
        cost_data = get_model_cost(model_name)
        cost = ModelCost(
            input_cost_per_thousand=cost_data["input"],
            output_cost_per_thousand=cost_data["output"]
        )
        
        # Get API key from environment
        import os
        api_key_env = f"{provider_name.upper()}_API_KEY"
        api_key = os.getenv(api_key_env, "")
        
        if not api_key:
            self._logger.warning(
                f"No API key found in {api_key_env} for provider {provider_name}"
            )
        
        # Create provider info
        provider_info = ProviderInfo(
            name=provider_name.capitalize(),
            default_api_key=api_key
        )
        
        # Create model info
        return ModelInfo(
            id=model_id,
            name=model_name,
            cost=cost,
            provider=provider_info,
            context_window=cost_data.get("context", 4096)
        )
    
    def list_models(self) -> List[str]:
        """List all instantiated models.
        
        Returns:
            List of model IDs that have been instantiated.
            
        Note:
            This only returns models that have been requested,
            not all possible models.
            
        Examples:
            >>> registry.get_model("gpt-4")
            >>> registry.get_model("claude-3")
            >>> models = registry.list_models()
            >>> # Returns ["gpt-4", "claude-3"]
        """
        with self._lock:
            return list(self._models.keys())
    
    def clear_cache(self) -> None:
        """Clear all cached model instances.
        
        This is mainly useful for testing or when you need to
        force re-instantiation of models.
        
        Examples:
            >>> registry.clear_cache()
            >>> # All models will be re-created on next access
        """
        with self._lock:
            self._models.clear()
            self._logger.info("Cleared model cache")
    
    def discover_models(self) -> int:
        """Compatibility method for old code expecting discovery.
        
        Returns:
            0 - no models discovered (discovery is not used in new design)
        """
        return 0
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is already instantiated.
        
        Args:
            model_id: Model identifier to check.
            
        Returns:
            True if model is in cache, False otherwise.
            
        Examples:
            >>> registry.is_model_cached("gpt-4")
            False
            >>> registry.get_model("gpt-4")
            >>> registry.is_model_cached("gpt-4")
            True
        """
        with self._lock:
            return model_id in self._models