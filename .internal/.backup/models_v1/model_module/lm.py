"""
Compatibility wrapper for LMModule to ModelBinding migration.

This module provides backward compatibility during the transition from LMModule
to the models API. It will be removed in v2.0.

DEPRECATED: Use ember.api.models instead.
"""

import warnings
import logging
from typing import Any, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Track shown warnings to avoid repetition
_shown_warnings: Set[str] = set()


class LMModuleConfig(BaseModel):
    """
    DEPRECATED: Configuration for LMModule. Use models.instance() instead.
    
    This class is maintained for backward compatibility only.
    """
    id: str = Field(
        default="openai:gpt-4o",
        description="Identifier for the underlying model provider.")
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Sampling temperature for model generation.")
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate in one call.")
    cot_prompt: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought prompt to append.")
    persona: Optional[str] = Field(
        default=None,
        description="Optional persona context to prepend to the query.")


class LMModule:
    """
    DEPRECATED: Compatibility wrapper for LMModule. Use models.instance() instead.
    
    This class provides backward compatibility for code using LMModule.
    During the migration, it maintains the same interface while using
    the underlying ModelService directly.
    
    Migration examples:
        # Old way
        lm_module = LMModule(config=LMModuleConfig(id="gpt-4", temperature=0.7))
        response = lm_module(prompt="Hello")
        
        # New way
        from ember.api import models
        model = models.instance("gpt-4", temperature=0.7)
        response = model("Hello").text
    """
    
    def __init__(
        self,
        config: LMModuleConfig,
        model_service: Optional[Any] = None,
        simulate_api: bool = False) -> None:
        """Initialize compatibility wrapper with deprecation warning."""
        # Show deprecation warning only once
        if "lmmodule_deprecation" not in _shown_warnings:
            warnings.warn(
                "LMModule is deprecated and will be removed in v2.0. "
                "Use models.instance() instead. "
                "See LMMODULE_MIGRATION_GUIDE.md for migration instructions.",
                DeprecationWarning,
                stacklevel=2
            )
            _shown_warnings.add("lmmodule_deprecation")
        
        self.config = config
        self.simulate_api = simulate_api
        self._logger = logger
        
        # Store the model service if provided, otherwise we'll get it lazily
        self._model_service = model_service
        self._cached_service = None
        
        # Store persona and cot_prompt for compatibility
        self._persona = config.persona
        self._cot_prompt = config.cot_prompt
        
        # Log migration suggestion only once per model
        warning_key = f"lmmodule_migration_{config.id}"
        if warning_key not in _shown_warnings:
            logger.debug(
                f"LMModule created for model '{config.id}'. "
                f"Consider migrating to: models.instance('{config.id}', temperature={config.temperature})"
            )
            _shown_warnings.add(warning_key)
    
    def _get_model_service(self):
        """Lazily get the model service to avoid circular imports."""
        if self._model_service:
            return self._model_service
            
        if not self._cached_service:
            # Import here to avoid circular dependency
            from ember.core.registry.model.initialization import initialize_registry
            from ember.core.registry.model.base.services.model_service import ModelService
            from ember.core.registry.model.base.services.usage_service import UsageService
            
            registry = initialize_registry(auto_discover=True, force_discovery=True)
            usage_service = UsageService()
            self._cached_service = ModelService(registry=registry, usage_service=usage_service)
            
        return self._cached_service
    
    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Maintain compatibility with string return type."""
        return self.forward(prompt=prompt, **kwargs)
    
    def forward(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using the underlying model service.
        
        Returns string for backward compatibility.
        """
        if self.simulate_api:
            self._logger.debug("Simulating API call for prompt: %s", prompt)
            return f"SIMULATED_RESPONSE: {prompt}"
        
        # Assemble full prompt with persona/cot if provided
        final_prompt = self._assemble_full_prompt(prompt)
        
        # Merge kwargs with config values
        merged_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        # Add any additional kwargs
        for k, v in kwargs.items():
            if k not in merged_params:
                merged_params[k] = v
        
        try:
            # Call the model using the underlying service
            model_service = self._get_model_service()
            response = model_service.invoke_model(
                model_id=self.config.id,
                prompt=final_prompt,
                **merged_params
            )
            
            # Return just the text for backward compatibility
            return response.data if hasattr(response, "data") else str(response)
            
        except Exception as e:
            # Maintain similar error behavior
            self._logger.error(f"Error calling model: {e}")
            raise
    
    def _assemble_full_prompt(self, user_prompt: str) -> str:
        """Maintain prompt assembly for compatibility."""
        segments = []
        
        if self._persona:
            segments.append(f"[Persona: {self._persona}]\n")
        
        segments.append(user_prompt.strip())
        
        if self._cot_prompt:
            segments.append(f"\n\n# Chain of Thought:\n{self._cot_prompt.strip()}")
        
        return "\n".join(segments).strip()


# Maintain the same function for compatibility
def get_default_model_service():
    """
    DEPRECATED: This function is no longer needed with the models API.
    
    Returns a default ModelService for compatibility.
    """
    if "get_default_model_service_deprecation" not in _shown_warnings:
        warnings.warn(
            "get_default_model_service() is deprecated. "
            "The models API handles service creation automatically.",
            DeprecationWarning,
            stacklevel=2
        )
        _shown_warnings.add("get_default_model_service_deprecation")
    
    from ember.core.registry.model.initialization import initialize_registry
    from ember.core.registry.model.base.services.model_service import ModelService
    from ember.core.registry.model.base.services.usage_service import UsageService
    
    registry = initialize_registry(auto_discover=True, force_discovery=True)
    usage_service = UsageService()
    return ModelService(registry=registry, usage_service=usage_service)