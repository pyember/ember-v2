"""Model catalog with all available models and their metadata.

This module provides the single source of truth for available models,
enabling both runtime discovery and IDE autocomplete support.
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    provider: str
    description: str
    context_window: int
    

# Complete catalog of available models
MODEL_CATALOG: Dict[str, ModelInfo] = {
    # OpenAI Models
    "gpt-4": ModelInfo(
        id="gpt-4", 
        provider="openai",
        description="Most capable GPT-4 model",
        context_window=8192
    ),
    "gpt-4-turbo": ModelInfo(
        id="gpt-4-turbo",
        provider="openai", 
        description="GPT-4 Turbo with 128K context",
        context_window=128000
    ),
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        provider="openai",
        description="Optimized GPT-4 model", 
        context_window=128000
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        description="Small, fast GPT-4 variant",
        context_window=128000
    ),
    "gpt-3.5-turbo": ModelInfo(
        id="gpt-3.5-turbo",
        provider="openai",
        description="Fast, efficient model",
        context_window=16385
    ),
    "gpt-3.5-turbo-16k": ModelInfo(
        id="gpt-3.5-turbo-16k",
        provider="openai",
        description="GPT-3.5 with 16K context", 
        context_window=16385
    ),
    
    # Anthropic Models
    "claude-3-opus": ModelInfo(
        id="claude-3-opus",
        provider="anthropic",
        description="Most capable Claude model",
        context_window=200000
    ),
    "claude-3-opus-20240229": ModelInfo(
        id="claude-3-opus-20240229",
        provider="anthropic",
        description="Most capable Claude model (versioned)",
        context_window=200000
    ),
    "claude-3-sonnet": ModelInfo(
        id="claude-3-sonnet", 
        provider="anthropic",
        description="Balanced Claude model",
        context_window=200000
    ),
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        description="Latest Claude 3.5 Sonnet model",
        context_window=200000
    ),
    "claude-3-haiku": ModelInfo(
        id="claude-3-haiku",
        provider="anthropic", 
        description="Fast, efficient Claude model",
        context_window=200000
    ),
    "claude-3-haiku-20240307": ModelInfo(
        id="claude-3-haiku-20240307",
        provider="anthropic",
        description="Fast, efficient Claude model (versioned)",
        context_window=200000
    ),
    "claude-2.1": ModelInfo(
        id="claude-2.1",
        provider="anthropic",
        description="Previous generation Claude",
        context_window=200000
    ),
    "claude-instant-1.2": ModelInfo(
        id="claude-instant-1.2",
        provider="anthropic",
        description="Fast Claude variant",
        context_window=100000
    ),
    
    # Google Models
    "gemini-pro": ModelInfo(
        id="gemini-pro",
        provider="google",
        description="Google's Gemini Pro model",
        context_window=32768
    ),
    "gemini-pro-vision": ModelInfo(
        id="gemini-pro-vision",
        provider="google",
        description="Multimodal Gemini model",
        context_window=32768
    ),
}


def list_available_models(provider: str = None) -> List[str]:
    """List all available model IDs.
    
    Args:
        provider: Optional provider filter (e.g., "openai", "anthropic")
        
    Returns:
        List of model IDs
    """
    if provider:
        return [
            model_id for model_id, info in MODEL_CATALOG.items()
            if info.provider == provider
        ]
    return list(MODEL_CATALOG.keys())


def get_providers() -> Set[str]:
    """Get unique set of providers."""
    return {info.provider for info in MODEL_CATALOG.values()}


def get_model_info(model_id: str) -> ModelInfo:
    """Get detailed information about a model.
    
    Args:
        model_id: The model identifier
        
    Returns:
        ModelInfo object
        
    Raises:
        KeyError: If model not found
    """
    if model_id not in MODEL_CATALOG:
        available = list(MODEL_CATALOG.keys())
        raise KeyError(
            f"Unknown model '{model_id}'. "
            f"Available models: {', '.join(sorted(available))}"
        )
    return MODEL_CATALOG[model_id]


# Model constants for IDE autocomplete
class Models:
    """Model constants for IDE autocomplete support.
    
    Usage:
        from ember.models import Models
        response = models(Models.GPT_4, "Hello")
    """
    # OpenAI
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    
    # Anthropic
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_INSTANT = "claude-instant-1.2"
    
    # Google
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"