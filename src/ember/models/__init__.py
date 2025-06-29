"""Ember models system.

This module provides the core infrastructure for language model interactions,
following principles of radical simplicity while maintaining SOLID compliance.

The models system is designed with three key principles:
1. Direct instantiation - No complex dependency injection
2. Explicit configuration - Clear, predictable behavior
3. Thread-safe operations - Safe for concurrent use

Architecture Notes:
    The models system implements a provider-agnostic abstraction layer that
    enables seamless switching between LLM providers while maintaining consistent
    interfaces and behavior guarantees.

    Design Decisions:
    - **Registry Pattern**: Centralized model discovery and instantiation
    - **Provider Abstraction**: Common interface across OpenAI, Anthropic, Google, etc.
    - **Lazy Loading**: Providers loaded only when first model is invoked
    - **Cost Tracking**: Built-in usage and cost calculation for all providers

    Performance Characteristics:
    - Model lookup: O(1) via dictionary
    - Provider loading: One-time cost, cached thereafter
    - Streaming: Zero-copy token iteration
    - Concurrent requests: Thread pool per provider

    Trade-offs:
    - Simplicity over flexibility: Fixed provider interface may limit advanced features
    - Memory over speed: Caches provider instances to avoid re-initialization
    - Consistency over optimization: Same code path for all providers

Examples:
    Basic usage through the high-level API:

    >>> from ember.api import models
    >>> response = models("gpt-4", "What is the capital of France?")
    >>> print(response.text)
    Paris is the capital of France.

    Direct registry usage for advanced control:

    >>> from ember.models import ModelRegistry
    >>> registry = ModelRegistry()
    >>> response = registry.invoke_model("claude-3-opus", "Explain quantum computing")
    >>> print(f"Tokens used: {response.usage.total_tokens}")

    Cost calculation utilities:

    >>> from ember.models import get_model_cost
    >>> cost_info = get_model_cost("gpt-4")
    >>> print(f"GPT-4 costs ${cost_info['input']}/1k input tokens")

Note:
    All public interfaces are designed for clarity and ease of use.
"""

from ember.models.costs import get_model_cost, get_model_costs
from ember.models.registry import ModelRegistry
from ember.models.schemas import (
    # Request/Response schemas
    ChatRequest,
    ChatResponse,
    ModelCost,
    # Model schemas
    ModelInfo,
    # Provider schemas
    ProviderInfo,
    ProviderParams,
    RateLimit,
    UsageRecord,
    # Usage schemas
    UsageStats,
    UsageSummary,
)

__all__ = [
    # Core components
    "ModelRegistry",
    # Cost functions
    "get_model_costs",
    "get_model_cost",
    # Schemas - Provider
    "ProviderInfo",
    "ProviderParams",
    # Schemas - Model
    "ModelInfo",
    "ModelCost",
    "RateLimit",
    # Schemas - Request/Response
    "ChatRequest",
    "ChatResponse",
    # Schemas - Usage
    "UsageStats",
    "UsageRecord",
    "UsageSummary",
]
