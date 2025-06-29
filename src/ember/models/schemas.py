"""Consolidated schemas for the models system.

This module defines the data structures that form the backbone of Ember's
models system. Rather than scattering type definitions across multiple files,
everything is consolidated here for clarity and ease of maintenance.

Design Principles:
    1. Simplicity over flexibility - Each schema has a clear purpose
    2. Provider agnostic - Common interface regardless of backend
    3. Type safety - Full typing for better IDE support and validation
    4. Zero magic - Plain dataclasses, no metaclass complexity

The schemas are organized into logical groups:
    - Provider schemas: Configuration for model providers
    - Model schemas: Model metadata and capabilities
    - Request/Response: Universal API interface
    - Usage tracking: Cost calculation and analytics

Examples:
    Basic request/response flow:

    >>> from ember.models.schemas import ChatRequest, ChatResponse
    >>> request = ChatRequest(
    ...     prompt="Explain quantum computing",
    ...     max_tokens=100,
    ...     temperature=0.7
    ... )

    Usage tracking:

    >>> from ember.models.schemas import UsageStats
    >>> usage = UsageStats(
    ...     prompt_tokens=50,
    ...     completion_tokens=100,
    ...     cost_usd=0.0015
    ... )
    >>> print(f"Total cost: ${usage.cost_usd:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

# Provider-related schemas


@dataclass
class ProviderInfo:
    """Metadata about a language model provider.

    This schema captures the essential configuration needed to interact
    with a provider's API. It's intentionally minimal - just enough to
    establish a connection without overengineering.

    Attributes:
        name: Provider identifier (e.g., "openai", "anthropic").
        default_api_key: Fallback API key if not specified per-model.
        base_url: Custom API endpoint for enterprise/proxy setups.
        custom_args: Provider-specific configuration options.

    Examples:
        Standard provider:

        >>> provider = ProviderInfo(
        ...     name="openai",
        ...     default_api_key="sk-..."
        ... )

        Enterprise proxy:

        >>> provider = ProviderInfo(
        ...     name="openai",
        ...     base_url="https://api.company.com/v1",
        ...     custom_args={"org_id": "org-123"}
        ... )
    """

    name: str
    default_api_key: Optional[str] = None
    base_url: Optional[str] = None
    custom_args: Optional[Dict[str, Any]] = None


class ProviderParams(TypedDict, total=False):
    """Provider-specific parameters for model invocation.

    This TypedDict allows passing through provider-specific parameters
    without coupling the core API to individual provider quirks. It's
    designed to be extended as providers add new features.

    The total=False setting means all fields are optional, supporting
    the principle of progressive disclosure - basic usage requires no
    provider-specific knowledge.

    Examples:
        OpenAI function calling:

        >>> params: ProviderParams = {
        ...     "tools": [{"type": "function", "function": {...}}],
        ...     "tool_choice": "auto"
        ... }

        Anthropic system prompts:

        >>> params: ProviderParams = {
        ...     "system": "You are a helpful assistant",
        ...     "metadata": {"user_id": "123"}
        ... }
    """

    # OpenAI specific
    response_format: Optional[Dict[str, str]]
    seed: Optional[int]
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]

    # Anthropic specific
    system: Optional[str]
    metadata: Optional[Dict[str, Any]]

    # Common
    stop: Optional[Union[str, List[str]]]
    logprobs: Optional[bool]
    top_logprobs: Optional[int]
    user: Optional[str]


# Model configuration schemas


@dataclass
class ModelCost:
    """Cost information for a model.

    This schema standardizes cost representation across all providers.
    Costs are always in USD per 1000 tokens for consistency and to
    avoid floating point precision issues with per-token costs.

    Attributes:
        input_cost_per_thousand: USD cost per 1000 input tokens.
        output_cost_per_thousand: USD cost per 1000 output tokens.

    Properties:
        input_cost_per_token: Calculated cost per single input token.
        output_cost_per_token: Calculated cost per single output token.

    Examples:
        Define costs:

        >>> gpt4_cost = ModelCost(
        ...     input_cost_per_thousand=30.0,
        ...     output_cost_per_thousand=60.0
        ... )

        Calculate request cost:

        >>> cost = ModelCost(input_cost_per_thousand=30.0,
        ...                  output_cost_per_thousand=60.0)
        >>> input_tokens = 1500
        >>> output_tokens = 500
        >>> total = (
        ...     (input_tokens / 1000) * cost.input_cost_per_thousand +
        ...     (output_tokens / 1000) * cost.output_cost_per_thousand
        ... )
        >>> print(f"Total cost: ${total:.4f}")
        Total cost: $75.0000
    """

    input_cost_per_thousand: float
    output_cost_per_thousand: float

    @property
    def input_cost_per_token(self) -> float:
        """Cost per single input token."""
        return self.input_cost_per_thousand / 1000.0

    @property
    def output_cost_per_token(self) -> float:
        """Cost per single output token."""
        return self.output_cost_per_thousand / 1000.0


@dataclass
class RateLimit:
    """Rate limiting parameters for a model.

    Captures provider-imposed rate limits for capacity planning.
    Both fields are optional since not all providers expose limits.

    Attributes:
        tokens_per_minute: Maximum tokens per minute (input + output).
        requests_per_minute: Maximum API requests per minute.

    Examples:
        >>> limits = RateLimit(
        ...     tokens_per_minute=90000,
        ...     requests_per_minute=3500
        ... )
    """

    tokens_per_minute: Optional[int] = None
    requests_per_minute: Optional[int] = None


@dataclass
class ModelInfo:
    """Complete metadata and configuration for a model.

    This schema serves as the single source of truth for a model's
    capabilities and configuration. It combines static metadata with
    runtime configuration in a clean, extensible structure.

    The design follows the principle of progressive disclosure:
    only id, name, and provider are required. Everything else has
    sensible defaults.

    Attributes:
        id: Unique model identifier (e.g., "gpt-4", "claude-3-opus").
        name: Human-readable model name.
        provider: Provider name (e.g., "openai", "anthropic").
        context_window: Maximum token context size.
        cost: Pricing information.
        rate_limit: API rate limits.
        api_key: Model-specific API key (overrides provider default).
        base_url: Model-specific endpoint (overrides provider default).
        supports_streaming: Whether streaming responses are supported.
        supports_functions: Whether function/tool calling is supported.
        supports_vision: Whether image inputs are supported.

    Examples:
        Basic model:

        >>> model = ModelInfo(
        ...     id="gpt-4",
        ...     name="GPT-4",
        ...     provider="openai"
        ... )

        Full configuration:

        >>> model = ModelInfo(
        ...     id="gpt-4-turbo",
        ...     name="GPT-4 Turbo",
        ...     provider="openai",
        ...     context_window=128000,
        ...     cost=ModelCost(input_cost_per_thousand=10.0,
        ...                    output_cost_per_thousand=30.0),
        ...     rate_limit=RateLimit(tokens_per_minute=300000),
        ...     supports_streaming=True,
        ...     supports_functions=True,
        ...     supports_vision=True
        ... )
    """

    id: str
    name: str
    provider: str
    context_window: int = 4096
    cost: Optional[ModelCost] = None
    rate_limit: Optional[RateLimit] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    supports_streaming: bool = False
    supports_functions: bool = False
    supports_vision: bool = False

    def get_api_key(self, provider_info: Optional[ProviderInfo] = None) -> Optional[str]:
        """Get API key with fallback to provider defaults.

        Implements a two-level lookup: model-specific key takes precedence,
        then falls back to provider default.

        Args:
            provider_info: Optional provider configuration for fallback.

        Returns:
            API key string or None if not configured.
        """
        if self.api_key:
            return self.api_key
        if provider_info and provider_info.default_api_key:
            return provider_info.default_api_key
        return None

    def get_base_url(self, provider_info: Optional[ProviderInfo] = None) -> Optional[str]:
        """Get base URL with fallback to provider defaults.

        Supports model-specific endpoints for enterprise deployments.

        Args:
            provider_info: Optional provider configuration for fallback.

        Returns:
            Base URL string or None if using default endpoint.
        """
        if self.base_url:
            return self.base_url
        if provider_info and provider_info.base_url:
            return provider_info.base_url
        return None


# Request/Response schemas


@dataclass
class ChatRequest:
    """Universal chat request model.

    This schema defines a provider-agnostic interface for model requests.
    It captures the common parameters across all providers while allowing
    provider-specific extensions through the provider_params field.

    The design prioritizes the common case (simple prompt) while supporting
    advanced features when needed.

    Attributes:
        prompt: The input text to send to the model.
        context: Conversation history or system messages.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative).
        top_p: Nucleus sampling parameter.
        stop: Stop sequences to end generation.
        stream: Whether to stream the response.
        provider_params: Provider-specific parameters.

    Examples:
        Simple request:

        >>> request = ChatRequest(prompt="Hello, world!")

        Advanced request:

        >>> request = ChatRequest(
        ...     prompt="Write a haiku about programming",
        ...     max_tokens=50,
        ...     temperature=0.8,
        ...     stop=["\n\n"],
        ...     provider_params={"presence_penalty": 0.6}
        ... )

        With conversation context:

        >>> request = ChatRequest(
        ...     prompt="What did I just ask?",
        ...     context=[
        ...         {"role": "user", "content": "What is Python?"},
        ...         {"role": "assistant", "content": "Python is..."}
        ...     ]
        ... )
    """

    prompt: str
    context: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    provider_params: Optional[ProviderParams] = None

    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific format.

        This method is intentionally not implemented here, following the
        principle that format conversion belongs with the provider code
        that understands the target format.

        Args:
            provider: Target provider name.

        Raises:
            NotImplementedError: Always - providers handle conversion.
        """
        raise NotImplementedError("Providers should implement format conversion")


@dataclass
class ChatResponse:
    """Universal chat response model.

    This schema provides a consistent interface for model responses,
    abstracting away provider-specific response formats. It includes
    both the generated text and metadata for tracking and analysis.

    Attributes:
        data: The generated text response.
        usage: Token usage and cost information.
        model_id: Identifier of the model that generated this response.
        raw_output: Original provider response for advanced use cases.
        created_at: Timestamp of response generation.

    Properties:
        text: Alias for data (backward compatibility).

    Examples:
        Basic response:

        >>> response = ChatResponse(
        ...     data="Paris is the capital of France.",
        ...     model_id="gpt-4"
        ... )

        With usage tracking:

        >>> response = ChatResponse(
        ...     data="Here's a Python function...",
        ...     usage=UsageStats(
        ...         prompt_tokens=50,
        ...         completion_tokens=150,
        ...         cost_usd=0.006
        ...     ),
        ...     model_id="claude-3-opus"
        ... )
        >>> print(f"Cost: ${response.usage.cost_usd:.4f}")
        Cost: $0.0060
    """

    data: str
    usage: Optional[UsageStats] = None
    model_id: Optional[str] = None
    raw_output: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def text(self) -> str:
        """Alias for data for backward compatibility.

        Some users prefer 'text' over 'data'. This property ensures
        both work without duplicating storage.

        Returns:
            The generated text content.
        """
        return self.data


# Usage tracking schemas


@dataclass
class UsageStats:
    """Token usage statistics with cost calculation.

    This schema tracks resource consumption for cost management and
    optimization. It supports aggregation for summary statistics.

    The design includes both individual token counts and total cost,
    making it easy to understand the cost breakdown of requests.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated response.
        total_tokens: Total tokens (auto-calculated if not provided).
        cost_usd: Total cost in USD (calculated by registry).

    Examples:
        Track single request:

        >>> usage = UsageStats(
        ...     prompt_tokens=100,
        ...     completion_tokens=200,
        ...     cost_usd=0.009
        ... )
        >>> print(f"Cost per token: ${usage.cost_usd / usage.total_tokens:.6f}")

        Aggregate multiple requests:

        >>> total = UsageStats()
        >>> for response in responses:
        ...     total.add(response.usage)
        >>> print(f"Total cost: ${total.cost_usd:.2f}")

        Operator overloading:

        >>> usage1 = UsageStats(prompt_tokens=100, completion_tokens=50)
        >>> usage2 = UsageStats(prompt_tokens=200, completion_tokens=100)
        >>> combined = usage1 + usage2
        >>> print(f"Combined tokens: {combined.total_tokens}")
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None

    def __post_init__(self):
        """Ensure total_tokens is consistent.

        Auto-calculates total if not explicitly provided, maintaining
        data integrity without requiring manual calculation.
        """
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add(self, other: UsageStats) -> None:
        """Add another usage stats to this one in-place.

        Useful for accumulating usage across multiple requests.

        Args:
            other: Usage stats to add to this instance.

        Examples:
            >>> total = UsageStats()
            >>> total.add(UsageStats(prompt_tokens=100, completion_tokens=50))
            >>> total.add(UsageStats(prompt_tokens=200, completion_tokens=100))
            >>> print(total.total_tokens)  # 450
        """
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        if self.cost_usd is not None and other.cost_usd is not None:
            self.cost_usd += other.cost_usd
        elif other.cost_usd is not None:
            self.cost_usd = other.cost_usd

    def __add__(self, other: UsageStats) -> UsageStats:
        """Create new UsageStats by adding two together.

        Supports the + operator for intuitive aggregation.

        Args:
            other: Usage stats to add.

        Returns:
            New UsageStats with combined values.

        Examples:
            >>> day1 = UsageStats(prompt_tokens=1000, completion_tokens=2000)
            >>> day2 = UsageStats(prompt_tokens=1500, completion_tokens=2500)
            >>> total = day1 + day2
            >>> print(f"Total tokens: {total.total_tokens}")
        """
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=(self.cost_usd or 0) + (other.cost_usd or 0),
        )


@dataclass
class UsageRecord:
    """Single usage record with timestamp.

    Captures a point-in-time usage snapshot for historical tracking
    and analytics. Each record is immutable once created.

    Attributes:
        usage: The usage statistics for this record.
        timestamp: When this usage occurred (UTC).
        model_id: Which model was used.
        request_id: Optional correlation ID for request tracking.

    Examples:
        >>> record = UsageRecord(
        ...     usage=UsageStats(prompt_tokens=100, completion_tokens=200),
        ...     model_id="gpt-4",
        ...     request_id="req-12345"
        ... )
    """

    usage: UsageStats
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class UsageSummary:
    """Cumulative usage summary for a model.

    Provides aggregated statistics for cost analysis and optimization.
    This schema supports building dashboards and usage reports.

    Attributes:
        model_name: The model being tracked.
        total_usage: Aggregated usage statistics.
        request_count: Number of requests made.
        first_used: Timestamp of first usage.
        last_used: Timestamp of most recent usage.

    Examples:
        Build usage report:

        >>> summary = UsageSummary(model_name="gpt-4")
        >>> for record in usage_history:
        ...     summary.add_usage(record.usage)
        >>>
        >>> print(f"Model: {summary.model_name}")
        >>> print(f"Requests: {summary.request_count}")
        >>> print(f"Total cost: ${summary.total_usage.cost_usd:.2f}")
        >>> print(f"Avg tokens/request: "
        ...       f"{summary.total_usage.total_tokens / summary.request_count:.1f}")
    """

    model_name: str
    total_usage: UsageStats = field(default_factory=UsageStats)
    request_count: int = 0
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None

    def add_usage(self, usage: UsageStats) -> None:
        """Add a usage record to the summary.

        Updates all summary statistics including timestamps.

        Args:
            usage: Usage statistics to add to the summary.

        Examples:
            >>> summary = UsageSummary(model_name="claude-3-opus")
            >>> summary.add_usage(UsageStats(prompt_tokens=100,
            ...                              completion_tokens=200))
            >>> print(f"Total tokens: {summary.total_usage.total_tokens}")
        """
        self.total_usage.add(usage)
        self.request_count += 1

        now = datetime.utcnow()
        if self.first_used is None:
            self.first_used = now
        self.last_used = now


# Type aliases for clarity and better type hints throughout the codebase.
# These make function signatures more readable and self-documenting.
ModelID = str  # e.g., "gpt-4", "claude-3-opus"
ProviderName = str  # e.g., "openai", "anthropic"
ResponseData = Union[str, Dict[str, Any]]  # Flexible response format
