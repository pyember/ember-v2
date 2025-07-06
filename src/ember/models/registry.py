"""Model registry with integrated service functionality.

This module implements a thread-safe registry that manages language model instances,
combining registry and service functionality into a cohesive component.

The registry follows several key design principles:
1. Lazy instantiation - Models are created only when first requested
2. Thread safety - Single lock proven sufficient for all operations
3. Integrated functionality - Cost calculation and usage tracking built-in
4. Direct instantiation - No complex factories or dependency injection

The implementation balances SOLID principles with radical simplicity, as advocated
by Dean, Ghemawat, Jobs, Brockman, Ritchie, Knuth, Carmack, and Martin.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from ember._internal.exceptions import (
    ModelNotFoundError,
    ModelProviderError,
    ProviderAPIError,
)
from ember.models.costs import get_model_cost
from ember.models.providers import (
    get_provider_class,
    resolve_model_id,
)
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe registry with integrated service functionality.

    This registry provides a unified interface for managing language model instances,
    handling everything from instantiation to usage tracking in a single component.

    The design eliminates the traditional service layer by integrating essential
    functionality directly into the registry, reducing indirection while maintaining
    clean separation of concerns.

    Thread Safety:
        Uses a single lock for all operations. Performance profiling showed this
        approach is superior to per-model locks for typical usage patterns.

    Lazy Instantiation:
        Models are created only when first requested, reducing startup time and
        memory usage for unused models.

    Cost Integration:
        Automatically calculates costs based on token usage, using either hardcoded
        defaults or environment-based overrides.

    Examples:
        Basic usage:

        >>> registry = ModelRegistry()
        >>> response = registry.invoke_model("gpt-4", "Hello, world!")
        >>> print(response.data)
        Hello! How can I help you today?

        With parameters:

        >>> response = registry.invoke_model(
        ...     "claude-3-opus",
        ...     "Write a haiku about programming",
        ...     temperature=0.8,
        ...     max_tokens=50
        ... )
        >>> print(f"Cost: ${response.usage.cost_usd:.4f}")

        Async usage:

        >>> async def generate():
        ...     response = await registry.invoke_model_async(
        ...         "gpt-4", "Explain async programming"
        ...     )
        ...     return response.data

        Usage tracking:

        >>> # After multiple invocations
        >>> summary = registry.get_usage_summary("gpt-4")
        >>> print(f"Total tokens used: {summary.total_tokens}")
        >>> print(f"Total cost: ${summary.cost_usd:.2f}")

    Attributes:
        _models: Cache of instantiated provider instances.
        _lock: Thread synchronization primitive.
        _usage_records: Token usage history by model.
        _metrics: Optional metrics collectors for monitoring.
    """

    def __init__(
        self, metrics: Optional[Dict[str, Any]] = None, context: Optional[Any] = None
    ) -> None:
        """Initialize the model registry.

        Args:
            metrics: Optional dictionary of metric collectors. Keys should be
                metric names (e.g., 'invocation_duration', 'model_invocations').
                Values should be collector objects with appropriate methods.
            context: Optional EmberContext for dependency injection. If provided,
                will use context for configuration and credentials.

        Example:
            >>> # With Prometheus metrics
            >>> from prometheus_client import Counter, Histogram
            >>> metrics = {
            ...     'model_invocations': Counter('model_calls', 'Total model calls'),
            ...     'invocation_duration': Histogram('model_duration', 'Call duration')
            ... }
            >>> registry = ModelRegistry(metrics=metrics)

            >>> # With context injection
            >>> from ember._internal.context import EmberContext
            >>> ctx = EmberContext()
            >>> registry = ModelRegistry(context=ctx)
        """
        self._models: Dict[str, BaseProvider] = {}
        self._lock = threading.Lock()
        self._usage_records: Dict[str, List[UsageStats]] = {}
        self._metrics = metrics or {}
        self._logger = logger
        self._context = context

    def get_model(self, model_id: str) -> BaseProvider:
        """Get or create a model provider instance.

        This method implements lazy instantiation with thread-safe caching.
        Models are created only on first request and cached for subsequent use.

        The double-checked locking pattern ensures thread safety while minimizing
        lock contention for the common case of accessing cached models.

        Args:
            model_id: Model identifier. Can be:
                - Simple name: "gpt-4", "claude-3-opus"
                - Explicit provider: "openai/gpt-4", "anthropic/claude-3"

        Returns:
            Provider instance ready for invocation.

        Raises:
            ModelNotFoundError: If the model or provider cannot be found.
            ModelProviderError: If API key is missing or invalid.

        Examples:
            >>> registry = ModelRegistry()
            >>>
            >>> # Simple model name (provider inferred)
            >>> model = registry.get_model("gpt-4")
            >>>
            >>> # Explicit provider specification
            >>> model = registry.get_model("openai/gpt-4")
            >>>
            >>> # Handles errors gracefully
            >>> try:
            ...     model = registry.get_model("unknown-model")
            >>> except ModelNotFoundError as e:
            ...     print(f"Model not found: {e}")
        """
        # Fast path: check cache without lock
        if model_id in self._models:
            return self._models[model_id]

        # Slow path: create model with lock
        with self._lock:
            # Double-check pattern after acquiring lock
            if model_id in self._models:
                return self._models[model_id]

            # Create and cache the model
            try:
                model = self._create_model(model_id)
                self._models[model_id] = model
                return model
            except Exception as e:
                self._logger.error(f"Failed to create model {model_id}: {e}")
                raise

    def _create_model(self, model_id: str) -> BaseProvider:
        """Create a new model provider instance.

        Internal method that handles provider resolution, API key retrieval,
        and provider instantiation.

        Args:
            model_id: Model identifier to instantiate.

        Returns:
            New provider instance.

        Raises:
            ModelNotFoundError: If provider cannot be determined or found.
            ModelProviderError: If API key is missing.
        """
        # Resolve provider name from model ID
        provider_name, model_name = resolve_model_id(model_id)

        if provider_name == "unknown":
            from ember.models.catalog import list_available_models

            available = list_available_models()
            raise ModelNotFoundError(
                f"Cannot determine provider for model '{model_id}'. "
                f"Available models: {', '.join(sorted(available))}",
                context={"model_id": model_id, "available_models": available},
            )

        # Get provider implementation class
        try:
            provider_class = get_provider_class(provider_name)
        except ValueError as e:
            raise ModelNotFoundError(
                f"Provider '{provider_name}' not found",
                context={"provider": provider_name, "model_id": model_id},
            ) from e

        # Retrieve API key from environment
        api_key = self._get_api_key(provider_name)
        if not api_key:
            env_var = f"{provider_name.upper()}_API_KEY"

            # Try interactive setup if in TTY
            from ember.core.setup_launcher import launch_setup_if_needed

            api_key = launch_setup_if_needed(provider_name, env_var, model_id)

            if not api_key:
                # User didn't provide key or not in interactive mode
                from ember.core.setup_launcher import format_non_interactive_error

                raise ModelProviderError(
                    format_non_interactive_error(provider_name, env_var, model_id),
                    context={"model_id": model_id, "provider": provider_name},
                )

        # Instantiate provider
        try:
            provider = provider_class(api_key=api_key)
            return provider
        except ValueError as e:
            # Handle missing API key error from BaseProvider
            if "API key required" in str(e):
                # Re-raise as ModelProviderError for better error handling
                from ember.core.setup_launcher import format_non_interactive_error

                error_msg = format_non_interactive_error(
                    provider_name, f"{provider_name.upper()}_API_KEY", model_id
                )
                raise ModelProviderError(
                    error_msg, context={"model_id": model_id, "provider": provider_name}
                ) from e
            else:
                # Other ValueError, re-raise as ModelNotFoundError
                raise ModelNotFoundError(
                    f"Failed to instantiate provider for model {model_id}",
                    context={"model_id": model_id, "provider": provider_name},
                ) from e
        except Exception as e:
            self._logger.error(
                f"Exception instantiating provider {provider_name} for model {model_id}: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise ModelNotFoundError(
                f"Failed to instantiate provider for model {model_id}: "
                f"{type(e).__name__}: {str(e)}",
                context={"model_id": model_id, "provider": provider_name},
            ) from e

    def invoke_model(self, model_id: str, prompt: str, **kwargs: Any) -> ChatResponse:
        """Invoke a model with integrated tracking.

        This method combines model invocation with automatic cost calculation,
        usage tracking, and optional metrics recording.

        Args:
            model_id: Model identifier (e.g., "gpt-4", "claude-3-opus").
            prompt: The prompt text to send to the model.
            **kwargs: Additional parameters passed to the model:
                - temperature: Sampling temperature (0.0 to 2.0)
                - max_tokens: Maximum response length
                - top_p: Nucleus sampling parameter
                - stop: Stop sequences
                - context: System message or context
                - Any provider-specific parameters

        Returns:
            ChatResponse containing the model output, usage statistics,
            and calculated costs.

        Raises:
            ProviderAPIError: If the model invocation fails.
            ModelNotFoundError: If the model cannot be found.
            ModelProviderError: If API key is missing.

        Examples:
            Basic invocation:

            >>> response = registry.invoke_model("gpt-4", "Hello!")
            >>> print(response.data)

            With parameters:

            >>> response = registry.invoke_model(
            ...     "claude-3-opus",
            ...     "Write a story",
            ...     temperature=0.9,
            ...     max_tokens=500,
            ...     context="You are a creative writer"
            ... )

            Error handling:

            >>> try:
            ...     response = registry.invoke_model("gpt-4", prompt)
            ... except ProviderAPIError as e:
            ...     if e.context.get("error_type") == "rate_limit":
            ...         print("Rate limited, please retry later")
        """
        start_time = time.time()

        # Get the model provider
        model = self.get_model(model_id)

        # Record invocation metric if available
        if "model_invocations" in self._metrics:
            self._metrics["model_invocations"].labels(model_id=model_id).inc()

        try:
            # Invoke with optional metrics timing
            if "invocation_duration" in self._metrics:
                with self._metrics["invocation_duration"].labels(model_id=model_id).time():
                    response = model.complete(prompt, model_id, **kwargs)
            else:
                response = model.complete(prompt, model_id, **kwargs)

            # Calculate and add cost if usage is available
            if response.usage:
                cost = self._calculate_cost(model_id, response.usage)
                response.usage.cost_usd = cost

                # Track usage for summaries
                self._track_usage(model_id, response.usage)

                # Track cost accuracy if we have actual costs
                from ember.models.cost_tracker import track_usage

                track_usage(response.usage, model_id)

            # Ensure model_id is in response
            if not response.model_id:
                response.model_id = model_id

            return response

        except Exception as e:
            self._logger.exception(f"Error invoking model '{model_id}'")
            raise ProviderAPIError(
                f"Error invoking model {model_id}", context={"model_id": model_id}
            ) from e
        finally:
            duration = time.time() - start_time
            self._logger.debug(f"Model {model_id} invocation took {duration:.3f}s")

    async def invoke_model_async(self, model_id: str, prompt: str, **kwargs: Any) -> ChatResponse:
        """Asynchronously invoke a model.

        Supports both async-native providers and sync providers (run in thread pool).
        Includes the same tracking and cost calculation as sync invocation.

        Args:
            model_id: Model identifier.
            prompt: The prompt text.
            **kwargs: Additional model parameters.

        Returns:
            ChatResponse with costs calculated.

        Raises:
            ProviderAPIError: If invocation fails.

        Examples:
            >>> async def generate_async():
            ...     response = await registry.invoke_model_async(
            ...         "gpt-4",
            ...         "Explain async/await in Python"
            ...     )
            ...     return response.data
            ...
            >>> # Run with asyncio
            >>> import asyncio
            >>> result = asyncio.run(generate_async())
        """
        model = self.get_model(model_id)

        try:
            # Check if provider supports async natively
            if asyncio.iscoroutinefunction(model.complete):
                response = await model.complete(prompt, model_id, **kwargs)
            else:
                # Run sync model in thread pool to avoid blocking
                response = await asyncio.to_thread(model.complete, prompt, model_id, **kwargs)

            # Calculate cost and track usage
            if response.usage:
                cost = self._calculate_cost(model_id, response.usage)
                response.usage.cost_usd = cost
                self._track_usage(model_id, response.usage)

            return response

        except Exception as e:
            self._logger.exception(f"Async error invoking model '{model_id}'")
            raise ProviderAPIError(
                f"Async error invoking model {model_id}", context={"model_id": model_id}
            ) from e

    def _calculate_cost(self, model_id: str, usage: UsageStats) -> float:
        """Calculate cost for token usage.

        Uses cost data from the costs module, which supports both hardcoded
        defaults and environment-based overrides.

        Args:
            model_id: Model identifier.
            usage: Token usage statistics.

        Returns:
            Total cost in USD.
        """
        cost_info = get_model_cost(model_id)
        if not cost_info:
            return 0.0

        # Calculate input and output costs separately
        input_cost = (usage.prompt_tokens / 1000.0) * cost_info.get("input", 0.0)
        output_cost = (usage.completion_tokens / 1000.0) * cost_info.get("output", 0.0)

        return round(input_cost + output_cost, 6)  # Round to 6 decimal places

    def _track_usage(self, model_id: str, usage: UsageStats) -> None:
        """Track usage statistics for a model.

        Thread-safe recording of usage data for later analysis.

        Args:
            model_id: Model identifier.
            usage: Usage statistics to record.
        """
        with self._lock:
            if model_id not in self._usage_records:
                self._usage_records[model_id] = []
            self._usage_records[model_id].append(usage)

            # Limit history to last 1000 records per model
            if len(self._usage_records[model_id]) > 1000:
                self._usage_records[model_id] = self._usage_records[model_id][-1000:]

    def get_usage_summary(self, model_id: str) -> Optional[UsageStats]:
        """Get cumulative usage statistics for a model.

        Aggregates all recorded usage for the specified model into a single
        summary.

        Args:
            model_id: Model identifier.

        Returns:
            Aggregated usage statistics, or None if no usage recorded.

        Examples:
            >>> # After several invocations
            >>> summary = registry.get_usage_summary("gpt-4")
            >>> if summary:
            ...     print(f"Total tokens: {summary.total_tokens}")
            ...     print(f"Total cost: ${summary.cost_usd:.2f}")
            ...     print(f"Average tokens per request: "
            ...           f"{summary.total_tokens / len(records):.1f}")
        """
        with self._lock:
            records = self._usage_records.get(model_id, [])
            if not records:
                return None

            # Aggregate all usage records
            total = UsageStats()
            for record in records:
                total.add(record)
            return total

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider.

        Uses context if available, otherwise falls back to direct environment
        and credential file checks.

        Args:
            provider: Provider name (e.g., "openai", "anthropic").

        Returns:
            API key string or None if not found.
        """
        # Use context if available (dependency injection)
        if self._context:
            env_var = f"{provider.upper()}_API_KEY"
            return self._context.get_credential(provider, env_var)

        # Fall back to direct checks if no context
        # Build list of possible environment variable names
        env_vars = [
            f"{provider.upper()}_API_KEY",  # Standard format
            f"EMBER_{provider.upper()}_API_KEY",  # Ember-specific
        ]

        # Add provider-specific variations
        if provider == "openai":
            env_vars.append("OPENAI_API_KEY")
        elif provider == "anthropic":
            env_vars.append("ANTHROPIC_API_KEY")
        elif provider in ["google", "deepmind"]:
            env_vars.extend(["GOOGLE_API_KEY", "GEMINI_API_KEY"])

        # Check each possible variable
        for var in env_vars:
            value = os.getenv(var)
            if value:
                return value

        # Check credentials file (like AWS CLI, gcloud, etc.)
        try:
            from ember.core.credentials import get_api_key

            api_key = get_api_key(provider, f"{provider.upper()}_API_KEY")
            if api_key:
                return api_key
        except ImportError:
            # Credentials module not available
            pass

        return None

    def list_models(self) -> List[str]:
        """List all cached model IDs.

        Returns:
            List of model IDs that have been instantiated and cached.

        Example:
            >>> registry.invoke_model("gpt-4", "test")
            >>> registry.invoke_model("claude-3-opus", "test")
            >>> print(registry.list_models())
            ['gpt-4', 'claude-3-opus']
        """
        return list(self._models.keys())

    def clear_cache(self) -> None:
        """Clear all cached models.

        Forces re-instantiation on next use. Useful for testing or when
        provider configurations change.

        Example:
            >>> registry.clear_cache()
            >>> # All models will be recreated on next use
        """
        with self._lock:
            self._models.clear()
            self._logger.info("Model cache cleared")
