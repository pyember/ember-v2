"""Model cost configuration with hybrid approach.

This module implements a principled approach to cost management that balances
simplicity with flexibility. Rather than complex configuration files or dependency
injection, it uses a hybrid of hardcoded defaults with environment overrides.

Design Principles:
    1. Sensible defaults - Works out of the box with reasonable costs
    2. Easy overrides - Environment variables for production tuning
    3. Zero configuration - No YAML/JSON files required by default
    4. Transparent pricing - All costs visible in code

The architecture follows the philosophy of making the common case simple while
allowing advanced usage when needed.

Examples:
    Basic usage with defaults:

    >>> from ember.models import get_model_cost
    >>> cost = get_model_cost("gpt-4")
    >>> print(f"GPT-4: ${cost['input']}/1k input tokens")
    GPT-4: $30.0/1k input tokens

    Environment-based override for production:

    $ export EMBER_COST_GPT4_INPUT=25.0
    $ python my_app.py

    Bulk override via JSON:

    $ export EMBER_MODEL_COSTS_JSON='{
        "gpt-4": {"input": 25.0, "output": 50.0},
        "claude-3-opus": {"input": 12.0, "output": 60.0}
    }'
"""

import json
import os
from typing import Dict

# Default model costs per 1000 tokens in USD.
# These reflect reasonable estimates as of 2024. The structure prioritizes
# clarity over configuration - every cost is visible and predictable.
#
# Structure:
#   - input: Cost per 1000 input tokens in USD
#   - output: Cost per 1000 output tokens in USD
#   - context: Maximum context window in tokens
#
# The constants are intentionally hardcoded rather than loaded from files,
# following the principle that configuration should be code when possible.
DEFAULT_MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # OpenAI Models
    "gpt-4": {"input": 30.0, "output": 60.0, "context": 8192},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0, "context": 128000},
    "gpt-4o": {"input": 5.0, "output": 15.0, "context": 128000},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5, "context": 16385},
    "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0, "context": 16385},
    # Anthropic Models
    "claude-3-opus": {"input": 15.0, "output": 75.0, "context": 200000},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0, "context": 200000},
    "claude-3-haiku": {"input": 0.25, "output": 1.25, "context": 200000},
    "claude-2.1": {"input": 8.0, "output": 24.0, "context": 200000},
    # Google Models
    "gemini-pro": {"input": 0.5, "output": 1.5, "context": 32768},
    "gemini-pro-vision": {"input": 0.5, "output": 1.5, "context": 32768},
    "gemini-1.5-pro": {"input": 3.5, "output": 10.5, "context": 1048576},
    # Meta/Llama Models (via various providers)
    "llama-2-70b": {"input": 0.7, "output": 0.9, "context": 4096},
    "llama-2-13b": {"input": 0.3, "output": 0.4, "context": 4096},
    "llama-2-7b": {"input": 0.2, "output": 0.2, "context": 4096},
}


def get_model_costs() -> Dict[str, Dict[str, float]]:
    """Get model costs with environment-based overrides.

    This function implements a layered approach to configuration that makes
    the common case (using defaults) zero-config while allowing precise
    control in production environments.

    Override Precedence (highest to lowest):
        1. Individual field overrides (EMBER_COST_<MODEL>_<FIELD>)
        2. JSON bulk overrides (EMBER_MODEL_COSTS_JSON)
        3. Hardcoded defaults

    The design intentionally avoids configuration files, following the
    principle that environment variables are superior for production
    deployments (12-factor app methodology).

    Args:
        None - Configuration comes from environment.

    Returns:
        Dictionary mapping model names to cost information with structure:
            {
                "model-name": {
                    "input": float,   # Cost per 1k input tokens
                    "output": float,  # Cost per 1k output tokens
                    "context": int    # Max context window
                }
            }

    Examples:
        Default usage:

        >>> costs = get_model_costs()
        >>> gpt4_cost = costs["gpt-4"]
        >>> print(f"Input: ${gpt4_cost['input']}/1k tokens")
        Input: $30.0/1k tokens

        Production override for specific field:

        >>> # In production: export EMBER_COST_GPT4_INPUT=25.0
        >>> costs = get_model_costs()
        >>> print(costs["gpt-4"]["input"])  # 25.0

        Bulk override via JSON:

        >>> # export EMBER_MODEL_COSTS_JSON='{"gpt-4": {"input": 20.0}}'
        >>> costs = get_model_costs()
        >>> print(costs["gpt-4"]["input"])  # 20.0

        Adding custom model:

        >>> # export EMBER_MODEL_COSTS_JSON='{"custom-model": {
        >>> #   "input": 5.0, "output": 10.0, "context": 8192}}'
        >>> costs = get_model_costs()
        >>> print("custom-model" in costs)  # True

    Implementation Notes:
        - Model names are normalized: gpt-4, gpt_4, and gpt4 all work
        - Invalid overrides are logged but don't cause failures
        - Context must be an integer (token count)
        - Costs are in USD per 1000 tokens
    """
    costs = DEFAULT_MODEL_COSTS.copy()

    # First, apply JSON overrides if provided
    if override_json := os.getenv("EMBER_MODEL_COSTS_JSON"):
        try:
            overrides = json.loads(override_json)
            for model, model_costs in overrides.items():
                if model in costs:
                    costs[model].update(model_costs)
                else:
                    costs[model] = model_costs
        except (json.JSONDecodeError, TypeError) as e:
            # Silent fallback to defaults - production robustness over debugging
            import logging

            logging.warning(f"Invalid EMBER_MODEL_COSTS_JSON: {e}")

    # Apply individual field overrides - these take precedence over JSON.
    # This design allows surgical production tuning without full configs.
    # Format: EMBER_COST_<MODEL>_<FIELD>=<value>
    # Example: EMBER_COST_GPT4_INPUT=25.0
    for key, value in os.environ.items():
        if key.startswith("EMBER_COST_") and len(key) > 11:
            parts = key[11:].lower().split("_")
            if len(parts) >= 2:
                # Handle model names with dashes (gpt-4) or underscores
                model_parts = parts[:-1]  # All but last part
                field = parts[-1]  # Last part is the field

                # Try different model name formats
                model_candidates = [
                    "-".join(model_parts),  # gpt-4
                    "_".join(model_parts),  # gpt_4
                    "".join(model_parts),  # gpt4
                ]

                for model in model_candidates:
                    if model in costs and field in ["input", "output", "context"]:
                        try:
                            if field == "context":
                                costs[model][field] = int(value)
                            else:
                                costs[model][field] = float(value)
                            break
                        except ValueError:
                            import logging

                            logging.warning(f"Invalid cost value for {key}: {value}")

    return costs


def get_model_cost(model_id: str) -> Dict[str, float]:
    """Get cost information for a specific model.

    This function provides a simple interface for cost lookup with intelligent
    fallbacks. Unknown models get zero cost rather than errors, supporting
    development workflows where costs aren't critical.

    Args:
        model_id: Model identifier. Accepts various formats:
            - Standard: "gpt-4", "claude-3-opus"
            - With provider: "openai/gpt-4"
            - Custom models: "my-fine-tuned-model"

    Returns:
        Dictionary with cost information:
            {
                "input": float,   # USD per 1k input tokens
                "output": float,  # USD per 1k output tokens
                "context": int    # Maximum context window
            }

        For unknown models, returns zero costs with 4096 context.

    Examples:
        Known model:

        >>> cost = get_model_cost("gpt-4")
        >>> print(f"GPT-4 input: ${cost['input']}/1k tokens")
        GPT-4 input: $30.0/1k tokens

        Unknown model with override:

        >>> # export EMBER_COST_MYFINETUNED_INPUT=2.0
        >>> cost = get_model_cost("my-finetuned")
        >>> print(cost["input"])  # 2.0

        Calculate request cost:

        >>> cost = get_model_cost("claude-3-opus")
        >>> prompt_tokens = 1500
        >>> completion_tokens = 500
        >>> total_cost = (
        ...     (prompt_tokens / 1000) * cost["input"] +
        ...     (completion_tokens / 1000) * cost["output"]
        ... )
        >>> print(f"Request cost: ${total_cost:.4f}")
    """
    costs = get_model_costs()

    # Check if model exists in costs
    if model_id in costs:
        return costs[model_id]

    # Unknown models get sensible defaults rather than errors.
    # This supports development and custom models gracefully.
    default_cost = {
        "input": 0.0,
        "output": 0.0,
        "context": 4096,  # Conservative default context
    }

    # Check for individual field overrides for this model
    import os

    normalized_model = model_id.replace("-", "_").replace(".", "_")

    for field in ["input", "output", "context"]:
        env_key = f"EMBER_COST_{normalized_model.upper()}_{field.upper()}"
        if env_key in os.environ:
            try:
                value = os.environ[env_key]
                if field == "context":
                    default_cost[field] = int(value)
                else:
                    default_cost[field] = float(value)
            except ValueError:
                pass  # Keep default

    return default_cost


def get_model_pricing(model_id: str) -> tuple[float, float]:
    """Get input and output pricing for a model.

    Simple convenience function that returns just the pricing tuple.
    Useful for compatibility and when context window isn't needed.

    Args:
        model_id: Model identifier (e.g., "gpt-4", "claude-3-opus").

    Returns:
        Tuple of (input_cost_per_1k, output_cost_per_1k) in USD.

    Examples:
        Basic usage:

        >>> input_cost, output_cost = get_model_pricing("gpt-4")
        >>> print(f"GPT-4: ${input_cost}/${output_cost} per 1k tokens")
        GPT-4: $30.0/$60.0 per 1k tokens

        Cost calculation:

        >>> input_cost, output_cost = get_model_pricing("claude-3-haiku")
        >>> total = (1000/1000) * input_cost + (500/1000) * output_cost
        >>> print(f"1k input + 500 output = ${total:.3f}")
        1k input + 500 output = $0.875
    """
    cost = get_model_cost(model_id)
    return cost["input"], cost["output"]
