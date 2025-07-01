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
from typing import Dict, Any


# Default model costs per 1,000,000 tokens in USD.
# These reflect current pricing as of 2025. The structure prioritizes
# clarity over configuration - every cost is visible and predictable.
#
# Structure:
#   - input: Cost per 1,000,000 input tokens in USD
#   - output: Cost per 1,000,000 output tokens in USD  
#   - context: Maximum context window in tokens
#
# The constants are intentionally hardcoded rather than loaded from files,
# following the principle that configuration should be code when possible.
DEFAULT_MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # OpenAI Models - Updated pricing as of 2025
    "gpt-4": {
        "input": 2.00,  # $2.00 per 1M tokens
        "output": 8.00,  # $8.00 per 1M tokens
        "context": 8192
    },
    "gpt-4.1": {
        "input": 2.00,  # $2.00 per 1M tokens
        "output": 8.00,  # $8.00 per 1M tokens
        "context": 128000
    },
    "gpt-4.1-2025-04-14": {
        "input": 2.00,  # $2.00 per 1M tokens
        "output": 8.00,  # $8.00 per 1M tokens
        "context": 128000
    },
    "gpt-4.1-mini": {
        "input": 0.40,  # $0.40 per 1M tokens
        "output": 1.60,  # $1.60 per 1M tokens
        "context": 128000
    },
    "gpt-4.1-mini-2025-04-14": {
        "input": 0.40,  # $0.40 per 1M tokens
        "output": 1.60,  # $1.60 per 1M tokens
        "context": 128000
    },
    "gpt-4.1-nano": {
        "input": 0.10,  # $0.10 per 1M tokens
        "output": 0.40,  # $0.40 per 1M tokens
        "context": 128000
    },
    "gpt-4.1-nano-2025-04-14": {
        "input": 0.10,  # $0.10 per 1M tokens
        "output": 0.40,  # $0.40 per 1M tokens
        "context": 128000
    },
    "gpt-4.5-preview": {
        "input": 75.00,  # $75.00 per 1M tokens
        "output": 150.00,  # $150.00 per 1M tokens
        "context": 128000
    },
    "gpt-4.5-preview-2025-02-27": {
        "input": 75.00,  # $75.00 per 1M tokens
        "output": 150.00,  # $150.00 per 1M tokens
        "context": 128000
    },
    "gpt-4-turbo": {
        "input": 10.0,
        "output": 30.0,
        "context": 128000
    },
    "gpt-4o": {
        "input": 2.50,  # $2.50 per 1M tokens
        "output": 10.00,  # $10.00 per 1M tokens
        "context": 128000
    },
    "gpt-4o-2024-08-06": {
        "input": 2.50,  # $2.50 per 1M tokens
        "output": 10.00,  # $10.00 per 1M tokens
        "context": 128000
    },
    "gpt-4o-audio-preview": {
        "input": 2.50,  # $2.50 per 1M tokens
        "output": 10.00,  # $10.00 per 1M tokens
        "context": 128000
    },
    "gpt-4o-audio-preview-2024-12-17": {
        "input": 2.50,  # $2.50 per 1M tokens
        "output": 10.00,  # $10.00 per 1M tokens
        "context": 128000
    },
    "gpt-4o-realtime-preview": {
        "input": 5.00,  # $5.00 per 1M tokens
        "output": 20.00,  # $20.00 per 1M tokens
        "context": 128000
    },
    "gpt-4o-realtime-preview-2025-06-03": {
        "input": 5.00,  # $5.00 per 1M tokens
        "output": 20.00,  # $20.00 per 1M tokens
        "context": 128000
    },
    "gpt-4o-mini": {
        "input": 0.15,  # $0.15 per 1M tokens
        "output": 0.60,  # $0.60 per 1M tokens
        "context": 128000
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.15,  # $0.15 per 1M tokens
        "output": 0.60,  # $0.60 per 1M tokens
        "context": 128000
    },
    "gpt-4o-mini-audio-preview": {
        "input": 0.15,  # $0.15 per 1M tokens
        "output": 0.60,  # $0.60 per 1M tokens
        "context": 128000
    },
    "gpt-4o-mini-audio-preview-2024-12-17": {
        "input": 0.15,  # $0.15 per 1M tokens
        "output": 0.60,  # $0.60 per 1M tokens
        "context": 128000
    },
    "gpt-4o-mini-realtime-preview": {
        "input": 0.60,  # $0.60 per 1M tokens
        "output": 2.40,  # $2.40 per 1M tokens
        "context": 128000
    },
    "gpt-4o-mini-realtime-preview-2024-12-17": {
        "input": 0.60,  # $0.60 per 1M tokens
        "output": 2.40,  # $2.40 per 1M tokens
        "context": 128000
    },
    "gpt-4o-mini-search-preview": {
        "input": 0.15,  # $0.15 per 1M tokens
        "output": 0.60,  # $0.60 per 1M tokens
        "context": 128000
    },
    "gpt-4o-mini-search-preview-2025-03-11": {
        "input": 0.15,  # $0.15 per 1M tokens
        "output": 0.60,  # $0.60 per 1M tokens
        "context": 128000
    },
    "gpt-4o-search-preview": {
        "input": 2.50,  # $2.50 per 1M tokens
        "output": 10.00,  # $10.00 per 1M tokens
        "context": 128000
    },
    "gpt-4o-search-preview-2025-03-11": {
        "input": 2.50,  # $2.50 per 1M tokens
        "output": 10.00,  # $10.00 per 1M tokens
        "context": 128000
    },
    "gpt-3.5-turbo": {
        "input": 0.5,
        "output": 1.5,
        "context": 16385
    },
    "gpt-3.5-turbo-16k": {
        "input": 3.0,
        "output": 4.0,
        "context": 16385
    },
    
    # OpenAI o-series models
    "o1": {
        "input": 15.00,  # $15.00 per 1M tokens
        "output": 60.00,  # $60.00 per 1M tokens
        "context": 128000
    },
    "o1-2024-12-17": {
        "input": 15.00,  # $15.00 per 1M tokens
        "output": 60.00,  # $60.00 per 1M tokens
        "context": 128000
    },
    "o1-pro": {
        "input": 150.00,  # $150.00 per 1M tokens
        "output": 600.00,  # $600.00 per 1M tokens
        "context": 128000
    },
    "o1-pro-2025-03-19": {
        "input": 150.00,  # $150.00 per 1M tokens
        "output": 600.00,  # $600.00 per 1M tokens
        "context": 128000
    },
    "o1-mini": {
        "input": 1.10,  # $1.10 per 1M tokens
        "output": 4.40,  # $4.40 per 1M tokens
        "context": 128000
    },
    "o1-mini-2024-09-12": {
        "input": 1.10,  # $1.10 per 1M tokens
        "output": 4.40,  # $4.40 per 1M tokens
        "context": 128000
    },
    "o3": {
        "input": 2.00,  # $2.00 per 1M tokens
        "output": 8.00,  # $8.00 per 1M tokens
        "context": 128000
    },
    "o3-2025-04-16": {
        "input": 2.00,  # $2.00 per 1M tokens
        "output": 8.00,  # $8.00 per 1M tokens
        "context": 128000
    },
    "o3-pro": {
        "input": 20.00,  # $20.00 per 1M tokens
        "output": 80.00,  # $80.00 per 1M tokens
        "context": 128000
    },
    "o3-pro-2025-06-10": {
        "input": 20.00,  # $20.00 per 1M tokens
        "output": 80.00,  # $80.00 per 1M tokens
        "context": 128000
    },
    "o3-mini": {
        "input": 1.10,  # $1.10 per 1M tokens
        "output": 4.40,  # $4.40 per 1M tokens
        "context": 128000
    },
    "o3-mini-2025-01-31": {
        "input": 1.10,  # $1.10 per 1M tokens
        "output": 4.40,  # $4.40 per 1M tokens
        "context": 128000
    },
    "o3-deep-research": {
        "input": 10.00,  # $10.00 per 1M tokens
        "output": 40.00,  # $40.00 per 1M tokens
        "context": 128000
    },
    "o3-deep-research-2025-06-26": {
        "input": 10.00,  # $10.00 per 1M tokens
        "output": 40.00,  # $40.00 per 1M tokens
        "context": 128000
    },
    "o4-mini": {
        "input": 1.10,  # $1.10 per 1M tokens
        "output": 4.40,  # $4.40 per 1M tokens
        "context": 128000
    },
    "o4-mini-2025-04-16": {
        "input": 1.10,  # $1.10 per 1M tokens
        "output": 4.40,  # $4.40 per 1M tokens
        "context": 128000
    },
    "o4-mini-deep-research": {
        "input": 2.00,  # $2.00 per 1M tokens
        "output": 8.00,  # $8.00 per 1M tokens
        "context": 128000
    },
    "o4-mini-deep-research-2025-06-26": {
        "input": 2.00,  # $2.00 per 1M tokens
        "output": 8.00,  # $8.00 per 1M tokens
        "context": 128000
    },
    
    # Anthropic Models
    "claude-3-opus": {
        "input": 15.0,
        "output": 75.0,
        "context": 200000
    },
    "claude-3-sonnet": {
        "input": 3.0,
        "output": 15.0,
        "context": 200000
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
        "context": 200000
    },
    "claude-2.1": {
        "input": 8.0,
        "output": 24.0,
        "context": 200000
    },
    
    # Google Models
    "gemini-pro": {
        "input": 0.5,
        "output": 1.5,
        "context": 32768
    },
    "gemini-pro-vision": {
        "input": 0.5,
        "output": 1.5,
        "context": 32768
    },
    "gemini-1.5-pro": {
        "input": 3.5,
        "output": 10.5,
        "context": 1048576
    },
    
    # Meta/Llama Models (via various providers)
    "llama-2-70b": {
        "input": 0.7,
        "output": 0.9,
        "context": 4096
    },
    "llama-2-13b": {
        "input": 0.3,
        "output": 0.4,
        "context": 4096
    },
    "llama-2-7b": {
        "input": 0.2,
        "output": 0.2,
        "context": 4096
    },
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
                    "input": float,   # Cost per 1M input tokens
                    "output": float,  # Cost per 1M output tokens  
                    "context": int    # Max context window
                }
            }
    
    Examples:
        Default usage:
        
        >>> costs = get_model_costs()
        >>> gpt4_cost = costs["gpt-4"]
        >>> print(f"Input: ${gpt4_cost['input']}/1M tokens")
        Input: $2.0/1M tokens
        
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
        - Costs are in USD per 1,000,000 tokens
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
                    "".join(model_parts),   # gpt4
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
                "input": float,   # USD per 1M input tokens
                "output": float,  # USD per 1M output tokens
                "context": int    # Maximum context window
            }
        
        For unknown models, returns zero costs with 4096 context.
        
    Examples:
        Known model:
        
        >>> cost = get_model_cost("gpt-4")
        >>> print(f"GPT-4 input: ${cost['input']}/1M tokens")
        GPT-4 input: $2.0/1M tokens
        
        Unknown model with override:
        
        >>> # export EMBER_COST_MYFINETUNED_INPUT=2.0
        >>> cost = get_model_cost("my-finetuned")
        >>> print(cost["input"])  # 2.0
        
        Calculate request cost:
        
        >>> cost = get_model_cost("claude-3-opus")
        >>> prompt_tokens = 1500
        >>> completion_tokens = 500
        >>> total_cost = (
        ...     (prompt_tokens / 1_000_000) * cost["input"] +
        ...     (completion_tokens / 1_000_000) * cost["output"]
        ... )
        >>> print(f"Request cost: ${total_cost:.6f}")
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
        "context": 4096  # Conservative default context
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
        Tuple of (input_cost_per_1M, output_cost_per_1M) in USD.
        
    Examples:
        Basic usage:
        
        >>> input_cost, output_cost = get_model_pricing("gpt-4")
        >>> print(f"GPT-4: ${input_cost}/${output_cost} per 1M tokens")
        GPT-4: $2.0/$8.0 per 1M tokens
        
        Cost calculation:
        
        >>> input_cost, output_cost = get_model_pricing("claude-3-haiku")
        >>> total = (1000/1_000_000) * input_cost + (500/1_000_000) * output_cost
        >>> print(f"1k input + 500 output = ${total:.6f}")
        1k input + 500 output = $0.000875
    """
    cost = get_model_cost(model_id)
    return cost["input"], cost["output"]