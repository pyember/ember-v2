"""Model cost configuration with hybrid approach.

This module provides default model costs and allows environment-based overrides
without complex configuration files.

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

import json
import os
from typing import Dict, Any


# Default model costs per 1000 tokens in USD
# These are reasonable defaults as of 2024, but can be overridden
DEFAULT_MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # OpenAI Models
    "gpt-4": {
        "input": 30.0,
        "output": 60.0,
        "context": 8192
    },
    "gpt-4-turbo": {
        "input": 10.0,
        "output": 30.0,
        "context": 128000
    },
    "gpt-4o": {
        "input": 5.0,
        "output": 15.0,
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
    
    The function returns default costs but allows overriding via:
    1. EMBER_MODEL_COSTS_JSON - Full JSON override
    2. EMBER_COST_<MODEL>_<FIELD> - Individual field overrides
    
    Examples:
        >>> # Default usage
        >>> costs = get_model_costs()
        >>> gpt4_input_cost = costs["gpt-4"]["input"]  # 30.0
        
        >>> # With environment override
        >>> os.environ["EMBER_COST_GPT4_INPUT"] = "25.0"
        >>> costs = get_model_costs()
        >>> gpt4_input_cost = costs["gpt-4"]["input"]  # 25.0
        
        >>> # With JSON override
        >>> os.environ["EMBER_MODEL_COSTS_JSON"] = '{"gpt-4": {"input": 20.0}}'
        >>> costs = get_model_costs()
        >>> gpt4_input_cost = costs["gpt-4"]["input"]  # 20.0
    
    Returns:
        Dict mapping model names to cost information (input, output, context).
        
    Note:
        Individual overrides (EMBER_COST_*) are applied after JSON overrides,
        so they take precedence.
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
            # Log but don't fail - use defaults
            import logging
            logging.warning(f"Invalid EMBER_MODEL_COSTS_JSON: {e}")
    
    # Then apply individual field overrides
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
    
    Args:
        model_id: The model identifier (e.g., "gpt-4", "claude-3-opus").
        
    Returns:
        Dict with "input", "output", and "context" fields.
        Returns default zero costs if model not found.
        
    Examples:
        >>> cost = get_model_cost("gpt-4")
        >>> print(f"Input: ${cost['input']}/1k tokens")
        Input: $30.0/1k tokens
    """
    costs = get_model_costs()
    
    # Check if model exists in costs
    if model_id in costs:
        return costs[model_id]
    
    # For unknown models, check for individual environment overrides
    default_cost = {
        "input": 0.0,
        "output": 0.0,
        "context": 4096  # Default context window
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


# Convenience function for backward compatibility
def get_model_pricing(model_id: str) -> tuple[float, float]:
    """Get input and output pricing for a model.
    
    Args:
        model_id: The model identifier.
        
    Returns:
        Tuple of (input_cost_per_thousand, output_cost_per_thousand).
        
    Examples:
        >>> input_cost, output_cost = get_model_pricing("gpt-4")
        >>> print(f"GPT-4 costs ${input_cost} per 1k input tokens")
        GPT-4 costs $30.0 per 1k input tokens
    """
    cost = get_model_cost(model_id)
    return cost["input"], cost["output"]