"""Simple, principled pricing system.

Following CLAUDE.md principles:
- One obvious way to do things
- No magic, explicit behavior
- Configuration as code
"""

import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional
from functools import lru_cache


class Pricing:
    """Dead simple pricing system.
    
    All prices in USD per million tokens. No unit conversions.
    No complex overrides. Just clean, simple pricing data.
    """
    
    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = yaml_path or Path(__file__).parent / "pricing.yaml"
        self._data = self._load_yaml()
    
    @lru_cache(maxsize=1)
    def _load_yaml(self) -> Dict[str, Dict[str, float]]:
        """Load pricing from YAML file."""
        if not self.yaml_path.exists():
            return {}
        
        with open(self.yaml_path) as f:
            data = yaml.safe_load(f)
        
        # Flatten the structure: model -> {input, output, context}
        result = {}
        for provider_data in data.get("providers", {}).values():
            for model_id, costs in provider_data.get("models", {}).items():
                result[model_id] = costs
        
        return result
    
    def get_model_pricing(self, model_id: str) -> Dict[str, float]:
        """Get pricing for a model.
        
        Returns:
            Dict with 'input', 'output', 'context' keys.
            Returns zeros for unknown models.
        """
        return self._data.get(model_id, {
            "input": 0.0,
            "output": 0.0,
            "context": 4096
        })
    
    def calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD.
        
        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Total cost in USD
        """
        pricing = self.get_model_pricing(model_id)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)
    
    def list_models(self) -> list[str]:
        """List all models with pricing data."""
        return sorted(self._data.keys())


# Global instance for backward compatibility
_pricing = Pricing()


def get_model_cost(model_id: str) -> Dict[str, float]:
    """Get cost information for a specific model.
    
    Backward compatibility wrapper.
    
    Returns:
        Dict with 'input', 'output', 'context' keys.
        Note: Returns per-THOUSAND token costs for compatibility.
    """
    pricing = _pricing.get_model_pricing(model_id)
    
    # Convert to per-thousand for backward compatibility
    return {
        "input": pricing["input"] / 1000.0,
        "output": pricing["output"] / 1000.0,
        "context": pricing["context"]
    }


def get_model_costs() -> Dict[str, Dict[str, float]]:
    """Get all model costs.
    
    Backward compatibility wrapper.
    
    Returns:
        Dict of model_id -> cost info.
        Note: Returns per-THOUSAND token costs for compatibility.
    """
    result = {}
    for model_id in _pricing.list_models():
        result[model_id] = get_model_cost(model_id)
    return result


def get_model_pricing(model_id: str) -> Tuple[float, float]:
    """Get input and output pricing for a model.
    
    Backward compatibility wrapper.
    
    Returns:
        Tuple of (input_cost_per_1k, output_cost_per_1k) in USD.
    """
    cost = get_model_cost(model_id)
    return cost["input"], cost["output"]