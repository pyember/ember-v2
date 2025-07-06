"""Model cost configuration - backward compatibility layer.

This module now delegates to the new pricing.py module but maintains
the old API for backward compatibility.
"""

# Re-export from new pricing module
from ember.models.pricing import (
    get_model_cost,
    get_model_costs,
    get_model_pricing,
)

__all__ = ["get_model_cost", "get_model_costs", "get_model_pricing"]
