# Ember Simplified API
# The public API - just 10 functions

"""
The entire Ember API in one import:
    from ember import llm, jit, vmap, pmap, chain, ensemble, retry, cache, measure, stream
"""

from ember.core.simple import (
    # Core functions
    llm,
    jit,
    vmap,
    pmap,
    chain,
    ensemble,
    retry,
    cache,
    measure,
    stream,
    
    # Configuration (advanced usage)
    register_provider,
    set_default_model,
    
    # Metrics (optional)
    get_metrics,
    reset_metrics,
    metrics_context,
    
    # Utilities
    clear_cache,
    majority_vote,
    mean_aggregator,
    first_valid,
    
    # Types (for provider implementations)
    Provider,
    Metrics
)

# Re-export everything
__all__ = [
    'llm',
    'jit', 
    'vmap',
    'pmap',
    'chain',
    'ensemble',
    'retry',
    'cache',
    'measure',
    'stream',
    'register_provider',
    'set_default_model',
    'get_metrics',
    'reset_metrics',
    'metrics_context',
    'clear_cache',
    'majority_vote',
    'mean_aggregator',
    'first_valid',
    'Provider',
    'Metrics'
]

# But emphasize the core 10
CORE_API = [
    'llm',
    'jit', 
    'vmap',
    'pmap',
    'chain',
    'ensemble',
    'retry',
    'cache',
    'measure',
    'stream',
]

def __dir__():
    """Show core API first when exploring."""
    return CORE_API + [x for x in __all__ if x not in CORE_API]