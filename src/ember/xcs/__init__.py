"""XCS (eXecution Coordination System) - Zero Configuration, Maximum Power.

The XCS provides automatic optimization for compound AI systems through
intelligent analysis of your code structure. Just add @jit to any function
or operator and XCS handles the rest.

Core API (just 4 functions):
- @jit: Zero-configuration optimization 
- @trace: Execution analysis and debugging
- vmap: Transform single-item → batch operations
- get_jit_stats: Performance metrics

Natural API: Just write Python. XCS handles the rest.
"""

# Import natural implementations that hide internal details
from ember.xcs.natural_v2 import (
    natural_jit as jit,
    natural_vmap as vmap,
    get_transformation_info as _get_transformation_info
)

# Analysis and debugging (already clean)
from ember.xcs.trace import trace


def get_jit_stats(func=None):
    """Get optimization statistics.
    
    Args:
        func: Optional function to get stats for
        
    Returns:
        User-friendly statistics dictionary
    """
    if func is not None:
        # Get stats for specific function
        info = _get_transformation_info(func)
        if info.get('has_jit'):
            return {
                'optimized': True,
                'transformations': info.get('transformations', []),
            }
        else:
            return {'optimized': False}
    else:
        # Global stats - simplified
        return {
            'version': '2.0.0',
            'natural_api': True,
            'transformations_available': ['jit', 'vmap', 'trace']
        }

# Public API - only what users actually need
__all__ = [
    # Core API - just 4 functions
    "jit",           # Automatic optimization
    "trace",         # Execution analysis  
    "get_jit_stats", # Performance monitoring
    "vmap",          # Single-item → batch transformation
    
    # No more exports - keep it simple!
    # Removed: pmap (not useful for I/O-bound operations)
    # Removed: Graph, Node (internal implementation details)
    # Removed: ExecutionOptions (zero-configuration philosophy)
    # Removed: explain_jit_selection (too much internal detail)
    # Removed: trace_execution (redundant with trace)
]