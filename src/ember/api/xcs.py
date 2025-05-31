"""Simplified XCS API - Zero Configuration, Maximum Power.

The XCS (eXecution Coordination System) provides automatic optimization
for compound AI systems with just four essential functions:

- @jit: Zero-configuration optimization for any function or operator
- @trace: Execution analysis and debugging  
- vmap: Transform single-item functions to batch processors
- get_jit_stats: Performance metrics and insights

That's it. No strategies to choose, no modes to configure.

Natural API: Just write Python. XCS handles the rest.
"""

# Import everything from the clean XCS module
# Use simple implementations for stability
from ember.xcs.jit.simple_jit import jit
from ember.xcs.simple_vmap import vmap
from ember.xcs.trace import trace

def get_jit_stats(func=None):
    """Get optimization statistics."""
    # Simple implementation for now
    return {
        'version': '2.0.0',
        'natural_api': True,
        'transformations_available': ['jit', 'vmap', 'trace']
    }

# Aliases for backward compatibility
autograph = trace  # Similar functionality for execution analysis

__all__ = [
    # Core API - just 4 functions
    "jit",           # Automatic optimization
    "trace",         # Execution analysis
    "get_jit_stats", # Performance monitoring
    "vmap",          # Single-item → batch transformation
    
    # Compatibility alias
    "autograph",     # Alias for trace
]