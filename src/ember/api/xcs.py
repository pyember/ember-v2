"""Simplified XCS API - Zero Configuration, Maximum Power.

The XCS (eXecution Coordination System) provides automatic optimization
for compound AI systems with just four essential functions:

- @jit: Zero-configuration optimization for any function or operator
- @trace: Execution analysis and debugging  
- vmap: Transform single-item functions to batch processors
- get_jit_stats: Performance metrics and insights

That's it. No strategies to choose, no modes to configure.
"""

# Core imports - only what users actually need
from ember.xcs import jit, trace, get_jit_stats, vmap

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