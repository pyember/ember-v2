"""XCS (eXecution Coordination System) - Zero Configuration, Maximum Power.

The XCS provides automatic optimization for compound AI systems through
intelligent analysis of your code structure. Just add @jit to any function
or operator and XCS handles the rest.

Core API (just 4 functions):
- @jit: Zero-configuration optimization 
- @trace: Execution analysis and debugging
- vmap: Transform single-item → batch operations
- get_jit_stats: Performance metrics
"""

# Core optimization
from ember.xcs.jit import jit, get_jit_stats

# Analysis and debugging  
from ember.xcs.trace import trace

# Transformations
from ember.xcs.transforms.vmap import vmap

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