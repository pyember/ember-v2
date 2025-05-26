"""XCS (Execution and Compilation System) - Simplified.

Clean, powerful APIs for optimization and analysis:
- @jit: Optimize Operators with parallelizable patterns
- @trace: Analyze execution for debugging and metrics
- Graph: Simple, powerful computation graph
"""

# Core optimization
from ember.xcs.jit import jit, get_jit_stats, explain_jit_selection

# Analysis and debugging
from ember.xcs.trace import trace, trace_execution

# Graph representation
from ember.xcs.graph.graph import Graph, Node

# Transformations (kept for compatibility)
from ember.xcs.transforms.vmap import vmap
from ember.xcs.transforms.pmap import pmap

# Execution (simplified)
from ember.xcs.graph import Graph
from ember.xcs.execution_options import ExecutionOptions

# For backward compatibility (with deprecation warning)
import warnings

def _deprecated_xcs_graph(*args, **kwargs):
    warnings.warn(
        "XCSGraph is deprecated. Use Graph instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return Graph(*args, **kwargs)

# Compatibility aliases
XCSGraph = Graph  # Direct alias for easier migration
JITMode = None  # No longer needed

# Public API - only what users actually need
__all__ = [
    # Optimization
    "jit",
    "get_jit_stats",
    "explain_jit_selection",
    
    # Analysis
    "trace",
    "trace_execution",
    
    # Graph
    "Graph",
    "Node",
    "ExecutionOptions",
    
    # Transformations
    "vmap",
    "pmap",
    
    # Compatibility
    "XCSGraph",  # Deprecated alias
]