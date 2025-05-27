"""Simplified XCS API - Clean and Powerful.

Only the essentials:
- jit: Optimize Operators
- trace: Analyze execution
- Graph: Build computation graphs
"""

# Core optimization and analysis
from ember.xcs import (
    jit,
    trace,
    trace_execution,
    Graph,
    Node,
    ExecutionOptions,
    get_jit_stats,
    explain_jit_selection)

# Transformations
from ember.xcs import vmap, pmap

# Legacy imports for compatibility
from ember.xcs import XCSGraph  # Deprecated, use Graph
from ember.xcs.execution_options import get_execution_options

# Aliases for compatibility
autograph = trace  # Similar functionality

__all__ = [
    # Optimization
    "jit",
    "get_jit_stats",
    "explain_jit_selection",
    
    # Analysis
    "trace",
    "trace_execution",
    "autograph",  # Alias for trace
    
    # Graph
    "Graph", 
    "Node",
    "ExecutionOptions",
    "get_execution_options",
    
    # Transformations
    "vmap",
    "pmap",
    
    # Deprecated
    "XCSGraph"]