"""Unified JIT compilation system for XCS.

Provides Just-In-Time compilation strategies for optimizing operator execution
through tracing, structural analysis, and enhanced dependency tracking.

The JIT system enables automatic caching of computation results, tracking of 
performance metrics, and selection of optimal compilation strategies based on
operator characteristics. These features combine to create efficient execution
pipelines with minimal user configuration.
"""

from typing import Any, Callable, Dict, Optional

# JIT caching system
from ember.xcs.jit.cache import JITCache, get_cache

# Core JIT decorator - these need to be imported after JITMode to avoid circular imports
from ember.xcs.jit.core import explain_jit_selection, get_jit_stats, jit

# Import JIT modes
from ember.xcs.jit.modes import JITMode

__all__ = [
    # Core JIT functionality
    "jit",
    "JITMode",
    "JITCache",
    "get_jit_stats",
    "explain_jit_selection",
    "get_cache",
]
