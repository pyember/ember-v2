"""Simplified JIT System.

Only real optimization - structural analysis of Operators.
No false promises about trace-based speedup.
"""

# Core JIT functionality
from ember.xcs.jit.core import (
    jit,
    get_jit_stats,
    explain_jit_selection)

# Keep cache for compatibility
from ember.xcs.jit.cache import JITCache, get_cache

__all__ = [
    "jit",
    "get_jit_stats", 
    "explain_jit_selection",
    "JITCache",
    "get_cache"]