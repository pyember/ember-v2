"""JIT compilation strategies.

Provides strategies for JIT compilation focused on structural analysis
of Operators. Each strategy analyzes operator patterns to enable
efficient parallel execution.
"""

from ember.xcs.jit.strategies.base_strategy import JITFallbackMixin, Strategy
from ember.xcs.jit.strategies.enhanced import EnhancedStrategy
from ember.xcs.jit.strategies.structural import StructuralStrategy

__all__ = [
    # Base protocol and utilities
    "Strategy",
    "JITFallbackMixin",
    # Concrete strategy implementations
    "StructuralStrategy",
    "EnhancedStrategy"]
