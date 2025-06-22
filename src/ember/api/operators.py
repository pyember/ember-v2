"""Operators API for Ember.

This module provides the public API for Ember's operator system, exposing
the simplified design with progressive disclosure.

Key components:
    Operator: Base class for all operators with optional validation
    @op: Decorator to convert functions to operators
    Common operators: Ensemble, Chain, Router, etc.

Basic usage:
    from ember.api import operators
    
    # Use the decorator for simple cases
    @operators.op
    def summarize(text: str) -> str: 
        return ember.model("gpt-4")(f"Summarize: {text}")
    
    # Or create operator classes for more control
    class MyOperator(operators.Operator):
        def forward(self, input):
            return process(input)
    @operators.measure
    def expensive_op(data):
        # Metrics tracked: latency, tokens, cost
        return model(data)
    
    # Compose operators
    pipeline = operators.chain(
        preprocess,
        summarize,
        postprocess
    )

Advanced usage:
    from ember.operators.advanced import Operator, TreeProtocol
    
    class CustomOperator(Operator):
        def forward(self, x):
            return self.process(x)

Experimental features:
    from ember.operators.experimental import trace, jit_compile
    
    # Trace execution for optimization
    traced = trace(my_operator)
    
    # Compile to optimized IR
    compiled = jit_compile(my_operator)

See Also:
    ember.core.operators: Core operator implementations
    ember.operators.advanced: Advanced operator patterns
    ember.operators.experimental: Experimental features
"""

# Core API imports
from ember.operators import (
    Operator,
    chain,
    ensemble,
)
# from ember._internal.registry.specification import Specification  # Deprecated - new operators use input_spec/output_spec
from ember.api.decorators import op

__all__ = [
    # Core classes and functions
    'Operator',
    'op',
    'chain',
    'ensemble',
    # 'Specification',  # Deprecated - new operators use input_spec/output_spec
]

# For backward compatibility and migration
def __getattr__(name):
    """Lazy loading for advanced and experimental features."""
    if name == 'advanced':
        from ember.operators import advanced
        return advanced
    elif name == 'experimental':
        from ember.operators import experimental
        return experimental
    elif name == 'legacy':
        from ember.operators import legacy
        return legacy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")