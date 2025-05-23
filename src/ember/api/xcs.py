"""XCS API for Ember.

This module provides a clean interface for working with the XCS (Accelerated Compound Systems)
execution framework in Ember, offering high-performance execution capabilities for computational
graphs, just-in-time tracing, and parallel execution transformations.

Examples:
    # Basic JIT compilation (defaults to trace-based)
    from ember.api import xcs

    @xcs.jit
    class MyOperator(Operator):
        def forward(self, *, inputs):
            # Complex computation here
            return result

    # Using trace-based JIT with specific options
    @xcs.jit.trace(sample_input={"query": "test"})
    class TracedOperator(Operator):
        def forward(self, *, inputs):
            # Complex computation here
            return result

    # Using structural JIT for optimized parallel execution
    @xcs.jit.structural(execution_strategy="parallel")
    class ParallelOperator(Operator):
        def __init__(self):
            self.op1 = SubOperator1()
            self.op2 = SubOperator2()

        def forward(self, *, inputs):
            # Multi-step computation automatically parallelized
            result1 = self.op1(inputs=inputs)
            result2 = self.op2(inputs=inputs)
            return combine(result1, result2)

    # Using vectorized mapping
    @xcs.vmap(in_axes=(0, None))
    def process_batch(inputs, model):
        return model(inputs)

    # Using parallel execution
    @xcs.pmap
    def parallel_process(inputs):
        return heavy_computation(inputs)
"""

# Import from the unified implementation directly
from ember.xcs import (
    DeviceMesh,
    ExecutionResult,
    JITOptions,
    PartitionSpec,
    TraceContextData,
    TracerContext,
    TraceRecord,
    TransformOptions,
    XCSExecutionOptions,
    XCSExecutionOptions as ExecutionOptions,  # Alias for compatibility
    autograph,
)
from ember.xcs import execute_graph
from ember.xcs import execute_graph as execute
from ember.xcs import execution_options, jit, mesh_sharded, pmap, vmap
from ember.xcs.graph import XCSGraph
from ember.xcs.engine.execution_options import get_execution_options


__all__ = [
    # Core functions
    "jit",
    "autograph",
    "execute",
    "execute_graph",
    "execution_options",
    "get_execution_options",
    # Transforms
    "vmap",
    "pmap",
    "mesh_sharded",
    "DeviceMesh",
    "PartitionSpec",
    # Tracing
    "TracerContext",
    "TraceRecord",
    "TraceContextData",
    # Graph
    "XCSGraph",
    # Types
    "XCSExecutionOptions",
    "ExecutionOptions",  # Alias
    "ExecutionResult",
    "JITOptions",
    "TransformOptions",
]
