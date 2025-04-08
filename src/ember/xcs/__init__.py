"""XCS: Ember Execution Framework.

Provides a high-performance computational engine for building, optimizing,
and executing complex operator pipelines. Core features include automatic
parallelization, just-in-time compilation, and intelligent scheduling.
"""

from ember.xcs.api.types import ExecutionResult as APIExecutionResult
from ember.xcs.api.types import JITOptions, TransformOptions, XCSExecutionOptions

# Public API - execution results and plans
from ember.xcs.common.plans import ExecutionResult, XCSPlan, XCSTask
from ember.xcs.engine.execution_options import ExecutionOptions

# Implementation components - execution
# Public API - execution control
from ember.xcs.engine.unified_engine import (
    ExecutionMetrics,
    GraphExecutor,
    execute_graph,
    execution_options,
)
from ember.xcs.graph.dependency_analyzer import DependencyAnalyzer
from ember.xcs.graph.graph_builder import EnhancedTraceGraphBuilder, GraphBuilder

# Implementation components - graph system
from ember.xcs.graph.xcs_graph import XCSGraph, XCSNode

# Public API - core optimization system
from ember.xcs.jit import JITCache, JITMode, explain_jit_selection, get_jit_stats, jit

# Implementation components - schedulers
from ember.xcs.schedulers.base_scheduler import BaseScheduler
from ember.xcs.schedulers.factory import create_scheduler
from ember.xcs.schedulers.unified_scheduler import (
    NoOpScheduler,
    ParallelScheduler,
    SequentialScheduler,
    TopologicalScheduler,
    WaveScheduler,
)
from ember.xcs.tracer._context_types import TraceContextData

# Public API - graph construction
from ember.xcs.tracer.autograph import AutoGraphBuilder, autograph

# Implementation components - tracing
from ember.xcs.tracer.xcs_tracing import TracerContext, TraceRecord
from ember.xcs.transforms.mesh import DeviceMesh, PartitionSpec, mesh_sharded
from ember.xcs.transforms.pmap import pjit, pmap  # Parallelization

# Implementation components - transformations
from ember.xcs.transforms.transform_base import (
    BaseTransformation,
    BatchingOptions,
    ParallelOptions,
    TransformError,
    compose,
)

# Public API - transformations for parallel execution
from ember.xcs.transforms.vmap import vmap  # Vectorization

# Explicitly define public interface
__all__ = [
    # Core user-facing API - optimization
    "jit",
    "JITMode",
    "get_jit_stats",
    "explain_jit_selection",
    # Core user-facing API - execution
    "execute_graph",
    "execution_options",
    "ExecutionOptions",
    "create_scheduler",
    "ExecutionResult",
    # Core user-facing API - transformations
    "vmap",
    "pmap",
    "pjit",
    "DeviceMesh",
    "PartitionSpec",
    "mesh_sharded",
    "compose",
    "TransformError",
    # Core user-facing API - graph building
    "autograph",
    "TracerContext",
    # Core user-facing API - configuration
    "JITOptions",
    "XCSExecutionOptions",
    "APIExecutionResult",
    "TransformOptions",
    # Implementation details - generally not needed by users
    "XCSGraph",
    "XCSNode",
    "DependencyAnalyzer",
    "GraphBuilder",
    "EnhancedTraceGraphBuilder",
    "TraceRecord",
    "TraceContextData",
    "AutoGraphBuilder",
    "XCSPlan",
    "XCSTask",
    "BaseScheduler",
    "NoOpScheduler",
    "ParallelScheduler",
    "SequentialScheduler",
    "TopologicalScheduler",
    "WaveScheduler",
    "GraphExecutor",
    "ExecutionMetrics",
    "JITCache",
    "BaseTransformation",
    "BatchingOptions",
    "ParallelOptions",
]
