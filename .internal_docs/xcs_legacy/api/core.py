"""
Core implementation of the XCS API.

Entry point for simplified access to XCS functionality, providing an abstraction
over the underlying execution engine, tracing, and transformation capabilities.
"""

from __future__ import annotations

# Standard library imports
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

# Local imports
from ember.core.exceptions import RegistryError
from ember.xcs.api.types import (
    ExecutionResult,
    JITOptions,
    TransformOptions,
    XCSExecutionOptions,
)
from ember.xcs.engine.unified_engine import ExecutionOptions, execute_graph
from ember.xcs.exceptions import (
    CompilationError,
    DataFlowError,
    ExecutionError,
    TraceError,
    TransformError,
    XCSError,
)
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.jit import jit as raw_jit
from ember.xcs.tracer.autograph import AutoGraphBuilder
from ember.xcs.tracer.xcs_tracing import TraceRecord
from ember.xcs.transforms.mesh import DeviceMesh, PartitionSpec
from ember.xcs.transforms.mesh import mesh_sharded as raw_mesh_sharded
from ember.xcs.transforms.pmap import pmap as raw_pmap
from ember.xcs.transforms.vmap import vmap as raw_vmap

# Type variables
T = TypeVar("T")
ResultT = TypeVar("ResultT")


class XCSAPI:
    """
    Main API class for XCS functionality.

    Providing a unified interface to the XCS (eXecutable Computation System)
    functionality, abstracting away the details of the implementation.

    This class provides a simplified interface to the system with a
    consistent interface across different components.

    Example:
        ```python
        from ember.xcs.api.core import XCSAPI

        # Creating API instance
        xcs = XCSAPI()

        # Using JIT compilation
        @xcs.jit
        class MyOperator(Operator):
            def forward(self, *, inputs):
                return {"result": process(inputs)}

        # Using vectorization
        batch_fn = xcs.vmap(single_item_fn)

        # Using parallelization
        parallel_fn = xcs.pmap(compute_intensive_fn)

        # Building and executing a graph
        graph = xcs.autograph(trace_records)
        result = xcs.execute(graph, inputs={"query": "test"})
        ```
    """

    def __init__(self) -> None:
        """Initialize the XCS API."""
        self._graph_builder = AutoGraphBuilder()

    def jit(
        self,
        target: Optional[Type[T]] = None,
        *,
        options: Optional[JITOptions] = None,
        **kwargs: Any,
    ) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
        """
        Just-In-Time compilation decorator for Ember Operators.

        Transforming Operator classes to automatically trace their execution
        and compile optimized execution plans. This brings significant performance benefits
        for complex operations and operator pipelines.

        Args:
            target: The operator class to decorate (when used directly)
            options: Configuration options for JIT compilation
            **kwargs: Additional options passed directly to the underlying implementation

        Returns:
            The decorated operator class or a decorator function

        Example:
            ```python
            from ember.xcs import xcs

            # Using as a direct decorator
            @xcs.jit
            class MyOperator(Operator):
                def forward(self, *, inputs):
                    # Complex logic here
                    return {"result": process(inputs["query"])}

            # Creating an instance and executing
            op = MyOperator()
            result = op(inputs={"query": "example"})

            # Using with advanced configuration options
            @xcs.jit(options=JITOptions(
                sample_input={"query": "Example"},
                force_trace=False,
                recursive=True
            ))
            class OptimizedOperator(Operator):
                def __init__(self):
                    self.sub_op1 = SubOperator1()
                    self.sub_op2 = SubOperator2()

                def forward(self, *, inputs):
                    # Multi-stage processing with optimized execution
                    intermediate = self.sub_op1(inputs=inputs)
                    return self.sub_op2(inputs=intermediate)
            ```
        """
        # Handle options
        opts = options or JITOptions()

        # Prepare kwargs for raw_jit
        jit_kwargs = {
            "sample_input": opts.sample_input,
            "force_trace": opts.force_trace,
            "recursive": opts.recursive,
            **kwargs,
        }

        # When used directly as @xcs.jit
        if target is not None:
            return raw_jit(**jit_kwargs)(target)

        # When used as @xcs.jit() or @xcs.jit(options=...)
        return lambda cls: raw_jit(**jit_kwargs)(cls)

    def autograph(self, records: List[TraceRecord]) -> XCSGraph:
        """Build execution graph from trace records.

        Analyzes trace records to create a dependency graph for execution.

        Args:
            records: Trace records from execution

        Returns:
            XCS execution graph
            
        Raises:
            TraceError: When trace records are invalid or cannot be analyzed
            DataFlowError: When data flow issues are detected in the graph
            
        Example:
            ```python
            with TracerContext() as tracer:
                op(inputs={"query": "Example"})
            graph = xcs.autograph(tracer.records)
            ```
        """
        if not records:
            raise TraceError(
                message="Cannot build graph from empty trace records",
                context={"record_count": 0}
            )
            
        try:
            return self._graph_builder.build_graph(records=records)
        except Exception as e:
            # Convert generic exceptions to proper XCS exceptions
            if "cycle" in str(e).lower() or "circular" in str(e).lower():
                raise DataFlowError(
                    message="Circular dependency detected in execution graph",
                    context={"record_count": len(records)},
                    cause=e
                )
            raise TraceError(
                message=f"Failed to build execution graph: {e}",
                context={"record_count": len(records)},
                cause=e
            )

    def execute(
        self,
        graph: XCSGraph,
        inputs: Dict[str, Any],
        options: Optional[XCSExecutionOptions] = None,
    ) -> ExecutionResult:
        """Execute XCS graph with given inputs.

        Args:
            graph: XCS graph to execute
            inputs: Input values for the graph
            options: Execution options for parallelism and timeout

        Returns:
            Execution result with outputs and timing

        Raises:
            ExecutionError: When graph execution fails
            
        Example:
            ```python
            exec_options = XCSExecutionOptions(max_workers=4, timeout=10000)
            result = xcs.execute(
                graph,
                inputs={"query": "Example"},
                options=exec_options
            )
            ```
        """
        opts = options or XCSExecutionOptions()

        engine_options = ExecutionOptions(
            scheduler_type="parallel",  # Always use parallel execution for performance
            max_workers=opts.max_workers,
            timeout_seconds=opts.timeout / 1000 if opts.timeout else None,
            collect_metrics=True,
        )

        start_time = time.time()

        try:
            outputs = execute_graph(
                graph=graph, global_input=inputs, options=engine_options
            )
        except Exception as e:
            # Convert generic exceptions to proper XCS exceptions
            if "timeout" in str(e).lower():
                timeout_value = opts.timeout / 1000 if opts.timeout else "default"
                raise ExecutionError(
                    message=f"Graph execution timed out after {timeout_value} seconds",
                    context={
                        "timeout_seconds": timeout_value,
                        "graph_node_count": len(graph.nodes) if hasattr(graph, "nodes") else 0,
                    },
                    cause=e
                )
                
            if "memory" in str(e).lower():
                raise ExecutionError(
                    message="Graph execution failed due to memory constraints",
                    context={"graph_input_size": sum(len(str(v)) for v in inputs.values())},
                    cause=e
                )
                
            raise ExecutionError(
                message=f"Graph execution failed: {str(e)}",
                context={
                    "graph_node_count": len(graph.nodes) if hasattr(graph, "nodes") else 0,
                    "input_keys": list(inputs.keys()),
                },
                cause=e
            )

        execution_time = time.time() - start_time
        return ExecutionResult(outputs=outputs, execution_time=execution_time)

    def vmap(
        self,
        fn: Callable[..., ResultT],
        options: Optional[TransformOptions] = None,
        **kwargs: Any,
    ) -> Callable[..., Dict[str, Any]]:
        """Vectorize function across its inputs.

        Args:
            fn: Function to vectorize
            options: Vectorization configuration
            **kwargs: Additional options for implementation

        Returns:
            Vectorized function
            
        Raises:
            TransformError: When vectorization fails
            
        Example:
            ```python
            def process_item(item):
                return item * 2

            batch_process = xcs.vmap(process_item)
            results = batch_process([1, 2, 3])  # [2, 4, 6]
            ```
        """
        opts = options or TransformOptions()
        vmap_kwargs = {"in_axes": opts.in_axes, "out_axes": opts.out_axes, **kwargs}
        
        try:
            return cast(
                Callable[..., Dict[str, Any]], 
                raw_vmap(fn, **vmap_kwargs)
            )
        except Exception as e:
            raise TransformError.for_transform(
                transform_name="vmap",
                message=f"Failed to apply vmap transform: {str(e)}",
                details={"function_name": fn.__name__ if hasattr(fn, "__name__") else "unknown"},
                cause=e
            )

    def pmap(
        self,
        fn: Callable[..., ResultT],
        options: Optional[TransformOptions] = None,
        **kwargs: Any,
    ) -> Callable[..., Dict[str, Any]]:
        """Parallelize function across multiple devices.

        Args:
            fn: Function to parallelize
            options: Parallelization configuration
            **kwargs: Additional options for implementation

        Returns:
            Parallelized function
            
        Raises:
            TransformError: When parallelization fails

        Example:
            ```python
            def process_item(item):
                return item * 2

            parallel_process = xcs.pmap(process_item)
            results = parallel_process([1, 2, 3])  # [2, 4, 6]
            ```
        """
        opts = options or TransformOptions()
        pmap_kwargs = {
            "in_axes": opts.in_axes,
            "out_axes": opts.out_axes,
            "devices": opts.devices,
            **kwargs,
        }

        try:
            return cast(
                Callable[..., Dict[str, Any]], 
                raw_pmap(fn, **pmap_kwargs)
            )
        except Exception as e:
            # Create context with detailed device information
            context = {
                "function_name": fn.__name__ if hasattr(fn, "__name__") else "unknown",
            }
            
            if opts.devices:
                context["devices"] = str(opts.devices)
                
            raise TransformError.for_transform(
                transform_name="pmap",
                message=f"Failed to apply pmap transform: {str(e)}",
                details=context,
                cause=e
            )

    def mesh_sharded(
        self,
        fn: Callable[..., ResultT],
        mesh: DeviceMesh,
        partition_spec: PartitionSpec,
        **kwargs: Any,
    ) -> Callable[..., Dict[str, Any]]:
        """Apply mesh-based sharding to a function.

        Args:
            fn: Function to shard
            mesh: Device mesh to use
            partition_spec: Partition specification
            **kwargs: Additional options for implementation

        Returns:
            Sharded function
            
        Raises:
            TransformError: When mesh sharding fails

        Example:
            ```python
            mesh = xcs.DeviceMesh(devices=[0, 1, 2, 3], mesh_shape=(2, 2))
            pspec = xcs.PartitionSpec(0, 1)
            sharded_fn = xcs.mesh_sharded(fn, mesh, pspec)
            ```
        """
        try:
            return cast(
                Callable[..., Dict[str, Any]],
                raw_mesh_sharded(fn, mesh, partition_spec, **kwargs)
            )
        except Exception as e:
            # Create detailed context for diagnostics
            context = {
                "function_name": fn.__name__ if hasattr(fn, "__name__") else "unknown",
                "mesh_shape": str(mesh.mesh_shape),
                "device_count": len(mesh.devices),
                "partition_spec": str(partition_spec),
            }
            
            raise TransformError.for_transform(
                transform_name="mesh_sharded",
                message=f"Failed to apply mesh sharding: {str(e)}",
                details=context,
                cause=e
            )

    # Re-export classes for convenience
    DeviceMesh = DeviceMesh
    PartitionSpec = PartitionSpec
