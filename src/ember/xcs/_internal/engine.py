"""Execution engine - hidden from users.

This implements all the complex scheduling and execution logic,
but users never see any of it. They just get fast execution.
"""

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ember.xcs._internal.ir import IRGraph, IRNode
from ember.xcs._internal.parallelism import GraphParallelismAnalysis


@dataclass
class ExecutionContext:
    """Context for executing a graph."""

    variables: Dict[str, Any]  # Variable storage
    cache: Dict[str, Any]  # Result cache

    def get(self, var_name: str) -> Any:
        """Get variable value."""
        return self.variables.get(var_name)

    def set(self, var_name: str, value: Any):
        """Set variable value."""
        self.variables[var_name] = value


class ExecutionEngine:
    """Executes IR graphs with automatic optimization.

    All the complexity is hidden here. Users never see:
    - Schedulers
    - Execution strategies
    - Parallelization details
    - Caching logic
    """

    def __init__(self):
        self.cache = {}  # Simple cache for now
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def execute(
        self,
        graph: IRGraph,
        args: tuple,
        kwargs: dict,
        parallelism_info: Optional[GraphParallelismAnalysis] = None,
        config: Optional[Any] = None,
    ) -> Any:
        """Execute computation graph with automatic optimization."""
        # Create execution context
        context = ExecutionContext(variables={}, cache={})

        # Map input arguments to variables
        for i, arg in enumerate(args):
            context.set(f"_arg_{i}", arg)

        # Choose execution strategy based on graph analysis
        if parallelism_info and self._should_parallelize(parallelism_info):
            result = self._execute_parallel(graph, context, parallelism_info)
        else:
            result = self._execute_sequential(graph, context)

        # Return the final result
        # Assuming last node produces the result
        if graph.nodes:
            last_node = list(graph.nodes.values())[-1]
            if last_node.outputs:
                return context.get(last_node.outputs[0])

        return result

    def _should_parallelize(self, info: GraphParallelismAnalysis) -> bool:
        """Decide if parallelization is worth it."""
        # Simple heuristic: parallelize if speedup > 1.5x
        return info.estimated_speedup > 1.5

    def _execute_sequential(self, graph: IRGraph, context: ExecutionContext) -> Any:
        """Simple sequential execution."""
        # Execute nodes in topological order
        for node_id in graph.topological_sort():
            node = graph.nodes[node_id]
            self._execute_node(node, context)

        return None

    def _execute_parallel(
        self, graph: IRGraph, context: ExecutionContext, info: GraphParallelismAnalysis
    ) -> Any:
        """Parallel execution with exact sequential semantics.

        Critical: If node 50 of 100 fails, we must raise the error
        at exactly the same point as sequential execution would.
        """
        # Execute in topological order but parallelize where possible
        topo_order = graph.topological_sort()

        # Track futures for nodes being executed
        pending_futures: Dict[str, concurrent.futures.Future] = {}

        for node_id in topo_order:
            node = graph.nodes[node_id]

            # Wait for dependencies to complete
            deps = graph.get_dependencies(node_id)
            for dep_id in deps:
                if dep_id in pending_futures:
                    try:
                        # Wait for dependency and get result
                        future = pending_futures[dep_id]
                        result = future.result()
                        # Update context with result
                        dep_node = graph.nodes[dep_id]
                        if dep_node.outputs and result is not None:
                            context.set(dep_node.outputs[0], result)
                    except Exception:
                        # Cancel all pending futures
                        for f in pending_futures.values():
                            f.cancel()
                        # Re-raise to preserve sequential semantics
                        raise

            # Check if this node can be parallelized with others
            can_parallelize = False
            for group in info.parallel_groups:
                if node_id in group:
                    # Check if all members of the group have satisfied dependencies
                    group_ready = all(
                        all(
                            d in context.variables
                            or d.startswith("_arg_")
                            or d.startswith("_literal_")
                            for d in graph.nodes[gid].inputs
                        )
                        for gid in group
                        if gid != node_id
                    )
                    if group_ready:
                        can_parallelize = True
                        break

            if can_parallelize:
                # Submit for parallel execution
                if node_id not in pending_futures:
                    future = self.executor.submit(
                        self._execute_node_isolated, node, dict(context.variables)
                    )
                    pending_futures[node_id] = future
                # Skip sequential execution - will collect result later
            else:
                # Execute sequentially only if not already submitted
                if node_id not in pending_futures:
                    try:
                        self._execute_node(node, context)
                    except Exception:
                        # Cancel all pending futures
                        for f in pending_futures.values():
                            f.cancel()
                        raise

        # Wait for any remaining futures
        for node_id, future in pending_futures.items():
            try:
                result = future.result()
                node = graph.nodes[node_id]
                if node.outputs and result is not None:
                    context.set(node.outputs[0], result)
            except Exception:
                # Cancel remaining futures
                for f in pending_futures.values():
                    if not f.done():
                        f.cancel()
                raise

        return None

    def _execute_wave_parallel(
        self,
        graph: IRGraph,
        wave: List[str],
        context: ExecutionContext,
        info: GraphParallelismAnalysis,
    ):
        """Execute a wave of nodes in parallel."""
        futures = []

        for node_id in wave:
            node = graph.nodes[node_id]
            node_info = info.node_info.get(node_id)

            if node_info and node_info.can_parallelize:
                # Submit to thread pool
                future = self.executor.submit(
                    self._execute_node_isolated,
                    node,
                    dict(context.variables),  # Copy for thread safety
                )
                futures.append((node_id, future))
            else:
                # Execute directly
                self._execute_node(node, context)

        # Collect results
        for node_id, future in futures:
            node = graph.nodes[node_id]
            result = future.result()
            # Store outputs
            if node.outputs and result is not None:
                context.set(node.outputs[0], result)

    def _execute_node(self, node: IRNode, context: ExecutionContext) -> Any:
        """Execute a single node.

        Critical: We must fail fast - no swallowing exceptions!
        """
        # Gather inputs
        input_values = []
        for input_var in node.inputs:
            value = context.get(input_var)
            if value is not None:
                input_values.append(value)

        # Execute operator - let exceptions propagate!
        if callable(node.operator):
            # Check if this is a return node (special handling)
            if node.metadata.get("is_return"):
                # Return nodes store their result directly
                result = node.metadata.get("result")
            elif "args" in node.metadata and "kwargs" in node.metadata:
                # Pure functions with stored args (traced args)
                result = node.operator(*node.metadata["args"], **node.metadata.get("kwargs", {}))
            elif node.metadata.get("is_orchestration") and "orchestration_args" in node.metadata:
                # Orchestration operations - use their stored args
                args = node.metadata["orchestration_args"]
                kwargs = node.metadata.get("orchestration_kwargs", {})
                result = node.operator(*args, **kwargs)
            else:
                # Fallback - use runtime inputs
                result = node.operator(*input_values)
        else:
            # Fallback for non-callable operators
            result = input_values[0] if input_values else None

        # Store outputs
        if node.outputs and result is not None:
            context.set(node.outputs[0], result)

        return result

    def _execute_node_isolated(self, node: IRNode, variables: Dict[str, Any]) -> Any:
        """Execute node in isolation (for parallel execution)."""
        # Create isolated context
        context = ExecutionContext(variables=variables, cache={})
        return self._execute_node(node, context)

    def _compute_execution_waves(self, graph: IRGraph) -> List[List[str]]:
        """Compute waves of nodes that can execute together."""
        waves = []
        remaining = set(graph.nodes.keys())
        completed = set()

        while remaining:
            # Find nodes whose dependencies are satisfied
            wave = []
            for node_id in remaining:
                deps = graph.get_dependencies(node_id)
                if deps.issubset(completed):
                    wave.append(node_id)

            if not wave:
                # Circular dependency or error
                # Execute remaining nodes sequentially
                wave = list(remaining)

            waves.append(wave)
            completed.update(wave)
            remaining.difference_update(wave)

        return waves

    def shutdown(self):
        """Clean shutdown of execution resources."""
        self.executor.shutdown(wait=True)
