"""Automatic parallelism discovery from operator structure.

Following Dean/Ghemawat: Discover parallelism from data structure,
not user configuration.
"""

from dataclasses import dataclass
from typing import Dict, List, Set

try:
    import jax.tree_util as tree_util

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    tree_util = None

from ember.xcs._internal.ir import (
    IRGraph,
    IRNode,
    ParallelismInfo,
    analyze_node_parallelism,
)

# Removed VectorizedOp import - not needed with new tracer-based approach


@dataclass(frozen=True)
class GraphParallelismAnalysis:
    """Complete parallelism analysis for a graph."""

    node_info: Dict[str, ParallelismInfo]
    parallel_groups: List[Set[str]]  # Groups of nodes that can run in parallel
    vectorizable_chains: List[List[str]]  # Chains that can be vectorized
    estimated_speedup: float
    bottlenecks: List[str]  # Node IDs that limit parallelism


class ParallelismAnalyzer:
    """Discovers parallelism opportunities in computation graphs."""

    def analyze_graph(self, graph: IRGraph) -> GraphParallelismAnalysis:
        """Analyze entire graph for parallelism opportunities."""
        # Analyze individual nodes
        node_info = {}
        for node_id, node in graph.nodes.items():
            node_info[node_id] = self._analyze_node(node, graph)

        # Find parallel groups (independent branches)
        parallel_groups = self._find_parallel_groups(graph)

        # Find vectorizable chains
        vectorizable_chains = self._find_vectorizable_chains(graph, node_info)

        # Identify bottlenecks
        bottlenecks = self._find_bottlenecks(graph, node_info, parallel_groups)

        # Estimate overall speedup
        speedup = self._estimate_speedup(node_info, parallel_groups, vectorizable_chains)

        return GraphParallelismAnalysis(
            node_info=node_info,
            parallel_groups=parallel_groups,
            vectorizable_chains=vectorizable_chains,
            estimated_speedup=speedup,
            bottlenecks=bottlenecks,
        )

    def _analyze_node(self, node: IRNode, graph: IRGraph) -> ParallelismInfo:
        """Analyze a single node in graph context."""
        # Start with basic node analysis
        base_info = analyze_node_parallelism(node)

        # Enhance with graph context
        info_dict = {
            "can_vmap": base_info.can_vmap,
            "can_pmap": base_info.can_pmap,
            "can_batch": base_info.can_batch,
            "can_parallelize": base_info.can_parallelize,
            "is_pure": base_info.is_pure,
            "estimated_speedup": base_info.estimated_speedup,
        }

        # Check if node has independent branches
        if graph.has_independent_branches(node.id):
            info_dict["can_parallelize"] = True
            info_dict["estimated_speedup"] = max(
                info_dict["estimated_speedup"], len(graph.get_dependents(node.id))
            )

        # Check if node is part of a comprehension (would be marked in metadata)
        if node.metadata.get("is_comprehension", False):
            info_dict["can_vmap"] = True
            info_dict["estimated_speedup"] = max(info_dict["estimated_speedup"], 4.0)

        return ParallelismInfo(**info_dict)

    def _find_parallel_groups(self, graph: IRGraph) -> List[Set[str]]:
        """Find groups of nodes that can execute in parallel."""
        parallel_groups = []

        # Use topological order to find parallel opportunities
        topo_order = graph.topological_sort()

        # Group nodes by their depth in the graph
        depth_map = self._compute_depths(graph, topo_order)
        depth_groups = {}

        for node_id, depth in depth_map.items():
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node_id)

        # Check which nodes at same depth can run in parallel
        for _depth, nodes in depth_groups.items():
            if len(nodes) > 1:
                # Check for data dependencies within group
                independent_nodes = set()
                for i, node1 in enumerate(nodes):
                    is_independent = True
                    for j, node2 in enumerate(nodes):
                        if i != j:
                            # Check if node1 depends on node2 or vice versa
                            if self._has_dependency(graph, node1, node2):
                                is_independent = False
                                break
                    if is_independent:
                        independent_nodes.add(node1)

                if len(independent_nodes) > 1:
                    parallel_groups.append(independent_nodes)

        return parallel_groups

    def _find_vectorizable_chains(
        self, graph: IRGraph, node_info: Dict[str, ParallelismInfo]
    ) -> List[List[str]]:
        """Find chains of operations that can be vectorized together."""
        chains = []
        visited = set()

        for node_id in graph.topological_sort():
            if node_id in visited:
                continue

            # Start a chain if node is vectorizable
            if node_info[node_id].can_vmap:
                chain = [node_id]
                visited.add(node_id)

                # Extend chain with compatible operations
                current = node_id
                while True:
                    dependents = graph.get_dependents(current)
                    if len(dependents) == 1:
                        next_id = next(iter(dependents))
                        if (
                            next_id not in visited
                            and node_info[next_id].can_vmap
                            and self._are_compatible_for_fusion(graph, current, next_id)
                        ):
                            chain.append(next_id)
                            visited.add(next_id)
                            current = next_id
                        else:
                            break
                    else:
                        break

                if len(chain) > 1:
                    chains.append(chain)

        return chains

    def _find_bottlenecks(
        self,
        graph: IRGraph,
        node_info: Dict[str, ParallelismInfo],
        parallel_groups: List[Set[str]],
    ) -> List[str]:
        """Identify nodes that limit parallelism."""
        bottlenecks = []

        # Find nodes that force synchronization
        for node_id, _node in graph.nodes.items():
            # Check if node has multiple dependencies (join point)
            deps = graph.get_dependencies(node_id)
            if len(deps) > 1:
                # Check if dependencies could run in parallel
                deps_parallel = False
                for group in parallel_groups:
                    if len(deps & group) > 1:
                        deps_parallel = True
                        break

                if deps_parallel and not node_info[node_id].can_batch:
                    # This node forces parallel branches to synchronize
                    bottlenecks.append(node_id)

            # Check if non-parallelizable node has parallel dependents
            if not node_info[node_id].can_parallelize:
                dependents = graph.get_dependents(node_id)
                if len(dependents) > 1:
                    bottlenecks.append(node_id)

        return bottlenecks

    def _estimate_speedup(
        self,
        node_info: Dict[str, ParallelismInfo],
        parallel_groups: List[Set[str]],
        vectorizable_chains: List[List[str]],
    ) -> float:
        """Estimate overall speedup from parallelism."""
        if not node_info:
            return 1.0

        # Calculate speedup from parallel groups
        parallel_speedup = 1.0
        for group in parallel_groups:
            # Assuming perfect parallelism within groups
            group_size = len(group)
            if group_size > 1:
                parallel_speedup *= group_size

        # Calculate speedup from vectorization
        vector_speedup = 1.0
        for chain in vectorizable_chains:
            # Vectorization typically gives 2-8x speedup
            chain_speedup = min(len(chain) * 2, 8)
            vector_speedup = max(vector_speedup, chain_speedup)

        # Combine speedups (not purely multiplicative due to overhead)
        combined = (parallel_speedup + vector_speedup) / 2

        # Cap at realistic maximum
        return min(combined, 10.0)

    def _compute_depths(self, graph: IRGraph, topo_order: List[str]) -> Dict[str, int]:
        """Compute depth of each node in the graph."""
        depths = {}

        for node_id in topo_order:
            deps = graph.get_dependencies(node_id)
            if not deps:
                depths[node_id] = 0
            else:
                max_dep_depth = max(depths.get(dep, 0) for dep in deps)
                depths[node_id] = max_dep_depth + 1

        return depths

    def _has_dependency(self, graph: IRGraph, node1: str, node2: str) -> bool:
        """Check if node1 depends on node2 (directly or indirectly)."""
        visited = set()
        queue = [node1]

        while queue:
            current = queue.pop(0)
            if current == node2:
                return True

            if current in visited:
                continue
            visited.add(current)

            deps = graph.get_dependencies(current)
            queue.extend(deps)

        return False

    def _are_compatible_for_fusion(self, graph: IRGraph, node1: str, node2: str) -> bool:
        """Check if two nodes can be fused into single vectorized operation."""
        # Simple compatibility check
        # In practice, would check operator types, data types, etc.
        n1 = graph.nodes[node1]
        n2 = graph.nodes[node2]

        # Both must be pure (no side effects)
        if not (n1.metadata.get("is_pure", True) and n2.metadata.get("is_pure", True)):
            return False

        # Must have compatible input/output counts
        if len(n1.outputs) != len(n2.inputs):
            return False

        return True
