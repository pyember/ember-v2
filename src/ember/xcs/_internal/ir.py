"""Clean IR design for XCS.

Following Ritchie/Thompson: Simple, composable, does one thing well.
Immutable data structures for functional transformations.

Architecture Philosophy - Why an Intermediate Representation:
    XCS uses an IR to bridge the gap between Python code and optimized execution.
    This design enables sophisticated optimizations impossible at the Python level:

    1. **Graph Analysis**: Identify parallelization opportunities, redundant
       computations, and optimization patterns by analyzing the DAG structure.

    2. **Backend Flexibility**: Manifest in different execution backends without changing user code.

    3. **Transformation Pipelines**: Apply multiple optimization passes cleanly
       with immutable data structures and functional transformations.

    4. **Mixed Computation**: Seamlessly integrate pure computations, model calls,
       and tool invocations in a single graph.

Design Decisions - Why a DAG:
    The Directed Acyclic Graph representation was chosen for specific reasons:

    1. **Natural Data Flow**: Computation naturally flows from inputs to outputs
       without cycles, matching functional programming principles.

    2. **Dependency Analysis**: O(1) dependency lookups enable efficient scheduling
       and parallelization decisions.

    3. **Immutability**: DAG structure with immutable nodes prevents subtle bugs
       from graph mutations during optimization.

    4. **Composability**: Sub-graphs can be extracted, transformed, and recomposed
       without affecting the original graph.

Node Design - Why These Fields:
    Each IRNode contains exactly what's needed for optimization, nothing more:

    - **id**: Unique identifier for graph algorithms
    - **operator**: The actual computation (function or Module)
    - **inputs/outputs**: Variable names for data flow tracking
    - **metadata**: Extensible optimization hints without breaking compatibility

    This minimal design follows the Unix philosophy: simple components that
    compose well. Complex behavior emerges from graph structure, not node complexity.

Performance Characteristics:
    The IR is designed for analysis speed, not execution speed:

    - Node creation: O(1) with frozen dataclasses
    - Dependency lookup: O(1) via output_to_producer index
    - Topological sort: O(V + E) standard algorithm
    - Graph transformation: O(V) for most optimizations
    - Memory: ~200 bytes per node + operator reference

Trade-offs:
    - Analysis time vs execution time: IR adds overhead but enables optimization
    - Immutability vs mutation: Safer but requires copying for modifications
    - Simplicity vs features: No built-in control flow (handled by operators)
    - Static vs dynamic: Graph structure fixed at trace time

Why Not Alternative Designs:
    1. **AST-based**: Was too tied to Python syntax, hard to optimize
    2. **SSA form**: Overkill for our use case, adds complexity
    3. **Mutable graph**: Race conditions in parallel optimization passes
    4. **Tree structure**: Can't represent shared computations efficiently
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Set, Tuple

try:
    import jax.tree_util as tree_util

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    tree_util = None

from ember._internal.module import Module


@dataclass(frozen=True)
class IRNode:
    """A single computation node in the IR graph.

    Immutable, simple, no hidden behavior.
    Just data about what to compute.
    """

    id: str
    operator: Any  # The actual callable (EmberModule or function)
    inputs: Tuple[str, ...]  # Variable names this node reads
    outputs: Tuple[str, ...]  # Variable names this node writes

    # Metadata for optimization, not behavior
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Extract parallelism info from operator structure."""
        # If it's an Ember Module, extract pytree structure
        if isinstance(self.operator, Module):
            try:
                # Use JAX's tree_util to flatten the module if available
                if HAS_JAX and tree_util:
                    flat_values, treedef = tree_util.tree_flatten(self.operator)
                    # Create new metadata dict with pytree info
                    new_metadata = dict(self.metadata)
                    new_metadata["pytree_structure"] = {
                        "flat_values": flat_values,
                        "treedef": treedef,
                    }
                    new_metadata["is_ember_module"] = True
                    object.__setattr__(self, "metadata", new_metadata)
                else:
                    # Without JAX, just mark as module
                    new_metadata = dict(self.metadata)
                    new_metadata["is_ember_module"] = True
                    object.__setattr__(self, "metadata", new_metadata)
            except Exception:
                # If flattening fails, just mark as module
                new_metadata = dict(self.metadata)
                new_metadata["is_ember_module"] = True
                object.__setattr__(self, "metadata", new_metadata)


@dataclass(frozen=True)
class IRGraph:
    """The complete computation graph.

    Just nodes and their connections. Nothing more.
    Immutable for safe transformations.
    """

    nodes: Dict[str, IRNode] = field(default_factory=dict)
    edges: Dict[str, FrozenSet[str]] = field(default_factory=dict)  # node_id -> downstream nodes
    # Index for O(1) dependency lookups: output_var -> producer_node_id
    _output_to_producer: Dict[str, str] = field(default_factory=dict)

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get all nodes that must execute before this one."""
        target_node = self.nodes.get(node_id)
        if not target_node:
            return set()

        dependencies = set()
        # Use index for O(1) lookup per input
        for input_var in target_node.inputs:
            producer_id = self._output_to_producer.get(input_var)
            if producer_id:
                dependencies.add(producer_id)

        return dependencies

    def get_dependents(self, node_id: str) -> FrozenSet[str]:
        """Get all nodes that depend on this one."""
        return self.edges.get(node_id, frozenset())

    def has_independent_branches(self, node_id: str) -> bool:
        """Check if node's dependents can run in parallel."""
        dependents = self.get_dependents(node_id)
        if len(dependents) <= 1:
            return False

        # Check if dependents share any inputs
        dependent_inputs = []
        for dep_id in dependents:
            dep_node = self.nodes.get(dep_id)
            if dep_node:
                dependent_inputs.append(set(dep_node.inputs))

        # If no shared inputs between any pair, they're independent
        for i, inputs1 in enumerate(dependent_inputs):
            for inputs2 in dependent_inputs[i + 1 :]:
                if inputs1 & inputs2:  # Shared inputs
                    return False
        return True

    def topological_sort(self) -> List[str]:
        """Return nodes in execution order."""
        # Simple Kahn's algorithm
        in_degree = {node_id: 0 for node_id in self.nodes}

        # Calculate in-degrees
        for node_id in self.nodes:
            deps = self.get_dependencies(node_id)
            in_degree[node_id] = len(deps)

        # Start with nodes that have no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for dependents
            for dependent in self.get_dependents(current):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    def with_node(self, node: IRNode) -> "IRGraph":
        """Create new graph with additional node (immutable)."""
        new_nodes = dict(self.nodes)
        new_nodes[node.id] = node

        # Update output index
        new_output_to_producer = dict(self._output_to_producer)
        for output in node.outputs:
            new_output_to_producer[output] = node.id

        # Update edges based on data dependencies
        new_edges = dict(self.edges)
        # Use index for O(1) edge building
        for input_var in node.inputs:
            producer_id = new_output_to_producer.get(input_var)
            if producer_id:
                # Add edge from producer to new node
                current_deps = set(new_edges.get(producer_id, frozenset()))
                current_deps.add(node.id)
                new_edges[producer_id] = frozenset(current_deps)

        return IRGraph(nodes=new_nodes, edges=new_edges, _output_to_producer=new_output_to_producer)

    def transform(self, transform_fn) -> "IRGraph":
        """Apply transformation to create new graph."""
        return transform_fn(self)


@dataclass(frozen=True)
class ParallelismInfo:
    """Information about parallelization opportunities."""

    can_vmap: bool = False  # Can be vectorized
    can_pmap: bool = False  # Can be parallelized across devices
    can_batch: bool = False  # Can process batches
    can_parallelize: bool = False  # Has independent branches
    is_pure: bool = True  # No side effects
    estimated_speedup: float = 1.0  # Expected performance gain


def analyze_node_parallelism(node: IRNode) -> ParallelismInfo:
    """Analyze a single node for parallelism opportunities."""
    # Default info
    info_dict = {
        "can_vmap": False,
        "can_pmap": False,
        "can_batch": False,
        "is_pure": True,
        "estimated_speedup": 1.0,
    }

    # Check if it's an EmberModule with pytree structure
    if node.metadata.get("is_ember_module"):
        info_dict["can_vmap"] = True
        info_dict["can_batch"] = True

        # If it has array-like fields, can be pmapped
        if "pytree_structure" in node.metadata:
            flat_values = node.metadata["pytree_structure"].get("flat_values", [])
            for value in flat_values:
                if _is_array_like(value):
                    info_dict["can_pmap"] = True
                    info_dict["estimated_speedup"] = 4.0  # Typical pmap speedup
                    break

        if info_dict["can_vmap"]:
            info_dict["estimated_speedup"] = max(info_dict["estimated_speedup"], 2.0)

    return ParallelismInfo(**info_dict)


def _is_array_like(value: Any) -> bool:
    """Check if value is array/tensor-like."""
    return (
        hasattr(value, "shape")
        or hasattr(value, "__array__")
        or type(value).__name__ in {"Tensor", "Array", "ndarray", "DeviceArray"}
    )
