# XCS Clean IR Design

## Core Principles

Following Ritchie/Thompson's Unix philosophy: **Everything is simple, composable, and does one thing well.**

## IR Node Structure

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Set
from ember.core.module import get_tree_functions

@dataclass(frozen=True)
class IRNode:
    """A single computation node in the IR graph.
    
    Immutable, simple, no hidden behavior.
    """
    id: str
    operator: Any  # The actual callable (EmberModule or function)
    inputs: Tuple[str, ...]  # Variable names this node reads
    outputs: Tuple[str, ...]  # Variable names this node writes
    
    # Metadata for optimization, not behavior
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and extract parallelism info from operator."""
        # If it's an EmberModule, we can extract structure
        if hasattr(self.operator, '__dataclass_fields__'):
            # This operator has pytree structure we can analyze
            tree_fns = get_tree_functions(type(self.operator))
            if tree_fns:
                # Store flattened structure for parallelism analysis
                flat, treedef = tree_fns[0](self.operator)
                object.__setattr__(
                    self.metadata, 
                    'pytree_structure',
                    {'flat': flat, 'treedef': treedef}
                )

@dataclass(frozen=True)
class IRGraph:
    """The complete computation graph.
    
    Just nodes and edges, nothing more.
    """
    nodes: Dict[str, IRNode]
    edges: Dict[str, Set[str]]  # node_id -> set of downstream node_ids
    
    def analyze_parallelism(self) -> Dict[str, Set[str]]:
        """Discover parallelism opportunities from structure."""
        parallel_groups = {}
        
        for node_id, node in self.nodes.items():
            # Check if node has pytree structure
            if 'pytree_structure' in node.metadata:
                # This node can be vmapped/pmapped automatically
                parallel_groups[node_id] = {'vmap', 'pmap'}
            
            # Check for independent branches (ensemble pattern)
            if self._has_independent_branches(node_id):
                parallel_groups[node_id] = parallel_groups.get(node_id, set())
                parallel_groups[node_id].add('parallel')
        
        return parallel_groups
    
    def _has_independent_branches(self, node_id: str) -> bool:
        """Check if downstream nodes are independent."""
        children = self.edges.get(node_id, set())
        if len(children) <= 1:
            return False
            
        # Check if children share any inputs
        child_inputs = []
        for child_id in children:
            child = self.nodes[child_id]
            child_inputs.append(set(child.inputs))
        
        # If no shared inputs, they're independent
        for i, inputs1 in enumerate(child_inputs):
            for inputs2 in child_inputs[i+1:]:
                if inputs1 & inputs2:  # Shared inputs
                    return False
        return True

## IR Builder (Hidden from Users)

```python
class IRBuilder:
    """Builds IR from Python functions. Users never see this."""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.var_counter = 0
        
    def add_operation(self, operator: Any, inputs: List[str]) -> List[str]:
        """Add an operation to the graph."""
        node_id = f"op_{len(self.nodes)}"
        
        # Generate output variable names
        outputs = [f"var_{self.var_counter}"]
        self.var_counter += 1
        
        # Create node
        node = IRNode(
            id=node_id,
            operator=operator,
            inputs=tuple(inputs),
            outputs=tuple(outputs)
        )
        
        self.nodes[node_id] = node
        
        # Track edges based on data flow
        for var in inputs:
            # Find which node produced this variable
            producer = self._find_producer(var)
            if producer:
                self.edges.setdefault(producer, set()).add(node_id)
        
        return outputs
    
    def _find_producer(self, var: str) -> Optional[str]:
        """Find which node produced a variable."""
        for node_id, node in self.nodes.items():
            if var in node.outputs:
                return node_id
        return None
    
    def build(self) -> IRGraph:
        """Construct the final immutable graph."""
        return IRGraph(nodes=self.nodes.copy(), edges=self.edges.copy())

## Parallelism Discovery Example

```python
# User writes this:
@module
class Embed:
    model: Any
    def __call__(self, text: str) -> np.ndarray:
        return self.model.encode(text)

@module  
class Score:
    weights: np.ndarray
    def __call__(self, embedding: np.ndarray) -> float:
        return embedding @ self.weights

# User combines them:
def pipeline(texts: List[str]) -> List[float]:
    embedder = Embed(model=my_model)
    scorer = Score(weights=my_weights)
    
    # The IR builder sees this pattern:
    embeddings = [embedder(text) for text in texts]  # Parallel opportunity!
    scores = [scorer(emb) for emb in embeddings]     # Another parallel opportunity!
    
    return scores

# XCS automatically discovers:
# 1. embedder can be vmapped (it has pytree structure)
# 2. The list comprehension is a parallel pattern
# 3. scorer can also be vmapped
# 4. No explicit configuration needed!
```

## Key Insights

1. **Pytree Structure = Parallelism**: Modules with pytree registration can be automatically parallelized
2. **Data Dependencies = Scheduling**: The IR tracks who produces/consumes what
3. **No User Configuration**: Parallelism is discovered, not configured
4. **Immutable IR**: Transformations create new graphs, no mutation
5. **Simple Building Blocks**: Each piece does one thing well

## What This Enables

1. **Automatic Batching**: Detect when operations can be batched
2. **Automatic Parallelism**: Find independent computations
3. **Automatic Fusion**: Combine compatible operations
4. **Zero Configuration**: Users just write Python
5. **Progressive Optimization**: Start simple, optimize based on profiling

## What Users Never See

1. IR nodes and graphs
2. Parallelism analysis  
3. Scheduling decisions
4. Optimization strategies
5. Tree flattening/unflattening

The IR is purely an implementation detail that enables smart execution.