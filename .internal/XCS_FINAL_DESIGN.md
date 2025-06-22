# XCS Final Design: Simple Yet Powerful

## Core Question: What Features Do We Keep?

### Features We MUST Keep (Core Power of XCS)

1. **Global Graph Optimization** ✅
   - Topological sorting for correct execution order
   - Wave-based parallelism discovery
   - Automatic parallel execution of independent nodes

2. **Automatic Parallelism Discovery** ✅
   - Analyze graph structure to find parallelizable subgraphs
   - Execute independent operations concurrently
   - No manual annotation needed

3. **JIT Compilation** ✅
   - Trace execution patterns
   - Optimize hot paths
   - Cache compiled graphs

4. **Transformations** ✅
   - vmap for vectorization
   - pmap for parallel mapping
   - Composable transformations

### Features We Should REMOVE (Unnecessary Complexity)

1. **Multiple Scheduler Types** ❌
   - Just need sequential and parallel
   - Auto-selection based on graph structure

2. **Complex Options Objects** ❌
   - Simple function parameters
   - Defaults that work 99% of the time

3. **Abstract Base Classes Everywhere** ❌
   - Use functions and simple classes
   - Duck typing over inheritance

4. **Custom Exception Hierarchy** ❌
   - Use Python's built-in exceptions
   - Maybe 1-2 domain-specific ones max

## Proposed Final Architecture

### 1. Graph Execution Core (200 lines)

```python
class Graph:
    """Simple DAG with automatic parallelism discovery."""
    
    def __init__(self):
        self.nodes = {}
        
    def add(self, func, deps=None):
        """Add node, return ID."""
        node_id = str(uuid.uuid4())[:8]
        self.nodes[node_id] = Node(func, deps or [])
        return node_id
        
    def __call__(self, inputs, parallel=True):
        """Execute graph with automatic optimization."""
        # Discover parallelism automatically
        waves = self._analyze_parallelism()
        
        if not parallel or len(waves) == len(self.nodes):
            # Sequential execution
            return self._run_sequential(inputs, waves)
        else:
            # Parallel execution with discovered structure
            return self._run_parallel(inputs, waves)
    
    def _analyze_parallelism(self):
        """Discover parallel execution opportunities.
        
        Returns waves where each wave contains nodes that can run in parallel.
        This is where the "magic" happens - global optimization!
        """
        # Topological sort ensures correctness
        topo_order = self._topological_sort()
        
        # Group into waves based on dependencies
        waves = []
        completed = set()
        
        for node_id in topo_order:
            # Can this node run now?
            deps_satisfied = all(d in completed for d in self.nodes[node_id].deps)
            
            if deps_satisfied:
                # Add to current wave or start new wave
                if waves and self._can_add_to_wave(node_id, waves[-1], completed):
                    waves[-1].append(node_id)
                else:
                    waves.append([node_id])
            
            completed.add(node_id)
            
        return waves
```

### 2. JIT Compilation (150 lines)

```python
def jit(func):
    """Single adaptive JIT that just works."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to trace and compile
        if not hasattr(wrapper, '_compiled'):
            try:
                # Trace execution
                with tracing():
                    trace = func(*args, **kwargs)
                
                # Build optimized graph
                graph = build_graph(trace)
                
                # Apply optimizations
                graph = optimize_graph(graph)
                
                wrapper._compiled = graph
            except:
                # Can't trace? Just use original
                wrapper._compiled = None
        
        # Execute compiled version or fall back
        if wrapper._compiled:
            return wrapper._compiled(combine_args(args, kwargs))
        else:
            return func(*args, **kwargs)
    
    return wrapper

def optimize_graph(graph):
    """Apply global optimizations."""
    # Fuse compatible operations
    graph = fuse_operations(graph)
    
    # Identify parallel patterns
    graph = mark_parallel_regions(graph)
    
    # Eliminate redundant computation
    graph = eliminate_common_subexpressions(graph)
    
    return graph
```

### 3. Pattern Discovery (100 lines)

```python
def discover_patterns(graph):
    """Automatically discover parallelizable patterns.
    
    This is the key innovation - we analyze the graph structure
    to find map-like operations, reductions, etc.
    """
    patterns = {
        'maps': [],      # Independent operations on collection
        'reduces': [],   # Aggregation operations
        'broadcasts': [] # One-to-many operations
    }
    
    for node_id, node in graph.nodes.items():
        # Detect map pattern: multiple similar operations
        if is_map_pattern(node, graph):
            patterns['maps'].append(node_id)
            
        # Detect reduce pattern: many-to-one
        elif is_reduce_pattern(node, graph):
            patterns['reduces'].append(node_id)
            
        # Detect broadcast: one-to-many
        elif is_broadcast_pattern(node, graph):
            patterns['broadcasts'].append(node_id)
    
    return patterns

def is_map_pattern(node, graph):
    """Detect if node is part of a map operation."""
    # Look for siblings with same function but different inputs
    siblings = find_siblings(node, graph)
    return len(siblings) > 1 and all(
        s.func == node.func for s in siblings
    )
```

### 4. Transformations (150 lines)

```python
def vmap(func, in_axes=0):
    """Vectorize function over inputs."""
    def vmapped(*args):
        # Build graph for vectorized execution
        graph = Graph()
        
        # Analyze input structure
        batch_size = get_batch_size(args, in_axes)
        
        # Create parallel nodes for each batch element
        nodes = []
        for i in range(batch_size):
            batch_args = get_batch_element(args, i, in_axes)
            node_id = graph.add(lambda: func(*batch_args))
            nodes.append(node_id)
        
        # Execute with automatic parallelism
        results = graph(args)
        
        # Stack results
        return stack_results(results, nodes)
    
    return vmapped

def pmap(func):
    """Parallel map with automatic sharding."""
    def pmapped(inputs):
        graph = Graph()
        
        # Create nodes for each input
        nodes = [graph.add(lambda x=x: func(x)) for x in inputs]
        
        # Graph execution automatically parallelizes!
        results = graph({})
        
        return [results[n] for n in nodes]
    
    return pmapped
```

## Key Insights: How We Keep the Power

### 1. **Automatic Parallelism Discovery**
The graph analyzer automatically finds parallelizable subgraphs by examining dependencies. No manual annotation needed!

### 2. **Global Optimization**
By building a complete graph before execution, we can:
- Fuse operations
- Eliminate redundancy  
- Optimize data movement
- Batch similar operations

### 3. **Smart Defaults**
- `parallel=True` by default
- Automatic worker count based on graph structure
- Timeout only when needed

### 4. **Composability**
Simple functions compose naturally:
```python
@jit
@vmap
def process(x):
    return transform(x)

# Automatically vectorized AND JIT compiled!
```

## What This Achieves

### Simplicity
```python
# Build
graph = Graph()
n1 = graph.add(preprocess)
n2 = graph.add(compute, deps=[n1])

# Execute  
result = graph(inputs)  # Automatic parallelism!
```

### Power
- Discovers parallelism automatically
- Optimizes globally across the graph
- JIT compiles hot paths
- Handles complex patterns (map, reduce, broadcast)

### Performance
- Less overhead than current system
- Direct use of ThreadPoolExecutor
- Minimal abstraction penalty
- Smart batching of operations

## Implementation Plan

### Phase 1: Core Graph Execution
1. Implement Graph class with automatic parallelism discovery
2. Replace current executor with simple implementation
3. Verify performance parity

### Phase 2: Unified JIT
1. Single adaptive JIT strategy
2. Automatic tracing and compilation
3. Smart fallback for untraceable code

### Phase 3: Pattern Recognition
1. Automatic discovery of map/reduce patterns
2. Optimization of discovered patterns
3. Integration with transformations

### Phase 4: Cleanup
1. Delete old scheduler code
2. Remove ExecutionOptions completely
3. Simplify exception handling

## The Magic: It's All in the Graph Analysis

The key insight is that **all the power comes from graph analysis**, not from complex APIs:

1. **Dependency Analysis** → Correct execution order
2. **Wave Discovery** → Automatic parallelism
3. **Pattern Recognition** → Optimization opportunities
4. **Global View** → Cross-operation optimization

We don't need 12 parameters and 5 schedulers. We need smart graph analysis and simple execution.

## Example: Real-World Pipeline

```python
# Define pipeline
graph = Graph()

# Data loading (parallel)
load1 = graph.add(load_data, source="dataset1")
load2 = graph.add(load_data, source="dataset2")

# Preprocessing (parallel, automatic!)
prep1 = graph.add(preprocess, deps=[load1])
prep2 = graph.add(preprocess, deps=[load2])

# Merge
merged = graph.add(merge_datasets, deps=[prep1, prep2])

# Analysis (automatically parallelized if possible)
results = graph.add(analyze, deps=[merged])

# Execute - all parallelism discovered automatically!
output = graph({"config": config})
```

The graph automatically:
- Runs load1 and load2 in parallel
- Runs prep1 and prep2 in parallel
- Waits for merge
- Optimizes the entire pipeline

No scheduler selection. No options. Just smart execution.