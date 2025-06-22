# XCS Ultimate Simplification Plan

## Current State Analysis

After examining XCS in detail, it's massively over-engineered:

### 1. Schedulers (5 types, but only need 2)
- `SequentialScheduler` and `TopologicalScheduler` - identical functionality
- `ParallelScheduler` and `WaveScheduler` - identical functionality  
- `NoOpScheduler` - only for testing
- `"auto"` mode - just picks between sequential/parallel

**Solution**: Just have sequential and parallel execution. Delete 3 schedulers.

### 2. JIT Strategies (4 strategies + auto mode)
- `TraceStrategy` - trace-based compilation
- `StructuralStrategy` - structure analysis
- `EnhancedStrategy` - hybrid approach
- Auto mode with scoring - over-complex selection logic

**Solution**: One adaptive JIT that works. Period.

### 3. Graph (376 lines for a DAG)
- Field-level mappings that are rarely used
- Complex edge management
- Metadata everywhere
- Multiple ways to add nodes/edges

**Solution**: Simple DAG with nodes and edges. 50 lines max.

### 4. Dispatcher (Abstraction over ThreadPoolExecutor)
- Wraps ThreadPoolExecutor with timeout/retry logic
- Multiple executor backends ("auto", "thread", "async")
- Complex error handling

**Solution**: Just use ThreadPoolExecutor directly. It already has timeouts.

### 5. Exception Hierarchy (15+ custom exceptions)
```
XCSError
├── CompilationError
├── TraceError
├── TransformError
├── DataFlowError
├── ExecutionError
├── GraphError
├── SchedulerError
├── CacheError
├── TimeoutError
└── ... more
```

**Solution**: Use standard Python exceptions. Maybe 1-2 custom ones max.

## Proposed Simplification

### 1. Execution - Two Functions

```python
def run(graph: Graph, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run graph, parallel if beneficial."""
    if _should_parallelize(graph):
        return _run_parallel(graph, inputs)
    return _run_sequential(graph, inputs)

def run_parallel(graph: Graph, inputs: Dict[str, Any], workers: int = None) -> Dict[str, Any]:
    """Force parallel execution."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Simple wave execution
        ...
```

### 2. Graph - Simple DAG

```python
@dataclass
class Node:
    id: str
    func: Callable
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

class Graph:
    def __init__(self):
        self.nodes = {}
    
    def add(self, func: Callable, deps: List[str] = None) -> str:
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = Node(node_id, func, deps or [])
        return node_id
    
    def topological_sort(self) -> List[str]:
        # Standard topological sort, 20 lines
        ...
```

### 3. JIT - One Strategy

```python
@jit
def my_function(x):
    return x + 1

# That's it. No modes, no strategies, no options.
```

Implementation:
```python
def jit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to trace and optimize
        # If it fails, fall back to original
        try:
            if not hasattr(wrapper, '_compiled'):
                wrapper._compiled = _trace_and_compile(func, args, kwargs)
            return wrapper._compiled(*args, **kwargs)
        except:
            return func(*args, **kwargs)
    return wrapper
```

### 4. No Custom Exceptions

```python
# Just use standard exceptions
raise ValueError("Graph has cycles")
raise RuntimeError("Execution failed")
raise TimeoutError("Execution timed out")
```

### 5. No Dispatcher

```python
# Just use ThreadPoolExecutor directly
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(node.func, inputs) for node in wave]
    results = [f.result(timeout=30) for f in futures]
```

## Benefits

1. **80% less code** - From ~10,000 lines to ~2,000
2. **Easier to understand** - No abstraction layers
3. **Better performance** - Less overhead
4. **Easier to debug** - Standard Python patterns
5. **Easier to test** - Fewer edge cases

## Implementation Order

1. **Phase 1**: Simplify execution (done ✓)
2. **Phase 2**: Remove duplicate schedulers
3. **Phase 3**: Simplify Graph to basic DAG
4. **Phase 4**: Replace Dispatcher with ThreadPoolExecutor
5. **Phase 5**: Unify JIT strategies
6. **Phase 6**: Remove custom exceptions

## Code Size Reduction Estimate

```
Current:
- Schedulers: ~1,500 lines → 200 lines
- JIT strategies: ~2,000 lines → 300 lines  
- Graph: ~1,000 lines → 100 lines
- Dispatcher: ~500 lines → 0 lines
- Exceptions: ~300 lines → 0 lines
- ExecutionOptions: ~350 lines → 0 lines (done ✓)

Total reduction: ~5,650 lines removed
```

## End Result

XCS becomes what it should be:
1. A simple graph executor
2. With optional JIT compilation
3. That runs things in parallel when beneficial

No strategies, no options, no complex abstractions. Just simple, fast execution.