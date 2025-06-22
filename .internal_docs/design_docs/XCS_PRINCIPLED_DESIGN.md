# XCS Principled Design: Simple, Explicit, and Connected

## Executive Summary

This document presents the principled approach to fixing XCS, following CLAUDE.md guidelines and the actual practices of Dean & Ghemawat. The core insight: **connect what exists, don't add complexity**.

## Core Principles

### 1. Explicit Over Magic
- No hidden behavior changes
- No runtime adaptation
- Same input always produces same output
- Clear, predictable optimization

### 2. One Obvious Way
- Single API: `@jit`
- No configuration options
- No progressive disclosure
- It works or it doesn't

### 3. Root-Node Fix
- The problem: components exist but aren't connected
- The solution: wire them together
- Not needed: new algorithms, global learning, distributed execution

### 4. Measure and Iterate
- Start simple
- Measure actual performance
- Add complexity only where data shows need
- No premature optimization

## The Design

### API (Complete)

```python
from ember.xcs import jit

@jit
def my_function(x):
    return model(x)

# That's it. No other options.
```

### Implementation

```python
# ember/xcs/_simple.py
def jit(func: Callable) -> Callable:
    """Make function faster by discovering parallelism.
    
    Traces function execution, analyzes for parallel opportunities,
    and executes with optimization if beneficial. Falls back to
    original function if optimization fails or provides no benefit.
    
    Args:
        func: Function to optimize
        
    Returns:
        Optimized function with same signature and semantics
    """
    # Permanent optimization decision for this function object
    optimization_decision = None
    optimization_cache = {}
    executor_pool = None
    
    @functools.wraps(func)
    def optimized_func(*args, **kwargs):
        nonlocal optimization_decision, executor_pool
        
        # Make permanent decision on first call
        if optimization_decision is None:
            try:
                # One-shot tracing attempt
                tracer = PythonTracer()  # Uses sys.settrace
                graph = tracer.trace_function(func, args, kwargs)
                
                analyzer = ParallelismAnalyzer()
                analysis = analyzer.analyze_graph(graph)
                
                # Decision: can we optimize this function?
                optimization_decision = len(analysis.parallel_groups) > 0
            except TracingError:
                # Can't trace - permanent disable
                optimization_decision = False
        
        # If we can't optimize, just run original
        if not optimization_decision:
            return func(*args, **kwargs)
        
        # Get cache key from comprehensive signature
        cache_key = _make_cache_key(func, args, kwargs)
        
        # Check cache
        if cache_key not in optimization_cache:
            try:
                # Build and compile for this specific argument pattern
                graph = tracer.trace_function(func, args, kwargs)
                analysis = analyzer.analyze_graph(graph)
                
                engine = ExecutionEngine()
                executor = engine.compile(graph, analysis)
                optimization_cache[cache_key] = executor
                
                # Lazy-initialize thread pool if needed
                if executor_pool is None and analysis.needs_parallelism:
                    executor_pool = ThreadPoolExecutor(
                        max_workers=_compute_optimal_workers(analysis)
                    )
                    
            except Exception:
                # Compilation failed for these args - use original
                optimization_cache[cache_key] = None
        
        # Execute
        executor = optimization_cache[cache_key]
        if executor is not None:
            # Just execute - no retry logic, fail fast like JAX
            return executor.execute(args, kwargs, executor_pool)
        else:
            # No optimization for these args
            return func(*args, **kwargs)
    
    # Cleanup on deletion
    def cleanup():
        if executor_pool:
            executor_pool.shutdown(wait=False)
    
    import atexit
    atexit.register(cleanup)
    
    # Add simple introspection
    optimized_func._xcs_cache = optimization_cache
    optimized_func.stats = lambda: _get_stats(optimization_cache)
    
    return optimized_func

def _make_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Create comprehensive cache key from function and arguments."""
    import hashlib
    import inspect
    
    key_parts = [
        # Function identity and version
        id(func),
        hash(inspect.getsource(func)) if hasattr(func, '__code__') else id(func),
        
        # Argument signatures (not values)
        _signature_of_args(args),
        _signature_of_kwargs(kwargs),
        
        # Model identities
        _extract_model_ids(args, kwargs)
    ]
    
    return hashlib.sha256(str(key_parts).encode()).hexdigest()

def _signature_of_args(args: tuple) -> tuple:
    """Get type signature of arguments."""
    signatures = []
    for arg in args:
        if hasattr(arg, 'shape'):  # Array-like
            signatures.append(('array', arg.shape, arg.dtype))
        elif isinstance(arg, (list, tuple)):
            signatures.append((type(arg).__name__, len(arg)))
        elif isinstance(arg, ModelBinding):
            signatures.append(('model', arg.model_id))
        else:
            signatures.append(type(arg).__name__)
    return tuple(signatures)

def _compute_optimal_workers(analysis: ParallelismAnalysis) -> int:
    """Compute optimal number of workers based on workload."""
    if analysis.has_io_bound_operations:
        # I/O bound (LLM calls) - high parallelism
        return min(analysis.max_parallel_operations, 100)
    else:
        # CPU bound - limit to CPU count
        return min(analysis.max_parallel_operations, cpu_count())
```

### Component Connections

The key fix is connecting existing components:

```python
# ember/xcs/_internal/ir_builder.py
class IRBuilder:
    def trace_function(self, func: Callable, args: tuple, kwargs: dict) -> IRGraph:
        """Build IR graph from function execution."""
        # Use existing Python introspection, not AST
        tracer = ExecutionTracer()
        with tracer.tracing():
            # Execute with proxy objects
            proxy_args = self._create_proxies(args)
            proxy_result = func(*proxy_args, **kwargs)
        
        # Build graph from trace
        return self._build_graph(tracer.operations)
```

```python
# ember/xcs/_internal/parallelism.py
class ParallelismAnalyzer:
    def analyze_graph(self, graph: IRGraph) -> ParallelismAnalysis:
        """Analyze graph for parallelism opportunities."""
        # Find independent operations
        parallel_groups = self._find_independent_ops(graph)
        
        # Estimate speedup realistically
        speedup = self._estimate_real_speedup(parallel_groups)
        
        return ParallelismAnalysis(
            parallel_groups=parallel_groups,
            estimated_speedup=speedup
        )
    
    def _estimate_real_speedup(self, groups: List[Set[str]]) -> float:
        """Conservative speedup estimation."""
        if not groups:
            return 1.0
        
        # Account for threading overhead
        max_parallel = max(len(g) for g in groups)
        overhead = 0.1  # 10% overhead for thread management
        
        return max(1.0, max_parallel * (1 - overhead))
```

```python
# ember/xcs/_internal/engine.py
class ExecutionEngine:
    def compile(self, graph: IRGraph, analysis: ParallelismAnalysis) -> Executor:
        """Create executor for graph."""
        if analysis.parallel_groups:
            return ParallelExecutor(graph, analysis)
        else:
            return SequentialExecutor(graph)

class ParallelExecutor:
    def __init__(self, graph: IRGraph, analysis: ParallelismAnalysis):
        self.graph = graph
        self.parallel_groups = analysis.parallel_groups
        self.execution_order = graph.topological_sort()
    
    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Execute with discovered parallelism."""
        context = ExecutionContext(args, kwargs)
        
        # Execute each group
        for group in self.execution_order:
            if group in self.parallel_groups:
                # Parallel execution
                self._execute_parallel(group, context)
            else:
                # Sequential execution
                self._execute_sequential(group, context)
        
        return context.get_result()
    
    def _execute_parallel(self, nodes: Set[str], context: ExecutionContext, pool: ThreadPoolExecutor):
        """Execute nodes in parallel preserving sequential error semantics."""
        # Sort nodes to ensure deterministic execution order
        sorted_nodes = sorted(nodes)
        
        # Submit all tasks
        futures = {}
        for node_id in sorted_nodes:
            node = self.graph.nodes[node_id]
            future = pool.submit(self._execute_node, node, context)
            futures[node_id] = future
        
        # Collect results in submission order to preserve error timing
        for node_id in sorted_nodes:
            future = futures[node_id]
            try:
                result = future.result()
                context.set_result(node_id, result)
            except Exception as e:
                # Cancel remaining futures to preserve sequential semantics
                for remaining_id in sorted_nodes:
                    if remaining_id != node_id:
                        futures[remaining_id].cancel()
                
                # Propagate the exact exception
                if isinstance(e, UserFunctionError):
                    raise e.original_exception
                else:
                    raise
```

## What This Design Does NOT Include

Following the principle of simplicity:

1. **No global learning** - Each function optimizes independently
2. **No distributed execution** - Single machine only
3. **No configuration options** - It just works
4. **No progressive disclosure** - One API
5. **No runtime adaptation** - Predictable behavior
6. **No lock-free algorithms** - Use Python's standard threading
7. **No custom memory management** - Let Python handle it
8. **No SIMD optimizations** - Focus on parallelism first

## Testing Strategy

```python
def test_semantic_preservation():
    """Optimization preserves function behavior."""
    @jit
    def original(x, y):
        a = model1(x)
        b = model2(y)
        return combine(a, b)
    
    # Same result with and without jit
    result1 = original.__wrapped__(1, 2)  # Original
    result2 = original(1, 2)  # Optimized
    assert result1 == result2

def test_parallel_execution():
    """Parallel operations execute in parallel."""
    call_times = []
    
    def track_time(name):
        def op(x):
            call_times.append((name, time.time()))
            time.sleep(0.1)
            return x
        return op
    
    @jit
    def parallel_ops(x):
        a = track_time('op1')(x)
        b = track_time('op2')(x)  
        c = track_time('op3')(x)
        return a + b + c
    
    result = parallel_ops(1)
    
    # Check they started at ~same time
    times = [t for _, t in call_times]
    assert max(times) - min(times) < 0.05

def test_fallback_on_error():
    """Falls back to original on optimization failure."""
    @jit
    def unreliable(x):
        if random.random() > 0.5:
            raise ValueError("Random failure")
        return x * 2
    
    # Should work despite optimization challenges
    results = [unreliable(i) for i in range(10)]
    assert all(r == i * 2 for i, r in enumerate(results) if r is not None)
```

## Implementation Plan

### Week 1: Core Connection
1. Fix IRBuilder to actually trace execution
2. Fix ParallelismAnalyzer to find real parallel opportunities
3. Fix ExecutionEngine to execute parallel groups
4. Connect them in the @jit decorator

### Week 2: Robustness
1. Handle all Python constructs in tracing
2. Improve cache key generation
3. Add comprehensive error handling
4. Ensure thread safety

### Week 3: Testing
1. Semantic preservation tests
2. Performance validation tests
3. Error handling tests
4. Real-world workload tests

### Week 4: Polish
1. Performance profiling and optimization
2. Documentation
3. Integration tests with real Ember code
4. Release preparation

## Success Criteria

1. **It works**: @jit makes parallel functions faster
2. **It's predictable**: Same behavior every time
3. **It's simple**: No configuration needed
4. **It's robust**: Falls back gracefully on any failure
5. **It's measured**: Clear stats show when it helps

## What Success Looks Like

```python
# User code
@jit
def process_batch(items):
    return [expensive_model(item) for item in items]

# Just works - 3x faster on 4 cores
result = process_batch(data)

# Optional introspection
print(process_batch.stats())
# {'cache_entries': 1, 'optimized': 1, 'parallel_groups': 1, 'speedup': 3.2}
```

## Conclusion

This principled design:
- Solves the actual problem (disconnected components)
- Follows CLAUDE.md principles (explicit, simple, one way)
- Matches Dean/Ghemawat's approach (simple, measured, iterative)
- Can be implemented in 4 weeks
- Provides real value to users

No magic. No complexity. Just making parallel code run in parallel.