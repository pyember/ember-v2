# XCS Migration Plan: From Current State to Principled Design

## Current State Analysis

### What Works
- Clean architectural components exist
- ParallelismAnalyzer can identify parallel opportunities
- ExecutionEngine has parallel execution logic
- IRBuilder has structure for building graphs

### What's Broken
- `@jit` is a no-op - just calls original function
- Components aren't connected
- No actual tracing implementation
- Complex design docs propose too much magic

### What to Keep
- Existing component structure
- Clean separation of concerns
- Basic algorithms for parallelism detection

### What to Remove
- All mentions of global learning
- Distributed execution plans
- Lock-free algorithm complexity
- Configuration options
- Progressive disclosure

## Migration Steps

### Step 1: Delete Overcomplicated Design Docs

```bash
# Remove the overly complex designs
rm .internal_docs/design_docs/XCS_ENHANCED_DESIGN.md
rm .internal_docs/design_docs/XCS_FINAL_ENHANCED_DESIGN.md

# Keep only the principled design
# XCS_PRINCIPLED_DESIGN.md becomes the canonical design
```

### Step 2: Fix the Core Connection

```python
# ember/xcs/_simple.py
# Replace the current no-op implementation

def jit(func: Optional[Callable] = None) -> Callable:
    """Make function faster by discovering parallelism."""
    if func is None:
        # Handle @jit() syntax
        return jit
    
    # Single cache per function
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Cache key from argument structure
        key = _make_cache_key(args, kwargs)
        
        if key not in cache:
            try:
                # Connect the components!
                builder = IRBuilder()
                graph = builder.trace_function(func, args, kwargs)
                
                analyzer = ParallelismAnalyzer()
                analysis = analyzer.analyze_graph(graph)
                
                if analysis.estimated_speedup > 1.2:
                    engine = ExecutionEngine()
                    cache[key] = engine.compile(graph, analysis)
                else:
                    cache[key] = None
            except Exception:
                # Any failure -> no optimization
                cache[key] = None
        
        # Execute
        if cache[key] is not None:
            try:
                return cache[key].execute(args, kwargs)
            except Exception:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    # Simple stats
    wrapper.stats = lambda: {
        'cached_patterns': len(cache),
        'optimized': sum(1 for v in cache.values() if v is not None)
    }
    
    return wrapper
```

### Step 3: Implement Simple Tracing

```python
# ember/xcs/_internal/ir_builder.py
# Replace AST analysis with execution tracing

class IRBuilder:
    def trace_function(self, func: Callable, args: tuple, kwargs: dict) -> IRGraph:
        """Trace function execution to build graph."""
        # For now, create a simple graph representing the function
        # This is the key piece that needs implementation
        
        # Step 1: Try to identify parallel patterns
        if self._is_list_comprehension_pattern(func, args):
            return self._build_parallel_map_graph(func, args)
        
        # Step 2: Try to trace with proxy objects
        try:
            return self._trace_with_proxies(func, args, kwargs)
        except Exception:
            # Step 3: Fall back to single node
            return self._build_single_node_graph(func)
    
    def _trace_with_proxies(self, func, args, kwargs):
        """Trace using proxy objects to track operations."""
        # Create proxy objects
        proxy_args = self._create_proxies(args)
        operations = []
        
        # Monkey-patch model calls to record operations
        with OperationRecorder(operations):
            func(*proxy_args, **kwargs)
        
        # Build graph from recorded operations
        return self._operations_to_graph(operations)
```

### Step 4: Simplify Parallelism Analysis

```python
# ember/xcs/_internal/parallelism.py
# Focus on what actually works

class ParallelismAnalyzer:
    def analyze_graph(self, graph: IRGraph) -> GraphParallelismAnalysis:
        """Find simple parallelism patterns."""
        # Look for independent operations only
        parallel_groups = []
        
        # Find nodes with no dependencies on each other
        for nodes in graph.get_same_depth_nodes():
            if self._are_independent(nodes):
                parallel_groups.append(nodes)
        
        # Conservative speedup estimation
        speedup = 1.0
        if parallel_groups:
            largest_group = max(len(g) for g in parallel_groups)
            # Account for thread overhead
            speedup = max(1.0, largest_group * 0.8)
        
        return GraphParallelismAnalysis(
            parallel_groups=parallel_groups,
            estimated_speedup=speedup
        )
```

### Step 5: Simple Execution Engine

```python
# ember/xcs/_internal/engine.py
# Use standard Python threading, no magic

class ExecutionEngine:
    def compile(self, graph: IRGraph, analysis: GraphParallelismAnalysis) -> CompiledExecutor:
        """Create appropriate executor."""
        if analysis.parallel_groups:
            return ParallelExecutor(graph, analysis.parallel_groups)
        else:
            return SequentialExecutor(graph)

class ParallelExecutor:
    def __init__(self, graph: IRGraph, parallel_groups: List[Set[str]]):
        self.graph = graph
        self.parallel_groups = parallel_groups
    
    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Execute with simple thread pool."""
        context = {}
        
        # Map inputs
        for i, arg in enumerate(args):
            context[f'arg_{i}'] = arg
        
        # Execute nodes
        with ThreadPoolExecutor(max_workers=4) as executor:
            for node in self.graph.topological_sort():
                if any(node.id in group for group in self.parallel_groups):
                    # Part of parallel group - submit to pool
                    future = executor.submit(self._execute_node, node, context)
                    context[node.output] = future
                else:
                    # Sequential execution
                    result = self._execute_node(node, context)
                    context[node.output] = result
        
        # Resolve futures
        for key, value in context.items():
            if isinstance(value, Future):
                context[key] = value.result()
        
        # Return final result
        return context.get('result')
```

### Step 6: Remove Unnecessary Files

```bash
# Remove overengineered components
rm src/ember/xcs/jit/strategies/ir_based.py
rm src/ember/xcs/jit/strategies/pytree_aware.py  
rm src/ember/xcs/jit/strategies/tracing.py
rm src/ember/xcs/_internal/distributed.py  # If it exists
rm src/ember/xcs/_internal/global_learning.py  # If it exists

# Keep only what's needed
# - _simple.py (the API)
# - _internal/ir_builder.py (builds graphs)
# - _internal/parallelism.py (finds parallel ops)
# - _internal/engine.py (executes graphs)
```

### Step 7: Update Tests

```python
# tests/integration/xcs/test_jit_basic.py
# Simple, clear tests

def test_jit_makes_parallel_code_faster():
    """The one thing we promise: parallel code runs in parallel."""
    
    @jit
    def parallel_work(n):
        results = []
        for i in range(n):
            # These are independent - should parallelize
            a = expensive_operation(i)
            b = expensive_operation(i + 1)
            c = expensive_operation(i + 2)
            results.append(a + b + c)
        return results
    
    # Time with and without jit
    start = time.time()
    result1 = parallel_work.__wrapped__(4)  # Original
    time1 = time.time() - start
    
    start = time.time()
    result2 = parallel_work(4)  # Optimized
    time2 = time.time() - start
    
    # Should be faster (but still correct)
    assert result1 == result2
    assert time2 < time1 * 0.7  # At least 30% faster

def test_jit_preserves_semantics():
    """Optimization never changes behavior."""
    
    @jit
    def complex_function(x, y, z=10):
        if x > y:
            return model_a(x) + z
        else:
            return model_b(y) * z
    
    # Test many input combinations
    for x, y, z in test_cases:
        assert complex_function(x, y, z) == complex_function.__wrapped__(x, y, z)

def test_jit_handles_failures_gracefully():
    """Any failure -> fall back to original."""
    
    @jit
    def sometimes_fails(x):
        if random.random() > 0.5:
            raise ValueError("Random failure")
        return process(x)
    
    # Should never crash, just use original
    results = []
    for i in range(100):
        try:
            results.append(sometimes_fails(i))
        except ValueError:
            pass  # Expected
    
    # Should have some results
    assert len(results) > 0
```

## Timeline

### Week 1: Core Connection
- Monday: Update _simple.py to connect components
- Tuesday: Implement basic tracing in IRBuilder
- Wednesday: Simplify ParallelismAnalyzer
- Thursday: Fix ExecutionEngine
- Friday: Basic integration tests

### Week 2: Make It Work
- Monday-Tuesday: Handle common Python patterns
- Wednesday-Thursday: Improve error handling
- Friday: Performance testing

### Week 3: Make It Right
- Monday-Tuesday: Comprehensive test suite
- Wednesday-Thursday: Documentation
- Friday: Code review and cleanup

### Week 4: Make It Fast
- Monday-Tuesday: Profile and optimize hot paths
- Wednesday-Thursday: Real-world testing
- Friday: Release preparation

## Definition of Done

1. **@jit works**: Makes parallel code actually run in parallel
2. **Simple implementation**: Can be understood in one sitting
3. **No magic**: Behavior is predictable and explicit
4. **Comprehensive tests**: Edge cases, performance, errors all tested
5. **Clean code**: Follows Google Python Style Guide
6. **No regressions**: All existing Ember code still works

## What We're NOT Doing

Per CLAUDE.md principles:

1. **NOT** adding configuration options
2. **NOT** implementing global learning
3. **NOT** adding distributed execution
4. **NOT** using lock-free algorithms
5. **NOT** building progressive disclosure
6. **NOT** making behavior adaptive
7. **NOT** adding magic

## Final State

A simple, working implementation where:

```python
@jit
def my_parallel_function(data):
    return [model(x) for x in data]

# Just works - runs model calls in parallel
result = my_parallel_function(data)
```

That's it. No more, no less.