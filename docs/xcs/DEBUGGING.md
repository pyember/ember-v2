# XCS Debugging and Performance Analysis

## Overview

XCS provides two key tools for understanding and debugging your code:
- `@trace`: Execution analysis and bottleneck identification
- `get_jit_stats()`: Performance metrics for optimized code

## @trace - Execution Analysis

The `@trace` decorator helps you understand what happens when your code runs. It's NOT for optimization (use `@jit` for that) - it's for debugging and analysis.

### Basic Usage

```python
from ember.xcs import trace

@trace
def my_pipeline(*, inputs):
    # Your compound AI system
    result1 = operator1(inputs=inputs)
    result2 = operator2(inputs=result1)
    return operator3(inputs=result2)

# Run and get trace information
result = my_pipeline(inputs=data)
```

### Getting Trace Results

```python
@trace(print_summary=True)  # Print summary automatically
def my_function(*, inputs):
    # Your code here
    pass

# Or capture trace data
@trace
def my_function(*, inputs):
    # Your code here
    pass

# The function now has trace data attached
result = my_function(inputs=data)
# Access trace data: result._trace_data (when available)
```

### What Trace Shows You

1. **Execution Timeline**: Which operations ran and for how long
2. **Bottlenecks**: Identifies the slowest operations
3. **Call Counts**: How many times each operator was called
4. **Dependencies**: The actual execution order

### Example Output

```
=== Trace Summary ===
Total operations: 5
Total duration: 2341.2ms

Slowest operations:
  LLMOperator: 2100.3ms
  DataProcessor: 200.1ms
  Aggregator: 40.8ms

Operation breakdown:
  LLMOperator:
    Count: 3
    Total: 2100.3ms
    Avg: 700.1ms
  DataProcessor:
    Count: 1
    Total: 200.1ms
    Avg: 200.1ms
```

### When to Use @trace

- **Debugging slow pipelines**: Find which operators take the most time
- **Understanding execution flow**: See the actual order of operations
- **Identifying redundant calls**: Spot operators called more than expected
- **Before optimization**: Understand baseline performance before adding `@jit`

## get_jit_stats() - JIT Performance Metrics

`get_jit_stats()` shows you how well the `@jit` optimization is working.

### Basic Usage

```python
from ember.xcs import jit, get_jit_stats

@jit
class MyOperator(Operator):
    def forward(self, *, inputs):
        # Your operator logic
        pass

# Use the operator
op = MyOperator()
results = op(inputs=data)

# Get performance statistics
stats = get_jit_stats(op)
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Avg execution time: {stats['avg_execution_time_ms']}ms")
```

### Global Statistics

```python
# Get overall JIT statistics for all functions
global_stats = get_jit_stats()
print(f"Total compilations: {global_stats['total_compilations']}")
print(f"Total cache hits: {global_stats['total_cache_hits']}")
```

### What Stats Tell You

- **Cache Hit Rate**: How often JIT reuses compiled code (higher is better)
- **Compilation Time**: One-time cost of optimization
- **Execution Time**: How fast your optimized code runs
- **Memory Usage**: Cache memory consumption

### When to Use get_jit_stats()

- **Performance monitoring**: Track optimization effectiveness in production
- **A/B testing**: Compare performance with and without `@jit`
- **Memory debugging**: Check if JIT cache is using too much memory
- **Optimization tuning**: Understand if JIT is helping your specific use case

## Complete Example: Debugging a Slow Pipeline

```python
from ember.xcs import jit, trace, get_jit_stats
from ember.api.operators import Operator

# Step 1: Trace to find bottlenecks
@trace(print_summary=True)
def slow_pipeline(*, inputs):
    # Multiple LLM calls
    summaries = [summarize_op(inputs=doc) for doc in inputs["docs"]]
    analysis = analyze_op(inputs={"summaries": summaries})
    return synthesize_op(inputs=analysis)

# Run with trace
result = slow_pipeline(inputs={"docs": documents})
# Output shows LLM calls are the bottleneck

# Step 2: Add JIT optimization
@jit
def optimized_pipeline(*, inputs):
    # Same code, but now optimized
    summaries = [summarize_op(inputs=doc) for doc in inputs["docs"]]
    analysis = analyze_op(inputs={"summaries": summaries})
    return synthesize_op(inputs=analysis)

# Step 3: Monitor optimization effectiveness
result = optimized_pipeline(inputs={"docs": documents})
stats = get_jit_stats(optimized_pipeline)

print(f"Performance improvement: {stats['speedup']}x")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

## Best Practices

### For @trace

1. **Use during development**: Remove `@trace` in production unless needed
2. **Start broad**: Trace entire pipelines before tracing individual operators
3. **Save traces**: Use `save_to="trace.json"` for later analysis
4. **Compare traces**: Run before and after optimization to see improvements

### For get_jit_stats()

1. **Monitor in production**: Periodically check stats to ensure optimization works
2. **Set alerts**: Warn if cache hit rate drops (might indicate issues)
3. **Clean cache if needed**: Reset stats if memory usage grows too large
4. **A/B test**: Compare stats between different implementations

## Common Patterns

### Pattern 1: Development Workflow
```python
# 1. Build with trace
@trace
def my_pipeline(inputs):
    return process(inputs)

# 2. Identify bottlenecks
result = my_pipeline(test_data)  # See what's slow

# 3. Optimize with JIT
@jit
def my_pipeline(inputs):
    return process(inputs)

# 4. Verify improvement
stats = get_jit_stats(my_pipeline)
```

### Pattern 2: Production Monitoring
```python
@jit
class ProductionOperator(Operator):
    def forward(self, *, inputs):
        return self.process(inputs)
    
    def get_health_metrics(self):
        stats = get_jit_stats(self)
        return {
            "jit_cache_hit_rate": stats.get("cache_hit_rate", 0),
            "avg_latency_ms": stats.get("avg_execution_time_ms", 0),
            "is_healthy": stats.get("cache_hit_rate", 0) > 0.8
        }
```

### Pattern 3: Debugging Specific Operations
```python
# Wrap specific operations with trace
@trace
def debug_specific_operation(inputs):
    return problematic_operator(inputs)

# Get detailed timing
result = debug_specific_operation(test_case)
```

## FAQ

### Why is get_jit_stats() a separate function?

`get_jit_stats()` is separate from `@jit` for several reasons:

1. **Separation of Concerns**: `@jit` optimizes code, `get_jit_stats()` monitors it
2. **Production Monitoring**: You can check stats without modifying decorated code
3. **Global Stats**: Can get system-wide statistics, not just per-function
4. **Dynamic Inspection**: Can check stats on any JIT-compiled function at runtime

```python
# You can monitor any JIT function without changing it
stats = get_jit_stats(some_jit_function)

# Or get global statistics
global_stats = get_jit_stats()  # All JIT functions
```

### Should I use @trace in production?

Generally no. `@trace` adds overhead to collect execution data. Use it during:
- Development and debugging
- Performance investigations
- One-off production debugging (remove after)

### Can I use @trace and @jit together?

Yes, but apply them in the right order:

```python
# CORRECT: JIT first, trace second
@trace
@jit
def my_function(inputs):
    pass

# This traces the optimized execution
```

### What's the difference between @trace metrics and get_jit_stats()?

- **@trace**: Shows what happened in a specific execution (timeline, bottlenecks)
- **get_jit_stats()**: Shows aggregate performance over many executions (cache hits, average times)

Use @trace to find problems, @jit to fix them, and get_jit_stats() to verify the fix worked.

## Summary

- **@trace**: For understanding and debugging execution flow
- **get_jit_stats()**: For monitoring optimization effectiveness
- **Use together**: Trace to find problems, JIT to fix them, stats to verify

Remember: These tools are for analysis and monitoring. The actual optimization happens automatically with `@jit`.