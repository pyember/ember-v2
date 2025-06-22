# XCS Execution API Migration Guide

## Summary

We've simplified the XCS execution API from a complex ExecutionOptions object with 12+ parameters to a simple function with just 2 optional parameters.

## Old API

```python
# Complex ExecutionOptions
from ember.xcs.engine import ExecutionOptions, execute_graph

options = ExecutionOptions(
    scheduler="parallel",
    max_workers=4,
    timeout_seconds=30,
    use_parallel=True,
    enable_caching=True,
    trace_execution=False,
    collect_metrics=False,
    # ... many more options
)

result = execute_graph(graph, inputs, options=options)

# Or with context manager
from ember.xcs.engine import execution_options

with execution_options(scheduler="parallel", max_workers=4):
    result = execute_graph(graph, inputs)
```

## New API

```python
from ember.xcs.engine import execute_graph

# Default: automatic parallel execution
result = execute_graph(graph, inputs)

# Sequential execution
result = execute_graph(graph, inputs, parallel=False)

# Parallel with specific worker count
result = execute_graph(graph, inputs, parallel=4)

# With timeout
result = execute_graph(graph, inputs, parallel=True, timeout=30.0)
```

## Migration Examples

### Example 1: Sequential Execution

```python
# Old
options = ExecutionOptions(scheduler="sequential")
result = execute_graph(graph, inputs, options=options)

# New
result = execute_graph(graph, inputs, parallel=False)
```

### Example 2: Parallel with Worker Count

```python
# Old
options = ExecutionOptions(scheduler="parallel", max_workers=8)
result = execute_graph(graph, inputs, options=options)

# New
result = execute_graph(graph, inputs, parallel=8)
```

### Example 3: Context Manager

```python
# Old
with execution_options(scheduler="parallel", max_workers=4):
    result1 = execute_graph(graph1, inputs1)
    result2 = execute_graph(graph2, inputs2)

# New - just pass parameters directly
result1 = execute_graph(graph1, inputs1, parallel=4)
result2 = execute_graph(graph2, inputs2, parallel=4)
```

## Backward Compatibility

The old API is still available for backward compatibility:

1. `ExecutionOptions` class still exists
2. `execution_options` context manager still works
3. `execute_graph_unified()` accepts old-style options

However, new code should use the simplified API.

## Benefits

1. **Simpler**: 2 parameters instead of 12+
2. **Clearer**: `parallel=False` is more obvious than `scheduler="sequential"`
3. **Type-safe**: `parallel: Union[bool, int]` is clearer than string schedulers
4. **Less code**: No need to import ExecutionOptions
5. **Faster**: No option validation overhead

## What We Removed

These parameters are no longer exposed in the simple API:
- `device_strategy` - Always "auto"
- `enable_caching` - Always enabled
- `trace_execution` - Use debug tools instead
- `collect_metrics` - Use profiling tools instead
- `debug` - Use logging instead
- `fail_fast` - Always fails fast
- `continue_on_error` - Use try/except
- `return_partial_results` - Not needed

If you absolutely need these advanced options, use `execute_graph_unified()` with ExecutionOptions.

## Philosophy

Following the principle of "one obvious way to do things", the new API makes the common case (parallel execution) the default and the special case (sequential or limited parallelism) explicit and simple.