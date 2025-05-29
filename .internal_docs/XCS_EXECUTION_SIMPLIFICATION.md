# XCS Execution Simplification

## Current Problem

ExecutionOptions has 12+ parameters:
- use_parallel
- max_workers
- device_strategy
- enable_caching
- trace_execution
- timeout_seconds
- collect_metrics
- debug
- scheduler
- executor
- fail_fast
- ...and more

This is absurd. Most code just wants to run things, optionally in parallel.

## Proposed Simplification

### Option 1: Just Functions (Recommended)

```python
# Default: run in parallel if beneficial
result = xcs.run(graph, inputs)

# Force sequential
result = xcs.run_sequential(graph, inputs)

# Force parallel with optional worker count
result = xcs.run_parallel(graph, inputs, workers=4)

# That's it. No options objects.
```

### Option 2: Single Parameter

```python
# If we must have options, just one boolean
result = xcs.run(graph, inputs, parallel=True)
result = xcs.run(graph, inputs, parallel=False)

# Or with workers
result = xcs.run(graph, inputs, parallel=4)  # 4 workers
```

### Option 3: Kill ExecutionOptions Entirely

Just execute graphs. The system should be smart enough to figure out if parallelism helps.

```python
result = execute_graph(graph, inputs)
# System automatically decides based on graph structure
```

## What We're Deleting

1. **ExecutionOptions class** - 346 lines of over-abstraction
2. **execution_options context manager** - Nobody needs this
3. **All the validation** - If you pass bad values, get bad results
4. **Thread-local storage** - Unnecessary complexity
5. **Global options** - Just pass what you need

## Implementation Plan

1. Replace ExecutionOptions with simple function parameters
2. Delete execution_options.py entirely
3. Update execute_graph to take just `parallel: bool | int = True`
4. Remove all the scheduler type strings - just use parallel or not
5. Delete all the validation code

## Code After Simplification

```python
def execute_graph(
    graph: Graph,
    inputs: Dict[str, Any],
    parallel: Union[bool, int] = True
) -> Dict[str, Any]:
    """Execute a graph, optionally in parallel.
    
    Args:
        graph: The graph to execute
        inputs: Input data
        parallel: True for auto parallel, False for sequential, 
                 or int for specific worker count
    """
    if parallel is False:
        return _execute_sequential(graph, inputs)
    
    workers = None if parallel is True else parallel
    return _execute_parallel(graph, inputs, workers)
```

That's it. No options objects, no context managers, no thread-local state.

## Benefits

1. **Simpler API** - One obvious way to do things
2. **Less code** - Delete 500+ lines
3. **Easier to understand** - No option objects to construct
4. **Better performance** - No option validation overhead
5. **Cleaner tests** - No need to test option combinations

## Migration

```python
# Before
with execution_options(scheduler="parallel", max_workers=4):
    result = execute_graph(graph, inputs)

# After  
result = execute_graph(graph, inputs, parallel=4)
```

This is what Jeff Dean and Sanjay would build - simple, fast, obvious.