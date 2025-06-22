# XCS Simplification Plan

## Vision
Make XCS so simple and elegant that using it feels natural and obvious. Follow the principle: "There should be one—and preferably only one—obvious way to do it."

## Current Problems

### 1. Too Many Execution Paths
- `Dispatcher`, `executor_unified.py`, `ThreadPoolExecutor`, various schedulers
- `xcs_engine.py` vs `unified_engine.py` 
- Multiple ways to configure execution

### 2. Overly Complex JIT System  
- 4 different strategies with overlapping functionality
- Complex strategy selection heuristics
- Multiple decorators doing similar things

### 3. Configuration Overload
- ExecutionOptions has 12+ parameters
- Multiple ways to set options (global, thread-local, context)
- Backward compatibility cruft

### 4. Naming Confusion
- Engine vs Executor vs Scheduler vs Coordinator
- Trace vs Tracer vs Tracing
- Unclear boundaries between concepts

### 5. Exception Complexity
- 8+ exception types with complex inheritance
- Mixed responsibilities (logging + exceptions)
- Redundant context management

## Simplification Actions

### Phase 1: Unify Execution (✅ Partially Complete)
1. ✅ Replace all `ThreadPoolExecutor` usage with `Dispatcher`
2. ❌ Remove `executor_unified.py` (has learning system we don't need)
3. ❌ Merge `unified_engine.py` into `xcs_engine.py`
4. ❌ Remove redundant scheduler implementations

### Phase 2: Simplify JIT
1. ❌ Merge all JIT strategies into one adaptive strategy
2. ❌ Remove strategy selection logic
3. ❌ Single `@jit` decorator that works for everything
4. ❌ Remove `autograph`, `structural_jit`, etc. - just use `@jit`

### Phase 3: Streamline Configuration  
1. ❌ Reduce ExecutionOptions to essential parameters:
   - `parallel: bool = True`
   - `max_workers: Optional[int] = None`
   - `timeout: Optional[float] = None`
2. ❌ Remove thread-local storage complexity
3. ❌ Remove backward compatibility mappings
4. ❌ Single way to configure: context manager

### Phase 4: Clean Up Naming
1. ❌ Pick one concept: "Executor" (not Engine, Scheduler, Coordinator)
2. ❌ Rename files/classes for consistency
3. ❌ Clear module boundaries

### Phase 5: Simplify Exceptions
1. ❌ Reduce to 3 core exceptions:
   - `XCSError` - Base exception
   - `ExecutionError` - Runtime execution failures  
   - `ConfigurationError` - Invalid configuration
2. ❌ Remove complex context management
3. ❌ Separate logging from exceptions

## End State

### One Way to Execute
```python
from ember.xcs import jit, execute

# Simple execution
result = execute(my_function, inputs)

# With options
with execution_options(parallel=True, max_workers=4):
    result = execute(my_function, inputs)
```

### One Way to Optimize
```python
@jit  # Just works, no strategies to choose
def my_function(inputs):
    return process(inputs)
```

### Simple Configuration
```python
# Only what matters
with execution_options(
    parallel=True,      # Use parallel execution
    max_workers=4,      # Worker count
    timeout=30.0        # Timeout in seconds
):
    result = my_function(inputs)
```

### Clear Errors
```python
try:
    result = execute(fn, inputs)
except ExecutionError as e:
    print(f"Execution failed: {e}")
except ConfigurationError as e:
    print(f"Invalid config: {e}")
```

## Benefits
- Drastically reduced cognitive load
- Clear, obvious API
- Easier to maintain and extend
- Better performance (less overhead)
- Improved developer experience

## Next Steps
1. Review and approve this plan
2. Implement Phase 1 (unify execution) 
3. Test thoroughly at each phase
4. Update documentation as we go
5. Deprecate old APIs gracefully