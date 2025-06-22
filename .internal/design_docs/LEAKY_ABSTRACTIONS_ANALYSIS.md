# Leaky Abstractions in Natural API Implementation

## 1. Error Messages Expose Internal Details

### Issue: Adapter Errors Show Internal Conversion
```python
# From adapters.py
raise AdapterError(
    f"Error adapting {func.__name__} from internal format: {e}"
)
```

**Problem**: Users see "adapting from internal format" which exposes that there's an internal representation.

**Fix**: Use natural error messages:
```python
raise TypeError(f"{func.__name__}() {str(e)}")
```

### Issue: JIT Execution Errors
From test output:
```
[ERROR] ember.xcs.jit.execution_utils: Error executing JIT graph: internal_wrapper() got an unexpected keyword argument 'x'
```

**Problem**: "JIT graph" and "internal_wrapper" are implementation details.

**Fix**: Catch and re-raise with clean messages in natural.py.

## 2. Metadata Attributes Leak Implementation

### Issue: Visible Internal Attributes
```python
# These are exposed on functions:
natural_wrapper._is_jit_compiled = True
natural_wrapper._original_func = func
vmapped._is_vmapped = True
```

**Problem**: Users can see these implementation details.

**Fix**: Use a cleaner approach with a single metadata object:
```python
# Store in function's __dict__ under a private key
func.__dict__['__xcs_metadata__'] = XCSMetadata(
    transformations=['jit', 'vmap'],
    original=original_func
)
```

## 3. Dictionary Pattern Still Visible

### Issue: Trace Output Still Shows Dict Pattern
When using @trace, users might see internal dictionary representations in execution traces.

**Fix**: The trace decorator should also use adapters to show natural representations.

## 4. Operator Calling Convention Inconsistency

### Issue: Operators Still Require Dict I/O
```python
class MyOperator(Operator):
    def forward(self, *, inputs):  # Still forced pattern
        return {"result": inputs["x"]}
```

**Problem**: Even with natural calling support, operators still must be written with dict I/O.

**Fix**: Allow natural forward methods:
```python
class MyOperator(Operator):
    def forward(self, x, y):  # Natural!
        return x + y
```

## 5. VMap Return Type Inconsistency

### Issue: Dict Batches Return Different Structure
```python
# Input: list of dicts
result = vmap(func)([{...}, {...}])
# Output structure depends on function return type
```

**Problem**: Return structure changes based on input type.

**Fix**: Consistent transformation - always return same structure as single call would.

## 6. Composition Reveals Implementation

### Issue: Special Handling for Composition
```python
if hasattr(func, '_is_vmapped') and hasattr(func, '_original_func'):
    # Special case handling
```

**Problem**: We check for specific attributes to handle composition.

**Fix**: Use proper function wrapping protocol:
```python
# Use functools.wraps properly
# Check for __wrapped__ attribute (standard Python)
```

## 7. Type Information Loss

### Issue: Type Hints Not Preserved
The natural wrappers don't preserve type annotations properly.

**Fix**: Copy type hints to wrapper:
```python
natural_wrapper.__annotations__ = func.__annotations__
```

## 8. Performance Monitoring Leaks Internal Details

### Issue: get_jit_stats() Shows Internal Metrics
```python
stats = xcs.get_jit_stats()
# Shows: cache_hits, graph_nodes, etc.
```

**Problem**: Exposes graph-based implementation.

**Fix**: Show user-friendly metrics:
```python
{
    "calls": 100,
    "avg_speedup": 3.2,
    "compilation_time": 0.05,
}
```

## 9. Import Structure Reveals Implementation

### Issue: Deep Import Paths
```python
from ember.xcs.jit.natural import natural_jit
from ember.xcs.transforms.natural_vmap import natural_vmap
```

**Problem**: "natural" in path suggests there's an "unnatural" version.

**Fix**: Clean imports:
```python
from ember.xcs import jit, vmap  # Just works
```

## 10. Fallback Behavior Inconsistency

### Issue: Silent Fallbacks Hide Problems
When JIT compilation fails, it falls back to original function silently.

**Problem**: Users don't know if optimization is actually happening.

**Fix**: Provide optional strict mode:
```python
@xcs.jit(strict=True)  # Fail if can't optimize
def critical_function(x):
    return x
```

## Recommendations

1. **Error Handling Layer**: Add a translation layer that converts all internal errors to natural Python errors.

2. **Metadata Protocol**: Define a clean metadata protocol that doesn't expose internals.

3. **Consistent Signatures**: Ensure all transformations preserve exact function signatures.

4. **Natural Operators**: Extend Operator base class to support natural forward methods.

5. **Clean Public API**: Hide all implementation modules behind a clean facade.

6. **User-Friendly Metrics**: Transform internal metrics to user-meaningful values.

7. **Documentation**: Never mention internal representations in docs or errors.