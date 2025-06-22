# Test Update Plan for Simplified XCS API

## Overview
After merging xcs-radical-simplification, many tests need updating to align with the minimal 4-function XCS API.

## XCS API Changes
- **Kept**: `jit`, `trace`, `get_jit_stats`, `vmap`
- **Removed**: `pmap`, `Graph`, `Node`, `ExecutionOptions`, `execution_options`, `explain_jit_selection`, `JITMode`

## Test Files to Update

### 1. Unit Tests (`tests/unit/xcs/`)
- **test_simple_graph.py**: Remove - tests internal Graph implementation
- **test_core_graph.py**: Remove - tests internal Graph implementation  
- **Keep backup files**: Already suffixed with .backup

### 2. Integration Tests (`tests/integration/xcs/`)
- **test_graph_integration.py**: Remove - tests internal Graph
- **Keep backup files**: Already suffixed with .backup

### 3. JIT Tests to Update
- **tests/integration/tracer/test_enhanced_jit.py**:
  - Remove `execution_options` import and usage
  - Remove any `JITMode` references
  - Test only with @jit decorator (no mode selection)

### 4. Tests to Create
- **tests/unit/xcs/test_public_api.py**: Test the 4 public functions
- **tests/integration/xcs/test_jit_operators.py**: Test @jit with operators
- **tests/unit/xcs/test_vmap.py**: Test vmap transformation

## Example Test Updates

### Before (with removed APIs):
```python
from ember.xcs import jit, JITMode, execution_options, Graph

@jit(mode=JITMode.ENHANCED)
class MyOperator(Operator):
    pass

with execution_options(scheduler="parallel"):
    result = operator(inputs)

graph = Graph()
```

### After (simplified API):
```python
from ember.api import xcs

@xcs.jit  # No mode parameter
class MyOperator(Operator):
    pass

# No execution_options - it just works
result = operator(inputs)

# No Graph access - internal implementation
```

## Action Items

1. Remove graph-specific test files
2. Update JIT tests to remove mode selection
3. Remove execution_options usage
4. Create new tests for the minimal public API
5. Ensure all imports use `ember.api.xcs` not `ember.xcs`