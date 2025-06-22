# XCS Cleanup Plan

## Phase 1: Delete Old Code (Immediate)

### Files to Delete
```bash
# Old schedulers
src/ember/xcs/schedulers/unified_scheduler.py
src/ember/xcs/schedulers/base_scheduler_impl.py
src/ember/xcs/engine/unified_engine.py  # Already deleted

# Old JIT strategies
src/ember/xcs/jit/strategies/trace.py
src/ember/xcs/jit/strategies/structural.py
src/ember/xcs/jit/strategies/enhanced.py
src/ember/xcs/jit/strategies/base_strategy.py

# Complex options
src/ember/xcs/engine/execution_options.py

# Dispatcher
src/ember/xcs/utils/executor.py

# Old graph
src/ember/xcs/graph/xcs_graph.py  # Keep for now, update later

# Custom exceptions
src/ember/xcs/exceptions.py  # Reduce to 1-2 exceptions
```

### Code to Simplify
- `xcs_engine.py` - Remove legacy functions, keep only new `execute_graph`
- `factory.py` - Simplify to just create sequential/parallel
- `__init__.py` files - Update exports

## Phase 2: Update Integration Points

### API Module (`src/ember/xcs/api/core.py`)
- Remove ExecutionOptions usage
- Use new simple execute_graph
- Update to use new Graph class

### Transform Module Updates
- Update existing vmap.py to use new implementation
- Update pmap.py to use new implementation
- Delete complex transformation base classes

### JIT Module Updates  
- Replace complex jit/core.py with simple version
- Delete strategy selector
- Update execution_utils.py

## Phase 3: Test Updates

### Update Test Imports
```python
# Old
from ember.xcs.engine import ExecutionOptions, execute_graph
from ember.xcs.graph.xcs_graph import Graph

# New
from ember.xcs.simple import Graph, run
```

### Simplify Test Cases
- Remove ExecutionOptions tests
- Remove scheduler selection tests
- Remove JIT strategy tests
- Add new pattern detection tests

## Phase 4: Documentation

### Update READMEs
- Main XCS README
- Update code examples
- Remove references to deleted features

### Migration Guide
```python
# automated_migration.py
def migrate_code(old_code: str) -> str:
    """Convert old XCS code to new API."""
    # Replace imports
    new_code = old_code.replace(
        "from ember.xcs.graph.xcs_graph import Graph",
        "from ember.xcs.simple import Graph"
    )
    # Replace ExecutionOptions
    new_code = re.sub(
        r'ExecutionOptions\([^)]+\)',
        '{}',  # Remove options
        new_code
    )
    # More replacements...
    return new_code
```

## Phase 5: Performance Validation

### Benchmarks to Run
1. Basic graph execution (sequential vs parallel)
2. JIT compilation overhead
3. Pattern detection accuracy
4. Memory usage comparison

### Expected Results
- Less memory usage (fewer objects)
- Faster startup (no option validation)
- Same or better execution speed
- Simpler profiling output

## Estimated Impact

### Lines of Code to Delete
- Schedulers: ~1,500 lines
- JIT strategies: ~2,000 lines  
- ExecutionOptions: ~350 lines
- Dispatcher: ~500 lines
- Custom exceptions: ~250 lines
- **Total: ~4,600 lines deleted**

### Final State
- XCS core: ~1,000 lines (from ~6,000)
- Much easier to understand
- More powerful through simplicity
- Better performance

## Order of Operations

1. **First**: Create backup branch
2. **Delete**: Remove old files
3. **Update**: Fix imports and integration points
4. **Test**: Ensure everything works
5. **Document**: Update all docs
6. **Benchmark**: Validate performance