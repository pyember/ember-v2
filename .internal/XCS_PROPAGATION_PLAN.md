# XCS Simplification Propagation Plan

## What Needs to Be Propagated

### 1. Graph Implementation Updates

**Replace Graph with Graph:**
- `src/ember/api/xcs.py` - Update import
- `src/ember/xcs/graph/__init__.py` - Remove Graph export
- Update all test files using Graph
- Update all examples using Graph

**Current state:**
```python
# Old (remove)
from ember.xcs.graph import Graph

# New (use everywhere)
from ember.xcs.graph.graph import Graph
```

### 2. JIT Simplification

**Remove structural strategy:**
- Remove from strategy imports in `jit/core.py`
- Remove structural strategy tests
- Update strategy selection to use simplified version
- Remove enhanced strategy (not needed)

**Simplify to:**
```python
# Only structural JIT provides real speedup
@jit  # Only works on Operators with parallelizable patterns
```

### 3. Clean Up Exports

**src/ember/xcs/__init__.py:**
- Remove Graph from exports
- Remove complex strategy selection functions
- Keep only: `jit`, `trace`, `Graph`

**src/ember/xcs/graph/__init__.py:**
- Remove Graph export
- Export only Graph and Node

### 4. Update Tests

**Remove/Update:**
- `tests/unit/xcs/engine/test_unified_engine.py` (unified engine deleted)
- `tests/unit/xcs/jit/test_trace_strategy.py` (trace not for optimization)
- Update all tests importing Graph

**Add:**
- Tests for new simplified Graph
- Tests for @trace decorator
- Tests for simplified JIT

### 5. Update Documentation

**Remove references to:**
- structural JIT
- Complex strategy selection
- Graph

**Add documentation for:**
- New Graph API
- @trace for debugging
- Simplified JIT model

## Implementation Order

### Phase 1: Core Updates (High Priority)
1. Update `src/ember/xcs/__init__.py` exports
2. Update `src/ember/xcs/graph/__init__.py` exports
3. Update `src/ember/api/xcs.py` imports

### Phase 2: JIT Simplification
1. Replace `jit/core.py` with simplified version
2. Remove unnecessary strategy files
3. Update strategy selection

### Phase 3: Test Updates
1. Update tests to use new Graph
2. Remove tests for removed features
3. Add tests for new features

### Phase 4: Documentation
1. Update docstrings
2. Update examples
3. Create migration guide

## Benefits After Propagation

1. **Cleaner API**: Only exports what actually works
2. **Less Code**: Remove thousands of lines
3. **Clearer Mental Model**: JIT optimizes, trace analyzes
4. **Better Performance**: Focus on what actually provides speedup

## Risks and Mitigation

**Risk**: Breaking existing code
**Mitigation**: Keep compatibility shim temporarily

**Risk**: Missing important functionality
**Mitigation**: Careful review of what's being removed

**Risk**: Test coverage gaps
**Mitigation**: Ensure new implementations are well-tested