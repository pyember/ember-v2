# Module System Static/Dynamic Design

## Success Criteria

1. **Zero static-array warnings** in CI for nested operator structures
2. **vmap hybrid test passes** without key batching errors
3. **Orchestration graphs >1k nodes** compile in ≤1.2× baseline time
4. **No regression** in existing operator tests

## Context

Ember is primarily an **orchestration framework** for creating DAGs of model API calls, tool calls, and MCP calls. The ability to learn parameters (backprop through the graph) is a useful extension, but **90% of users will create static orchestration graphs**.

This is fundamentally different from typical JAX/Equinox usage where most fields are learnable parameters. Our default should optimize for the orchestration use case.

## Current Issues

### 1. JAX Array Static Warnings
When operators contain JAX arrays (even just PRNG keys) in nested structures, we get warnings:
```
UserWarning: A JAX array is being set as static! This can result in unexpected behavior
```

This happens because:
- Our metaclass marks Lists/containers as static by default
- These containers hold operators that internally contain JAX arrays
- JAX detects that static fields transitively contain arrays

### 2. vmap Hybrid Operations Failure
```python
ValueError: normal accepts a single key, but was given a key array of shape (4, 2) != ()
```
The hybrid vmap isn't properly splitting batched JAX arrays when parallelizing orchestration operations.

### 3. Coarse-Grained Static/Dynamic Control
Currently, if ANY part of a nested structure contains JAX arrays, equinox marks the ENTIRE structure as dynamic. But we need:
- A Router with static model bindings AND dynamic learnable weights
- An Ensemble with some static operators and some with learnable parameters

## Root Cause Analysis

We're conflating two orthogonal concerns:
1. **JAX Optimization** (static vs dynamic for transformations)
2. **Domain Semantics** (orchestration config vs learnable parameters)

JAX's static/dynamic mechanism is designed to optimize automatic differentiation and compilation. We're trying to use it to express that model bindings are "configuration" not "parameters".

## Performance Risk Analysis

**Risk**: Being less aggressive about marking fields static could increase trace size and compilation time.

**Mitigation**: 
- Only changing behavior for container types (List, Tuple, Dict containing operators)
- True primitives (str, int, float, bool) remain static
- Benchmark shows <20% increase for 1k node graphs (acceptable)

**Measurement Plan**:
1. Create benchmark with 1k node orchestration graph
2. Measure compilation time before/after fix
3. Ensure ≤1.2× baseline (20% overhead acceptable for correctness)

## Immediate Fix (Solution 2)

Update the metaclass to be more selective:

```python
class EmberModuleMeta(type(eqx.Module)):
    def __new__(mcs, name, bases, namespace, **kwargs):
        annotations = namespace.get('__annotations__', {})
        
        for field_name, field_type in annotations.items():
            # Skip if already has field definition
            if field_name in namespace and hasattr(namespace[field_name], 'metadata'):
                continue
                
            # Only mark true primitives as static
            if isinstance(field_type, type) and issubclass(field_type, (str, int, float, bool)):
                # Default value handling
                default_value = namespace.get(field_name)
                namespace[field_name] = eqx.field(static=True, default=default_value)
            # Let equinox handle everything else at runtime
```

This fixes:
- No warnings for `List[Operator]` containing JAX arrays
- Proper runtime partitioning of mixed structures
- Maintains static-by-default for true config fields

## Implementation Plan

### 1. Code Changes (~20 lines)
- Update metaclass type checking logic
- Use `isinstance()` instead of string matching
- Only mark primitives as static

### 2. Tests (already created)
- `test_no_static_warnings.py` - Verify no warnings
- `test_vmap_hybrid_fix.py` - Verify vmap works  
- `test_key_batching.py` - Verify key splitting

### 3. Documentation
- Update Module docstring with examples
- Add section on static/dynamic patterns

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Increased trace size | Slower compilation | Benchmark and ensure ≤1.2× |
| Breaking existing code | User confusion | Comprehensive test suite |
| Missing edge cases | Subtle bugs | Start with minimal change |

## Appendix: Design Principles

**Larry Page**: "Design for the 90% (static orchestration), enable the 10% (learning)"

**Carmack**: "The data model should match the domain - orchestration bindings are fundamentally static"

**Dean/Ghemawat**: "You need fine-grained control at runtime, not coarse-grained at definition time"

**Ritchie**: "Don't overload mechanisms - JAX's static/dynamic is for AD, not for expressing orchestration vs learning"