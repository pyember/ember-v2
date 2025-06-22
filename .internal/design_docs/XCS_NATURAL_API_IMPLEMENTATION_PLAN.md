# XCS Natural API Implementation Plan

## Overview

This plan details the step-by-step implementation of the natural API for XCS, eliminating the forced dictionary I/O pattern while maintaining backward compatibility.

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Create Function Introspection Module
```python
# src/ember/xcs/introspection.py
class FunctionIntrospector:
    """Analyzes function signatures and calling patterns."""
    
    def analyze(self, func: Callable) -> FunctionMetadata:
        """Deep analysis of function signature and patterns."""
        
    def detect_style(self, func: Callable) -> CallStyle:
        """Detect if function uses natural, operator, or mixed style."""
        
    def extract_type_hints(self, func: Callable) -> TypeInfo:
        """Extract and preserve type information."""
```

#### 1.2 Build Adapter System
```python
# src/ember/xcs/adapters.py
class UniversalAdapter:
    """Adapts any Python callable to/from internal representation."""
    
    def adapt_inputs(self, args, kwargs) -> Dict[str, Any]:
        """Convert natural inputs to internal format."""
        
    def adapt_outputs(self, result: Dict[str, Any]) -> Any:
        """Convert internal outputs to natural format."""
        
    def create_wrapper(self, internal_func: Callable) -> Callable:
        """Create wrapper preserving original signature."""
```

### Phase 2: Transform JIT (Week 2)

#### 2.1 Refactor JIT to Use Adapters
```python
# src/ember/xcs/jit/natural.py
def jit(func: F) -> F:
    """JIT with natural function support."""
    # Detect function style
    introspector = FunctionIntrospector()
    metadata = introspector.analyze(func)
    
    if metadata.is_natural:
        # Natural path - adapt seamlessly
        adapter = UniversalAdapter(metadata)
        internal = adapter.to_internal(func)
        compiled = compile_graph(internal)
        return adapter.from_internal(compiled)
    else:
        # Legacy path - maintain compatibility
        return legacy_jit(func)
```

#### 2.2 Update Compilation Pipeline
- Modify graph builder to handle adapted functions
- Ensure type information flows through compilation
- Add natural function tests to JIT test suite

### Phase 3: Natural VMap (Week 3)

#### 3.1 Smart Batch Detection
```python
# src/ember/xcs/transforms/natural_vmap.py
class SmartBatchDetector:
    """Intelligently detects batch patterns in inputs."""
    
    def detect(self, args, kwargs, signature) -> BatchPattern:
        """Detect how inputs are batched."""
        
    def unbatch(self, args, kwargs, pattern) -> Iterator:
        """Yield individual items from batch."""
        
    def rebatch(self, results, pattern) -> Any:
        """Combine results back into batch structure."""
```

#### 3.2 Implement Natural VMap
- Support all common batching patterns
- Maintain performance with parallel execution
- Full compatibility with existing vmap

### Phase 4: Integration & Migration (Week 4)

#### 4.1 Update Operator Base Class
```python
class Operator:
    """Enhanced operator supporting both styles."""
    
    def __call__(self, *args, **kwargs):
        if args or not all(k == "inputs" for k in kwargs):
            # Natural style call
            return self._natural_call(*args, **kwargs)
        else:
            # Legacy style call
            return self._legacy_call(**kwargs)
```

#### 4.2 Create Migration Tools
- Automated script to update examples
- Compatibility warnings for deprecated patterns
- Documentation generator for new API

## Testing Strategy

### Unit Tests
```python
def test_natural_jit_preserves_signature():
    @jit
    def add(x: int, y: int) -> int:
        return x + y
    
    # Verify signature preserved
    assert inspect.signature(add) == inspect.signature(add.__wrapped__)
    
    # Verify behavior preserved
    assert add(2, 3) == 5
    
    # Verify types preserved
    assert add.__annotations__ == {'x': int, 'y': int, 'return': int}

def test_vmap_adapts_to_inputs():
    @vmap
    def process(x, y=1):
        return x * y
    
    # Single argument batching
    assert process([1, 2, 3]) == [1, 2, 3]
    
    # Multiple argument batching
    assert process([1, 2], [3, 4]) == [3, 8]
    
    # Keyword argument batching
    assert process(x=[1, 2], y=[3, 4]) == [3, 8]
    
    # Mixed batching
    assert process([1, 2], y=5) == [5, 10]
```

### Integration Tests
- Test all combinations of transformations
- Verify performance characteristics
- Ensure backward compatibility

### Performance Benchmarks
```python
def benchmark_natural_vs_legacy():
    # Natural style
    @jit
    def natural_add(x, y):
        return x + y
    
    # Legacy style
    @jit
    def legacy_add(*, inputs):
        return {"result": inputs["x"] + inputs["y"]}
    
    # Natural should be faster (no dict overhead)
    natural_time = timeit(lambda: natural_add(1, 2))
    legacy_time = timeit(lambda: legacy_add(inputs={"x": 1, "y": 2}))
    assert natural_time < legacy_time
```

## Migration Guide

### For Users
```python
# Old way (still works but discouraged)
@xcs.jit
def compute(*, inputs):
    return {"result": inputs["x"] + inputs["y"]}

result = compute(inputs={"x": 1, "y": 2})["result"]

# New way (natural and simple)
@xcs.jit
def compute(x, y):
    return x + y

result = compute(1, 2)
```

### For Library Developers
1. Update examples to use natural style
2. Add deprecation warnings to dict-only patterns
3. Provide migration tool for large codebases

## Success Criteria

1. **Zero Breaking Changes**: All existing code continues to work
2. **Natural Feel**: New users never see dictionary I/O
3. **Performance Win**: Natural style is measurably faster
4. **Type Safety**: Full type preservation through transformations
5. **Clean Errors**: Mistakes produce helpful Python errors

## Timeline

- Week 1: Core infrastructure
- Week 2: JIT transformation  
- Week 3: VMap and other transforms
- Week 4: Integration and migration tools
- Week 5: Documentation and examples
- Week 6: Performance optimization and release

## Conclusion

This implementation plan provides a clear path to making XCS feel like a natural extension of Python rather than a foreign framework. By meeting users where they are instead of forcing them to adopt our conventions, we achieve the simplicity and elegance that defines great software design.