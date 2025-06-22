# XCS Fresh Test Suite Plan

## Current Status
We have 4 working XCS test files covering core functionality:
1. `test_xcs_return_value_fix.py` - Ensures proper return value handling (6 tests)
2. `test_xcs_parallelism_fix.py` - Validates parallelism detection (3 tests)
3. `test_xcs_transformations.py` - Tests vmap, pmap, scan, grad (16 tests)
4. `test_xcs_deep_nesting_stress.py` - Stress tests for nested structures (5 tests)

## Coverage Gaps from Deprecated Tests

### 1. Advanced Vectorization
- Nested vmap scenarios
- Ragged sequences
- Dynamic shapes
- vmap/pmap composition

### 2. Edge Cases & Robustness
- Memory management
- Thread safety
- Error recovery
- Resource limits

### 3. Performance & Diagnostics
- Parallelism metrics
- Compilation overhead
- Execution timing
- Memory usage

### 4. Integration Patterns
- Mixed JAX/orchestration workflows
- Complex operator compositions
- Real-world usage patterns

## Proposed New Test Suite

### 1. `test_xcs_core.py` - Core XCS Functionality
```python
class TestXCSCore:
    def test_jit_basic_usage()
    def test_jit_with_operators()
    def test_jit_caching_behavior()
    def test_jit_error_messages()
    def test_stats_collection()
```

### 2. `test_xcs_vectorization.py` - Advanced Batching
```python
class TestXCSVectorization:
    def test_vmap_nested_structures()
    def test_vmap_with_modules()
    def test_vmap_variable_length_data()
    def test_vmap_pmap_composition()
    def test_vmap_performance_benefits()
```

### 3. `test_xcs_hybrid_workflows.py` - JAX + Orchestration
```python
class TestXCSHybridWorkflows:
    def test_tensor_orchestration_mix()
    def test_gradient_barriers()
    def test_parallel_llm_calls()
    def test_conditional_routing()
    def test_real_world_pipeline()
```

### 4. `test_xcs_robustness.py` - Edge Cases & Reliability
```python
class TestXCSRobustness:
    def test_large_graph_handling()
    def test_memory_efficiency()
    def test_error_recovery()
    def test_concurrent_usage()
    def test_recursive_patterns()
```

### 5. `test_xcs_performance.py` - Performance Validation
```python
class TestXCSPerformance:
    def test_parallelism_speedup()
    def test_compilation_overhead()
    def test_memory_usage_patterns()
    def test_scaling_characteristics()
```

## Implementation Guidelines

1. **Follow CLAUDE.md Principles**:
   - Clear, explicit test names
   - No magic - test actual behavior
   - Comprehensive coverage of common cases
   - Performance measurements with real timing

2. **Use Current APIs**:
   - Work with frozen/immutable modules
   - Use proper type annotations
   - Follow equinox patterns

3. **Focus on User Value**:
   - Test what users actually do
   - Provide clear examples
   - Validate performance benefits
   - Ensure good error messages

4. **Progressive Complexity**:
   - Start with simple cases
   - Build to complex scenarios
   - Show composition patterns
   - Demonstrate best practices

## Priority Order

1. **High Priority** (Week 1):
   - Core XCS functionality tests
   - Hybrid workflow tests (most unique value)

2. **Medium Priority** (Week 2):
   - Advanced vectorization tests
   - Performance validation tests

3. **Lower Priority** (Week 3):
   - Robustness/edge case tests
   - Comprehensive integration tests

## Success Criteria

- All tests pass reliably
- Clear documentation of expected behavior
- Performance benefits are measurable
- Error messages are helpful
- Tests serve as usage examples