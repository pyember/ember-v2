# Ember Test Plan

Following the principles of Jeff Dean, Sanjay Ghemawat, Steve Jobs, Robert C Martin, Greg Brockman, Ritchie, Knuth, Larry Page, and John Carmack.

## Core Principles
1. **No magic** - Tests should be explicit and obvious
2. **Fast** - All unit tests should run in under 10 seconds
3. **Deterministic** - Same result every time with fixed seeds
4. **Clear failures** - Error messages tell you exactly what's wrong
5. **Minimal** - Test the contract, not the implementation

## Test Structure

### Unit Tests

#### 1. Models API (`tests/unit/models/`)
- `test_models_function.py` - Test the models() function
  - Basic invocation: `models("gpt-4", "Hello")`
  - With parameters: `models("gpt-4", "Hello", temperature=0.7)`
  - Error cases: missing API key, invalid model
  - Response object: `.text`, `.usage`, `.model_id`
- `test_model_binding.py` - Test model.instance()
  - Creating bindings with preset parameters
  - Overriding parameters on call
  - Validation on creation
- `test_model_registry.py` - Test the registry internals
  - Thread safety
  - Caching behavior
  - Provider resolution

#### 2. Operators (`tests/unit/operators/`)
- `test_operator_base.py` - Test Operator base class
  - Functions are operators
  - Classes with __call__ are operators
  - JAX integration (pytree behavior)
- `test_common_operators.py` - Test built-in operators
  - Ensemble: voting, averaging
  - Chain: sequential composition
  - Router: conditional execution
  - Retry: error handling
  - Cache: memoization
- `test_learnable_router.py` - Test ML-based routing
  - Training the router
  - Inference behavior
  - JAX transformations

#### 3. XCS (`tests/unit/xcs/`)
- `test_jit.py` - Test @jit decorator
  - Basic function compilation
  - Automatic parallelization detection
  - Error propagation
  - Stats collection
- `test_vmap.py` - Test @vmap transformation
  - Automatic batching
  - Nested vmap
  - Integration with jit
- `test_pmap.py` - Test @pmap for multi-device
  - Device sharding
  - Collective operations
  - Error handling

#### 4. Data API (`tests/unit/data/`)
- `test_stream.py` - Test streaming interface
  - Basic iteration: `for item in stream("dataset")`
  - Filtering and transformation
  - Memory efficiency
- `test_load.py` - Test eager loading
  - Loading full datasets
  - Subset selection
  - Format conversions
- `test_sources.py` - Test data sources
  - HuggingFace integration
  - File loading (JSON, CSV, Parquet)
  - Custom sources

#### 5. Core (`tests/unit/core/`)
- `test_module.py` - Test Ember Module (equinox)
  - Attribute access
  - Immutability
  - JAX pytree registration
- `test_exceptions.py` - Test error handling
  - Clear error messages
  - Error context
  - Graceful degradation

### Integration Tests (`tests/integration/`)
- `test_end_to_end.py` - Complete workflows
  - Load data → Process with model → Ensemble results
  - Error handling across components
  - Resource management
- `test_jax_integration.py` - JAX ecosystem
  - Using Ember with Flax/Haiku
  - Gradient computation
  - JIT compilation of full pipelines

### Benchmarks (`tests/benchmarks/`)
- `benchmark_models.py` - Model performance
  - Latency per call
  - Throughput with batching
  - Memory usage
- `benchmark_operators.py` - Operator overhead
  - Composition cost
  - Caching effectiveness
  - Parallelization speedup
- `benchmark_xcs.py` - XCS optimizations
  - JIT compilation time
  - Runtime speedup
  - Memory efficiency

## Test Utilities

### Fixtures (`tests/conftest.py`)
```python
@pytest.fixture
def mock_model_response():
    """Standard mock response for model tests."""
    return ChatResponse(
        data="Test response",
        usage=UsageStats(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    )

@pytest.fixture
def temp_data_file(tmp_path):
    """Create temporary data file for testing."""
    data = [{"text": "Hello"}, {"text": "World"}]
    file_path = tmp_path / "test_data.json"
    file_path.write_text(json.dumps(data))
    return file_path
```

### Markers
- `@pytest.mark.slow` - Tests that take >1 second
- `@pytest.mark.requires_gpu` - Tests needing GPU
- `@pytest.mark.integration` - Integration tests

## Coverage Goals
- Unit tests: 90% coverage of public API
- Integration tests: Cover all major use cases
- No testing of:
  - Private methods (implementation details)
  - Generated code (trust equinox/JAX)
  - External libraries

## Running Tests
```bash
# All tests
pytest

# Fast tests only
pytest -m "not slow"

# With coverage
pytest --cov=ember --cov-report=html

# Specific module
pytest tests/unit/models/
```