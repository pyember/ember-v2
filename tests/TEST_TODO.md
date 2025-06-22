# Test Implementation TODO List

## Pre-Implementation Tasks

### Fix Import Issues
- [x] Update `tests/conftest.py` - migrate from `ember.core.registry.model` to `ember.api.models`
- [x] Fix `tests/integration/core/test_integration_core.py` - update operator imports (moved to .bak)
- [x] Update `tests/integration/core/registry/test_provider_discovery.py` - use new provider system
- [x] Fix `tests/golden/test_models_examples.py` - use `ember.api.models`
- [x] Remove all LMModule references from tests

### Clean Up Deprecated Tests
- [x] Remove LMModule-related tests
- [x] Remove old registry structure tests
- [x] Update or remove tests for deleted operator registry files

## Week 1: Core Correctness

### Model API Tests (`tests/unit/api/test_models_api.py`)
- [x] Basic invocation works
- [x] Deterministic with seed
- [x] No resource leaks (1000 calls)
- [x] Thread safety (100 concurrent calls)
- [x] Performance baseline (<50ms overhead)
- [x] Provider fallback chains
- [x] Cost calculation accuracy
- [x] Timeout handling
- [x] Golden example (documentation)
- [x] Streaming responses
- [x] Empty response handling
- [x] Unicode handling
- [x] Large prompt handling (10K chars)

### Data API Tests (`tests/unit/api/test_data_api.py`)
- [x] Basic loading from list/file/generator
- [x] Streaming truly streams (constant memory)
- [x] Transform pipeline integrity
- [x] Filter operations preserve data
- [x] Map operations correctness
- [x] Batch operations
- [x] Schema normalization
- [x] Format support (JSON, JSONL, CSV)
- [x] Large file handling (1GB+)
- [x] Generator efficiency
- [x] Resource cleanup
- [x] Error handling (corrupt data)
- [x] Golden examples

### Operators API Tests (`tests/unit/api/test_operators_api.py`)
- [x] Basic operator creation
- [x] Operator composition (chain, ensemble)
- [x] Type preservation through chains
- [x] Error propagation
- [x] Retry logic correctness
- [x] State management (stateful operators)
- [x] Caching behavior
- [x] Router operator
- [x] Golden examples
- [x] Thread safety

### XCS Tests (`tests/unit/api/test_xcs_api.py`)
- [x] Basic @jit compilation
- [x] Parallelization detection
- [x] Sequential fallback
- [x] Correctness with complex graphs
- [x] No race conditions
- [x] Deterministic optimization decision
- [x] Cache behavior
- [x] Recursive function handling
- [x] Error propagation
- [x] Performance overhead
- [x] Golden examples

### Resource Management Tests (`tests/unit/core/test_resources.py`)
- [x] No file descriptor leaks
- [x] No memory growth over 1K operations
- [x] Thread cleanup
- [x] Connection pooling
- [x] Context manager safety
- [x] Registry cleanup
- [x] Data generator cleanup
- [x] Circular reference cleanup

### Error Handling Tests (`tests/unit/core/test_errors.py`)
- [x] Clear error messages
- [x] Network failure handling
- [x] Invalid input handling
- [x] Timeout behavior
- [x] Partial failure recovery
- [x] Error context preservation
- [x] API key validation
- [x] Rate limit handling
- [x] Data format errors

## Week 2: Production Readiness

### Provider Integration Tests (`tests/integration/test_providers.py`)
- [ ] OpenAI real calls work
- [ ] Anthropic real calls work
- [ ] Google real calls work
- [ ] Provider-specific features
- [ ] Rate limit handling
- [ ] API key validation
- [ ] Network error recovery
- [ ] Response parsing
- [ ] Cost tracking across providers

### Fallback Chain Tests (`tests/integration/test_fallbacks.py`)
- [ ] Primary → Secondary works
- [ ] Multiple fallback levels
- [ ] Error propagation correct
- [ ] Cost tracking across fallbacks
- [ ] Logging/observability
- [ ] Partial success handling

### Concurrency Tests (`tests/integration/test_concurrency.py`)
- [x] Thread-safe model calls (partial in context tests)
- [x] Thread-safe registry access (exists)
- [ ] No deadlocks
- [ ] Async compatibility
- [ ] ThreadPool efficiency
- [ ] Race condition detection
- [ ] Concurrent data processing
- [ ] Parallel operator execution

### Performance Tests (`tests/benchmarks/test_performance.py`)
- [ ] Model overhead <50ms
- [ ] Data streaming >10K items/s
- [ ] Operator overhead minimal
- [ ] XCS compilation <200ms
- [ ] Memory usage flat over time
- [ ] Benchmark regression detection
- [ ] Latency percentiles (p50, p95, p99)

### Production Features Tests (`tests/integration/test_production.py`)
- [ ] Cost calculation correct
- [ ] Usage tracking accurate
- [ ] Retry logic with backoff
- [ ] Circuit breaker behavior
- [ ] Graceful degradation
- [ ] Monitoring hooks work

## Week 3: Engineering Excellence

### API Compatibility Tests (`tests/integration/test_compatibility.py`)
- [ ] Backwards compatibility
- [ ] Deprecation warnings
- [ ] Migration guide accuracy
- [ ] Version detection
- [ ] Feature flags work

### Golden Regression Tests (`tests/golden/test_regression.py`)
- [x] Some documentation examples (needs update)
- [ ] Common workflows
- [ ] Performance benchmarks
- [ ] API response formats
- [ ] Error message formats

### Memory Profiling Tests (`tests/integration/test_memory.py`)
- [ ] 1K operation stability
- [ ] Large object handling
- [ ] Garbage collection behavior
- [ ] Peak memory tracking
- [ ] Memory leak detection

### End-to-End Workflow Tests (`tests/e2e/test_workflows.py`)
- [ ] Build chatbot (5 min)
- [ ] Analyze dataset (10 min)
- [ ] Create eval pipeline (15 min)
- [ ] Build RAG system (20 min)
- [ ] Multi-model ensemble
- [ ] Stream processing pipeline

### Developer Experience Tests (`tests/e2e/test_dx.py`)
- [ ] Error messages helpful
- [ ] First success <2 min
- [ ] IDE autocomplete works
- [x] Type hints complete (partial - type tests exist)
- [ ] Documentation accuracy

## Test Infrastructure

### Test Utilities (`tests/test_utils.py`)
- [x] Response builders
- [x] Dataset builders
- [x] Resource monitoring
- [x] Performance baselines
- [x] Mock providers
- [x] Thread safety helpers
- [x] Golden file helpers
- [x] Deterministic helpers

### CI/CD Configuration
- [x] pytest.ini configuration (exists)
- [ ] Coverage configuration
- [ ] Parallel test execution
- [ ] Test categorization
- [ ] Benchmark tracking
- [ ] Flaky test detection

### Documentation
- [x] Test plan (TEST_PLAN.md)
- [x] Test README
- [ ] Contributing guide
- [ ] Debugging guide
- [ ] Performance tuning guide

## Priority Order

1. **Critical Fixes** (Do First):
   - Fix all import issues
   - Remove deprecated tests
   - Update conftest.py

2. **Core Tests** (Week 1):
   - Complete Model API tests with new structure
   - Add memory verification to Data tests
   - Migrate and complete Operator tests
   - Add missing XCS parallelization tests

3. **Production Tests** (Week 2):
   - Real provider integration
   - Fallback chains
   - Performance baselines
   - Concurrency verification

4. **Excellence Tests** (Week 3):
   - End-to-end workflows
   - Golden regression suite
   - Developer experience
   - Memory profiling