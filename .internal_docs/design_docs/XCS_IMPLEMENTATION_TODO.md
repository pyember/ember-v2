# XCS Implementation TODO List

## Overview
Transform XCS from disconnected components to working parallel execution system following our principled design.

## Phase 1: Core Infrastructure (Week 1)

### Day 1: Python Tracer
- [ ] Create `ember/xcs/_internal/tracer.py`
  - [ ] Implement `PythonTracer` class using `sys.settrace`
  - [ ] Handle all Python constructs (loops, comprehensions, generators)
  - [ ] Build operation recording mechanism
  - [ ] Add tests for tracer correctness
  
### Day 2: IR Builder Integration  
- [ ] Update `ember/xcs/_internal/ir_builder.py`
  - [ ] Remove AST analysis code
  - [ ] Integrate with PythonTracer
  - [ ] Build IRGraph from trace recordings
  - [ ] Handle nested function calls
  - [ ] Add comprehensive tests

### Day 3: Parallelism Analyzer
- [ ] Update `ember/xcs/_internal/parallelism.py`
  - [ ] Implement dependency analysis for real graphs
  - [ ] Detect I/O-bound vs CPU-bound operations
  - [ ] Find parallel groups accurately
  - [ ] Remove speedup threshold logic
  - [ ] Add tests for various patterns

### Day 4: Execution Engine
- [ ] Update `ember/xcs/_internal/engine.py`
  - [ ] Implement `ParallelExecutor` with sequential semantics
  - [ ] Add thread pool management (lazy init)
  - [ ] Implement deterministic ordering (sort nodes)
  - [ ] Handle error propagation correctly
  - [ ] Add execution tests

### Day 5: Connect Everything in @jit
- [ ] Update `ember/xcs/_simple.py`
  - [ ] Wire tracer → builder → analyzer → engine
  - [ ] Implement permanent optimization decisions
  - [ ] Add comprehensive cache key generation
  - [ ] Implement stats() method
  - [ ] Integration tests

## Phase 2: Robustness & Correctness (Week 2)

### Day 6-7: Error Handling
- [ ] Implement exact sequential error semantics
  - [ ] Create `UserFunctionError` wrapper
  - [ ] Cancel pending futures on error
  - [ ] Preserve original exceptions
  - [ ] Test error timing preservation
  
- [ ] Add fallback mechanisms
  - [ ] Permanent disable on tracing failure
  - [ ] No retry logic (fail fast)
  - [ ] Clean error messages
  - [ ] Test all error paths

### Day 8-9: Cache Key Correctness
- [ ] Implement comprehensive cache keys
  - [ ] Function identity and source hash
  - [ ] Argument type signatures
  - [ ] Handle nested data structures
  - [ ] Model identity extraction
  - [ ] Test cache key stability

### Day 10: Thread Safety
- [ ] Ensure thread-safe execution
  - [ ] Verify context immutability
  - [ ] Test concurrent @jit calls
  - [ ] Handle thread pool lifecycle
  - [ ] Add stress tests

## Phase 3: Pattern Support (Week 3)

### Day 11-12: Comprehension Patterns
- [ ] Support all comprehension types
  - [ ] List comprehensions: `[f(x) for x in items]`
  - [ ] Generator expressions: `(f(x) for x in items)`
  - [ ] Dict comprehensions: `{x: f(x) for x in items}`
  - [ ] Set comprehensions: `{f(x) for x in items}`
  - [ ] Nested comprehensions
  - [ ] Conditional comprehensions
  - [ ] Test each pattern

### Day 13: Explicit Loops
- [ ] Support explicit loop patterns
  - [ ] Simple for loops with append
  - [ ] Multiple operations per iteration
  - [ ] Break/continue handling
  - [ ] Nested loops
  - [ ] Test various patterns

### Day 14: Complex Patterns
- [ ] Handle advanced patterns
  - [ ] Chained operations: `f(g(x))`
  - [ ] Multiple independent operations
  - [ ] Mixed patterns in one function
  - [ ] Test edge cases

### Day 15: Performance Validation
- [ ] Benchmark real workloads
  - [ ] LLM call parallelization
  - [ ] CPU-bound parallelization
  - [ ] Overhead measurement
  - [ ] Create performance test suite

## Phase 4: Testing & Documentation (Week 4)

### Day 16-17: Comprehensive Test Suite
- [ ] Unit tests
  - [ ] Each component in isolation
  - [ ] Edge cases and error conditions
  - [ ] Thread safety tests
  - [ ] Performance regression tests

- [ ] Integration tests
  - [ ] Full pipeline tests
  - [ ] Real Ember model integration
  - [ ] Error propagation tests
  - [ ] Concurrent execution tests

### Day 18: Real-World Testing
- [ ] Test with actual Ember code
  - [ ] Find real parallelization opportunities
  - [ ] Measure actual speedups
  - [ ] Identify any issues
  - [ ] Create example notebooks

### Day 19: Documentation
- [ ] User documentation
  - [ ] Clear explanation of @jit
  - [ ] What can/cannot be parallelized
  - [ ] Performance expectations
  - [ ] Troubleshooting guide

- [ ] Developer documentation
  - [ ] Architecture overview
  - [ ] Component interactions
  - [ ] Extension points
  - [ ] Design decisions

### Day 20: Release Preparation
- [ ] Code cleanup
  - [ ] Remove dead code
  - [ ] Ensure Google Python Style
  - [ ] Add type hints everywhere
  - [ ] Final code review

- [ ] Release checklist
  - [ ] All tests passing
  - [ ] Documentation complete
  - [ ] Performance benchmarks
  - [ ] Migration guide (if needed)

## Implementation Files Checklist

### Core Files to Create/Update:
- [ ] `ember/xcs/_simple.py` - Main @jit implementation
- [ ] `ember/xcs/_internal/tracer.py` - Python execution tracer
- [ ] `ember/xcs/_internal/ir_builder.py` - IR graph construction
- [ ] `ember/xcs/_internal/parallelism.py` - Parallelism analysis
- [ ] `ember/xcs/_internal/engine.py` - Execution engine
- [ ] `ember/xcs/_internal/errors.py` - Error handling

### Test Files to Create:
- [ ] `tests/unit/xcs/test_tracer.py`
- [ ] `tests/unit/xcs/test_ir_builder.py`
- [ ] `tests/unit/xcs/test_parallelism.py`
- [ ] `tests/unit/xcs/test_engine.py`
- [ ] `tests/integration/xcs/test_jit_simple.py`
- [ ] `tests/integration/xcs/test_jit_patterns.py`
- [ ] `tests/integration/xcs/test_jit_errors.py`
- [ ] `tests/integration/xcs/test_jit_performance.py`

### Files to Remove:
- [ ] Complex strategy files (ir_based.py, pytree_aware.py, etc.)
- [ ] Overengineered components
- [ ] Design docs for rejected approaches

## Definition of Success

### Functional Requirements:
- [x] `@jit` makes parallel code run in parallel
- [x] Exact sequential semantics preserved
- [x] No configuration needed
- [x] Fails fast with clear errors

### Performance Requirements:
- [x] <1ms overhead for non-parallel code
- [x] >2x speedup for parallel LLM calls
- [x] Thread pool overhead <5% of execution time
- [x] Cache lookup <1μs

### Quality Requirements:
- [x] 100% test coverage for core paths
- [x] No regression in existing Ember code
- [x] Clear, maintainable code
- [x] Comprehensive documentation

## Risk Mitigation

### Technical Risks:
1. **Tracer Performance**: Profile early, optimize if needed
2. **Thread Pool Overhead**: Lazy initialization, careful sizing
3. **Cache Memory**: Bound cache size, add eviction if needed
4. **Complex Patterns**: Start simple, add patterns incrementally

### Schedule Risks:
1. **Underestimated Complexity**: Core functionality first, patterns later
2. **Testing Time**: Automated test generation where possible
3. **Integration Issues**: Test with real Ember code early

## Success Metrics

### Week 1: Core Working
- Trace simple functions
- Detect parallel opportunities
- Execute in parallel
- Basic tests passing

### Week 2: Robust
- Error handling correct
- Thread-safe execution
- Cache working properly
- More patterns supported

### Week 3: Complete
- All patterns supported
- Performance validated
- Real-world tested
- Documentation started

### Week 4: Shippable
- All tests passing
- Documentation complete
- Performance benchmarked
- Ready for users

## Daily Standup Questions

1. What did I complete yesterday?
2. What will I complete today?
3. Are there any blockers?
4. Is the design still correct?

## Notes

- **No Scope Creep**: Resist adding features not in design
- **Simple First**: Get basic case working before complex
- **Test Everything**: Each component fully tested
- **Measure Always**: Profile and benchmark continuously
- **Stay Principled**: Follow design decisions document

This plan implements exactly what we designed - no more, no less.