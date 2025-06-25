# Test Infrastructure Design

Following Google Python Style Guide and CLAUDE.md principles for world-class testing.

## Overview

Comprehensive test coverage for the new context system and CLI, designed with:
- Principled, root-node fixes
- Explicit behavior over magic
- Performance measurement
- Security-first approach
- 10x improvements in test quality

## Test Structure

### 1. Context System Tests (`test_ember_context.py`)

**Core Functionality**
- Singleton pattern verification with performance measurement
- Configuration get/set with dot notation
- Edge case handling (null, empty strings, invalid types)
- Performance: <1ms for 10k singleton accesses

**Thread Safety**
- Thread-local context isolation
- Concurrent mutation safety
- Barrier synchronization for deterministic testing
- Stress testing with 100 iterations × 10 threads

**Async Context Propagation**
- Context persistence across await boundaries
- Task isolation with asyncio.gather
- ContextVar propagation verification

**Inheritance & Isolation**
- Parent-child context relationships
- Deep copy verification for nested structures
- Registry isolation between contexts

**Persistence**
- Save/load cycle verification
- Corrupted file handling
- Atomic write operations

**Performance Benchmarks**
- Config lookup: <10ms for 1k nested lookups
- Child creation: <100ms for 100 children
- Measured with time.perf_counter()

### 2. CLI Command Tests (`test_cli_commands.py`)

**Design Principles**
- Minimal mocking - test real behavior
- Direct assertions without verbose setup
- Exit code verification for all paths

**Coverage**
- Main entry point (help, errors, interrupts)
- Configure command (get/set/list with JSON/YAML)
- Setup wizard (npm detection, environment passing)
- Models listing (providers and model discovery)
- Connection testing (success/failure paths)
- Version display

**Key Features**
- Mock isolation for external dependencies
- Output capture with capsys
- Proper exit codes (0, 1, 130 for SIGINT)

### 3. Security Tests (`test_security.py`)

**Credential Security**
- File permissions (0o600 - owner only)
- No credential logging
- Sanitized error messages
- Atomic writes to prevent corruption

**Input Validation**
- Path traversal prevention
- Command injection blocking
- Prototype pollution protection
- Length limits and character validation

**API Security**
- stdin reading for sensitive data
- No shell execution from user input
- Environment variable sanitization
- Memory clearing for sensitive data

**CLI Security**
- Command injection prevention
- Help text injection blocking
- Secure directory creation (0o700)

### 4. Integration Tests (`test_cli_context_integration.py`)

**Full Stack Testing**
- CLI → Context → Registry flow
- Configuration persistence across invocations
- Credential flow from save to use
- Migration integration
- Concurrent CLI access safety

**End-to-End Workflows**
- First-time user setup
- Model discovery and configuration
- Error recovery scenarios
- Missing credential handling

**Setup Wizard Integration**
- Mock npx execution
- Context updates from wizard
- Credential and config persistence

## Test Quality Metrics

### Coverage Goals
- Line coverage: >95%
- Branch coverage: >90%
- Security paths: 100%

### Performance Targets
- Test suite execution: <5 seconds
- Individual test: <100ms (except stress tests)
- No test should sleep >10ms

### Maintainability
- Single responsibility per test
- Clear test names describing behavior
- Minimal test interdependence
- Fixtures for common setups only

## Key Design Decisions

1. **Isolated Testing**: Each test uses `EmberContext(isolated=True)` to prevent interference

2. **Real File I/O**: Tests use actual tempfile operations rather than mocking filesystem

3. **Security First**: Dedicated security test suite with injection attempts and permission checks

4. **Performance Awareness**: Explicit performance tests with measured targets

5. **Integration Focus**: Separate integration tests verify component interactions

## Running Tests

```bash
# All tests
pytest tests/

# Specific suites
pytest tests/unit/core/test_ember_context.py -v
pytest tests/unit/cli/test_cli_commands.py -v
pytest tests/unit/core/test_security.py -v
pytest tests/integration/test_cli_context_integration.py -v

# With coverage
pytest tests/ --cov=ember --cov-report=html

# Performance tests only
pytest tests/ -k "performance" -v
```

## Future Enhancements

1. **Property-based testing** with Hypothesis for edge cases
2. **Mutation testing** to verify test effectiveness
3. **Load testing** for concurrent access patterns
4. **Fuzz testing** for security vulnerabilities
5. **Contract testing** for API compatibility

This test infrastructure ensures the context and CLI systems are:
- Robust against edge cases
- Secure by design
- Performant at scale
- Maintainable long-term