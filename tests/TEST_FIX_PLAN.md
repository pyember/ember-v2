# Test Fix Master Plan

## Executive Summary
Current state: 25/38 tests passing (66%)
Goal: 100% deterministic, fast, maintainable tests

## Root Cause Analysis

### 1. Import/Collection Errors (19 tests)
- Old XCS structure references
- Removed operator registry paths  
- Deprecated module imports

### 2. Mock/Dependency Issues (7 tests)
- API key requirements in unit tests
- External service dependencies
- Improper mock paths

### 3. API Mismatches (6 tests)  
- Method name changes (take→limit, to_list→collect)
- Registry path changes
- Signature mismatches

## The Masters' Strategy

### Phase 1: Foundation (Ritchie/Carmack)
"Fix the plumbing before the porcelain"

```python
# 1. Create test infrastructure module
# tests/infrastructure.py
class TestInfrastructure:
    """Centralized test configuration."""
    
    @staticmethod
    def mock_llm(responses=None):
        """Deterministic LLM for testing."""
        return MockLLM(
            responses=responses or {"default": "test response"},
            deterministic=True
        )
    
    @staticmethod
    def mock_data_source(data):
        """In-memory data source."""
        return InMemorySource(data)
```

### Phase 2: Systematic Fixes (Dean/Ghemawat)
"Measure twice, cut once"

1. **Fix all import paths**
   ```bash
   # Script to update all test imports
   find tests -name "*.py" -exec sed -i '' \
     -e 's/ember.core.registry.model/ember.api.models/g' \
     -e 's/ember.core.registry.operator/ember.core.operators/g' \
     -e 's/ember.xcs.schedulers/ember.xcs.jit/g' {} \;
   ```

2. **Create proper mocks**
   ```python
   # tests/mocks.py
   class MockProvider(BaseProvider):
       """Test provider that doesn't need API keys."""
       def __init__(self, responses=None):
           self.responses = responses or {}
           self.call_count = 0
       
       def complete(self, messages, **kwargs):
           self.call_count += 1
           return ChatResponse(
               content=self.responses.get(
                   messages[0].content, 
                   "Mock response"
               ),
               model="mock-model",
               usage=Usage(input_tokens=10, output_tokens=10)
           )
   ```

### Phase 3: Test Patterns (Martin)
"Clean tests are as important as clean code"

```python
class TestPattern:
    """Standard test structure all tests should follow."""
    
    def test_behavior_not_implementation(self):
        # GIVEN - Clear setup
        operator = MyOperator()
        input_data = {"text": "hello"}
        
        # WHEN - Single action
        result = operator(input_data)
        
        # THEN - Assert behavior
        assert result["processed"] == "HELLO"
        # Not: assert operator._internal_state == something
```

### Phase 4: Performance Guards (Page/Brockman)
"What gets measured gets improved"

```python
# tests/benchmarks/performance_guards.py
class PerformanceGuards:
    """Prevent performance regressions."""
    
    OPERATOR_LATENCY_MS = 10  # Single operator < 10ms
    CHAIN_OVERHEAD_PERCENT = 5  # Chain adds < 5% overhead
    MEMORY_GROWTH_MB = 1  # No test leaks > 1MB
```

### Phase 5: Developer Experience (Jobs)
"It just works"

```python
# Simple test runner with clear output
# tests/run_tests.py
def run_tests():
    """Run tests with beautiful, actionable output."""
    print("🧪 Running Ember Test Suite")
    
    results = pytest.main([
        "--tb=short",  # Short tracebacks
        "--durations=10",  # Show slow tests
        "-ra",  # Show all test outcomes
        "--strict-markers",  # Enforce test categories
    ])
    
    if results == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Fix the above failures")
        print("💡 Run with --pdb to debug")
```

## Implementation Order

### Week 1: Critical Path
1. Fix import/collection errors (1 day)
2. Create mock infrastructure (1 day)
3. Fix API mismatches (1 day)
4. Ensure determinism (2 days)

### Week 2: Robustness
1. Add performance benchmarks
2. Improve error messages
3. Create golden test suite
4. Document test patterns

## Success Metrics
- 100% tests passing
- Zero flaky tests (1000 runs)
- Full suite < 60 seconds
- New developer can write test in < 5 minutes

## Test Categories

### Unit Tests (Fast, Isolated)
```python
@pytest.mark.unit
def test_operator_transform():
    """Test operator in isolation."""
    op = UppercaseOperator()
    assert op("hello") == "HELLO"
```

### Integration Tests (Real Components)
```python
@pytest.mark.integration
def test_model_operator_pipeline():
    """Test real components together."""
    model = MockLLM()  # Still mocked external deps
    op = ModelOperator(model)
    result = op("What is 2+2?")
    assert "4" in result
```

### Golden Tests (Regression Prevention)
```python
@pytest.mark.golden
def test_complex_workflow():
    """Ensure complex behaviors don't regress."""
    result = run_analysis_pipeline(GOLDEN_INPUT)
    assert_matches_golden(result, "analysis_v1.json")
```

## The Knuth Touch
"Beware of bugs in the above code; I have only proved it correct, not tried it."

Every test should:
1. Document the contract
2. Prove correctness
3. Run deterministically
4. Fail with clear messages
5. Complete quickly

## Notes from Each Master

**Jeff Dean**: "Make the common case fast. Mock external services."

**Sanjay Ghemawat**: "Test resource cleanup. No test should leak."

**Steve Jobs**: "If a test fails, the error should tell you exactly what to fix."

**Robert C. Martin**: "A test should have one reason to fail."

**Greg Brockman**: "Test the AI behaviors users actually care about."

**Dennis Ritchie**: "Test the interface, not the implementation."

**Donald Knuth**: "Test edge cases with the same rigor as happy paths."

**Larry Page**: "If it takes > 1 second, it's too slow."

**John Carmack**: "Deterministic reproduction is non-negotiable."