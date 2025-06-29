# Ember Examples Improvement Plan

## Overview

This document outlines a comprehensive plan to:
1. Roll out the shared conditional execution pattern across all examples
2. Implement a golden testing framework for examples
3. Ensure all examples are executable, testable, and provide value both with and without API keys

## Phase 1: Golden Test Infrastructure

### 1.1 Create Base Test Framework
**Files to create:**
- `tests/examples/__init__.py`
- `tests/examples/test_base.py` - Base class for example tests
- `tests/examples/conftest.py` - Pytest fixtures and configuration
- `tests/examples/golden_outputs/` - Directory for golden output files

**Base Test Class Features:**
```python
class ExampleGoldenTest:
    - capture_output() - Captures stdout/stderr from example execution
    - validate_golden() - Compares output against golden files
    - test_with_api_keys() - Tests real execution (skipped in CI without keys)
    - test_without_api_keys() - Tests simulated execution
    - test_performance_bounds() - Ensures examples run within time limits
    - test_imports() - Validates all imports work correctly
```

### 1.2 Golden Output Management
**Files to create:**
- `tests/examples/update_golden.py` - Script to update golden outputs
- `tests/examples/golden_outputs/{example_name}_simulated.json`
- `tests/examples/golden_outputs/{example_name}_real.json` (optional)

**Golden Output Format:**
```json
{
  "version": "1.0",
  "example": "01_getting_started/first_model_call.py",
  "execution_mode": "simulated",
  "sections": [
    {
      "header": "Basic Model Call",
      "output": "Response: This is a simulated response...",
      "timing": 0.5
    }
  ],
  "total_time": 2.1,
  "api_keys_required": ["OPENAI_API_KEY"],
  "metrics": {
    "lines_of_code": 45,
    "api_calls": 1
  }
}
```

## Phase 2: Example Migration Plan

### 2.1 High Priority Migrations (Week 1)

#### File: `examples/01_getting_started/first_model_call.py`
**Current State:** Uses `ensure_api_key()` with manual demo_mode
**Migration Steps:**
1. Import `@conditional_llm` from `_shared.conditional_execution`
2. Replace demo_mode function with simulated_execution parameter
3. Wrap main() with decorator
4. Test both execution paths
5. Generate golden outputs

#### File: `examples/01_getting_started/basic_prompt_engineering.py`
**Current State:** Uses `ensure_api_key()` with manual demo_mode
**Migration Steps:**
1. Import `@conditional_llm` decorator
2. Convert demo examples to simulated responses
3. Ensure prompt engineering concepts are clear in both modes
4. Add realistic timing to simulated responses
5. Generate golden outputs

#### File: `examples/03_simplified_apis/natural_api_showcase.py`
**Current State:** Manual API key checking
**Migration Steps:**
1. Import decorator and update imports section
2. Create comprehensive simulated responses for all API patterns
3. Preserve educational value of API design patterns
4. Test streaming functionality in both modes
5. Generate golden outputs

#### File: `examples/03_simplified_apis/simplified_workflows.py`
**Current State:** Manual API key checking
**Migration Steps:**
1. Apply decorator to main function
2. Design simulated responses that demonstrate workflow patterns
3. Ensure error handling examples work in both modes
4. Validate workflow progression
5. Generate golden outputs

#### File: `examples/03_simplified_apis/zero_config_jit.py`
**Current State:** Manual API key checking
**Migration Steps:**
1. Apply decorator while preserving JIT demonstrations
2. Create simulated responses that show performance characteristics
3. Ensure JAX transformations work in both modes
4. Add timing comparisons
5. Generate golden outputs

### 2.2 Medium Priority Migrations (Week 2)

#### Directory: `examples/02_core_concepts/`
**Files:** 5 files needing investigation
**Migration Strategy:**
1. Audit each file for API usage
2. Apply decorator where appropriate
3. Create educational simulated responses
4. Focus on concept clarity over realistic outputs

#### Directory: `examples/04_compound_ai/`
**Files:** 3 files (judge_synthesis.py already migrated)
**Migration Strategy:**
1. Follow judge_synthesis.py pattern
2. Create complex simulated responses for compound patterns
3. Ensure multi-step workflows are clear

#### Directory: `examples/09_practical_patterns/`
**Files:** chain_of_thought.py, rag_pattern.py, structured_output.py
**Migration Strategy:**
1. Create sophisticated simulated responses
2. Show realistic CoT reasoning in simulation
3. Mock RAG retrieval results
4. Generate valid structured outputs

### 2.3 Low Priority Migrations (Week 3)

#### Remaining directories and files
**Strategy:**
1. Quick audit to identify API usage
2. Skip files with no API calls
3. Apply standard migration pattern
4. Focus on maintaining educational value

## Phase 3: Testing Implementation

### 3.1 Test File Structure
```
tests/examples/
├── __init__.py
├── test_base.py
├── conftest.py
├── update_golden.py
├── golden_outputs/
│   ├── 01_getting_started/
│   │   ├── first_model_call_simulated.json
│   │   └── basic_prompt_engineering_simulated.json
│   └── ... (other examples)
├── test_01_getting_started.py
├── test_02_core_concepts.py
├── test_03_simplified_apis.py
├── test_04_compound_ai.py
├── test_05_data_processing.py
├── test_06_performance_optimization.py
├── test_07_error_handling.py
├── test_08_advanced_patterns.py
├── test_09_practical_patterns.py
└── test_10_evaluation_suite.py
```

### 3.2 Test Implementation Pattern
```python
# tests/examples/test_01_getting_started.py
import pytest
from .test_base import ExampleGoldenTest

class TestGettingStartedExamples(ExampleGoldenTest):
    
    def test_hello_world(self):
        """Test the hello_world.py example."""
        self.run_example_test("01_getting_started/hello_world.py")
    
    def test_first_model_call(self):
        """Test the first_model_call.py example."""
        self.run_example_test(
            "01_getting_started/first_model_call.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=30.0
        )
    
    def test_basic_prompt_engineering(self):
        """Test the basic_prompt_engineering.py example."""
        self.run_example_test(
            "01_getting_started/basic_prompt_engineering.py",
            requires_api_keys=["OPENAI_API_KEY"],
            validate_sections=["Temperature Effects", "System Messages"]
        )
```

### 3.3 CI Pipeline Configuration
**File:** `.github/workflows/test-examples.yml`
```yaml
name: Test Examples

on:
  push:
    paths:
      - 'examples/**'
      - 'tests/examples/**'
  pull_request:
    paths:
      - 'examples/**'
      - 'tests/examples/**'

jobs:
  test-examples:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r examples/requirements.txt
        pip install pytest pytest-timeout
    
    - name: Run example tests (simulated only)
      run: |
        pytest tests/examples/ -v --no-api-keys
    
    - name: Validate golden outputs
      run: |
        python tests/examples/validate_golden.py
```

## Phase 4: Quality Assurance

### 4.1 Validation Checklist
For each migrated example:
- [ ] Runs without API keys (simulated mode)
- [ ] Runs with API keys (real mode)
- [ ] Produces educational output in both modes
- [ ] Completes within performance bounds
- [ ] Has generated golden outputs
- [ ] Passes all automated tests
- [ ] Maintains original educational intent

### 4.2 Metrics Collection
**File:** `tests/examples/collect_metrics.py`
- Count API calls per example
- Measure execution times
- Track code complexity
- Generate coverage reports
- Create example health dashboard

### 4.3 Documentation Updates
**Files to update:**
- `examples/README.md` - Document new testing approach
- `CONTRIBUTING.md` - Add example contribution guidelines
- Individual example files - Update docstrings

## Implementation Timeline

**Week 1:**
- Set up test infrastructure
- Migrate high-priority examples
- Generate initial golden outputs

**Week 2:**
- Continue migrations (medium priority)
- Refine test framework based on learnings
- Set up CI pipeline

**Week 3:**
- Complete remaining migrations
- Full test suite execution
- Documentation updates
- Metrics dashboard

**Week 4:**
- Quality assurance pass
- Performance optimization
- Final documentation
- Team training on new patterns

## Success Criteria

1. **100% Example Coverage:** All examples that make API calls use `@conditional_llm`
2. **Dual Execution:** Every example works both with and without API keys
3. **Test Coverage:** Every example has at least one test
4. **Performance:** All examples complete within defined time bounds
5. **CI Integration:** Automated testing on every PR
6. **Documentation:** Clear guidelines for adding new examples
7. **Metrics:** Dashboard showing example health and usage patterns

## Risks and Mitigations

**Risk:** Breaking existing examples
**Mitigation:** Comprehensive testing before migration, gradual rollout

**Risk:** Simulated responses becoming stale
**Mitigation:** Regular golden output updates, version tracking

**Risk:** Performance regression
**Mitigation:** Performance bounds in tests, monitoring

**Risk:** Loss of educational value
**Mitigation:** Careful review of simulated responses, user feedback

## Next Steps

1. Review and approve this plan
2. Create test infrastructure (Phase 1)
3. Begin high-priority migrations
4. Set up monitoring and metrics
5. Iterate based on feedback