# Ember Examples Implementation Plan

## Overview
This document outlines the implementation plan for the new Ember examples structure, including golden tests for CI integration.

## Implementation Status

### ✅ Completed
1. **Directory Structure**: Created numbered directories (01-10) with clear progression
2. **Legacy Preservation**: Moved existing examples to `legacy/` subdirectory
3. **Main README**: Created comprehensive navigation guide
4. **Shared Utilities**: Created `_shared/` directory with common helpers
5. **First Example**: Implemented `hello_world.py` with golden test

### 🚧 In Progress
- Golden test framework for all examples
- Core examples for each section

### 📋 To Do
- Complete examples for all sections
- Create README for each directory
- Update test configurations
- Add notebook examples

## Example Templates

### Standard Example Structure
```python
"""
Example: [Title]
Difficulty: Basic/Intermediate/Advanced
Time: ~X minutes
Prerequisites: [List previous examples]

Learning Objectives:
- Objective 1
- Objective 2

Key Concepts:
- Concept 1
- Concept 2
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import *
from ember.api import models, operators, non, xcs, data

def main():
    """Main example logic."""
    # Example implementation
    pass

if __name__ == "__main__":
    main()
```

### Golden Test Template
```python
def test_example_name(self):
    """Test that example_name.py runs successfully."""
    script_path = EXAMPLES_DIR / "example_name.py"
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert result.returncode == 0, f"Script failed with: {result.stderr}"
    # Add specific output checks
```

## Priority Examples to Implement

### Phase 1: Core Learning Path (Week 1)
1. **01_getting_started/**
   - ✅ hello_world.py
   - ⬜ first_model_call.py
   - ⬜ model_comparison.py
   - ⬜ basic_prompt_engineering.py

2. **02_core_concepts/**
   - ⬜ operators_basics.py
   - ⬜ type_safety.py
   - ⬜ context_management.py
   - ⬜ error_handling.py

3. **03_operators/**
   - ⬜ custom_operator.py
   - ⬜ operator_composition.py

4. **04_compound_ai/**
   - ⬜ simple_ensemble.py
   - ⬜ judge_synthesis.py

### Phase 2: Practical Patterns (Week 2)
1. **09_practical_patterns/**
   - ⬜ rag_pattern.py
   - ⬜ chain_of_thought.py
   - ⬜ structured_output.py
   - ⬜ retry_with_feedback.py

2. **10_evaluation_suite/**
   - ⬜ accuracy_evaluation.py
   - ⬜ consistency_testing.py
   - ⬜ benchmark_harness.py

### Phase 3: Advanced Features (Week 3)
1. **05_data_processing/**
   - ⬜ loading_datasets.py
   - ⬜ streaming_data.py
   - ⬜ batch_evaluation.py

2. **06_performance/**
   - ⬜ jit_basics.py
   - ⬜ parallelization.py
   - ⬜ caching_patterns.py

3. **07_advanced_patterns/**
   - ⬜ production_pipeline.py
   - ⬜ custom_schedulers.py

4. **08_integrations/**
   - ⬜ fastapi_server.py
   - ⬜ gradio_ui.py

## Golden Test Strategy

### Test Organization
```
tests/golden/
├── test_01_getting_started.py
├── test_02_core_concepts.py
├── test_03_operators.py
├── test_04_compound_ai.py
├── test_05_data_processing.py
├── test_06_performance.py
├── test_07_advanced_patterns.py
├── test_08_integrations.py
├── test_09_practical_patterns.py
└── test_10_evaluation_suite.py
```

### CI Integration
1. **Fast Tests**: Examples that complete in <30s run on every commit
2. **Integration Tests**: Examples with external API calls run on merge to main
3. **Performance Tests**: Benchmarks run nightly
4. **Notebook Tests**: Converted to scripts and tested weekly

### Test Categories
- **Smoke Tests**: Verify example runs without errors
- **Output Tests**: Check for expected output patterns
- **Performance Tests**: Ensure examples complete within time limits
- **API Tests**: Mock external API calls for reliability

## Success Metrics
1. **Coverage**: 100% of examples have golden tests
2. **Reliability**: <1% flake rate in CI
3. **Performance**: 90% of examples complete in <30s
4. **Documentation**: Every example has clear learning objectives

## Migration Checklist
- [x] Create new directory structure
- [x] Move legacy examples
- [x] Create main README
- [x] Create shared utilities
- [x] Implement first example with test
- [ ] Create all directory READMEs
- [ ] Implement Phase 1 examples
- [ ] Create golden tests for Phase 1
- [ ] Update CI configuration
- [ ] Implement Phase 2 examples
- [ ] Create golden tests for Phase 2
- [ ] Implement Phase 3 examples
- [ ] Create golden tests for Phase 3
- [ ] Add notebook examples
- [ ] Final documentation review

## Notes
- Each example should be self-contained and runnable
- Use mock models for examples to avoid API dependencies where possible
- Include both success and error cases in examples
- Ensure examples follow Google Python Style Guide
- Test on fresh environment to catch missing dependencies