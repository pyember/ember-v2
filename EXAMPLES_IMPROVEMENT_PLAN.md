# Comprehensive Examples Improvement Plan

## Executive Summary

This plan outlines a systematic approach to updating all Ember examples to use the latest APIs, ensure consistency, and maintain quality through golden tests.

## Current State Analysis

### Test Coverage
- **Total Examples**: ~40 files across 7 categories
- **Golden Tests Written**: 38 tests
- **Current Pass Rate**: 34% (13/38 tests passing)

### Key Issues Identified

1. **API Misalignment**
   - Many examples use outdated `initialize_registry()` pattern
   - Deep imports (`ember.core.registry.model`) instead of simplified API
   - Missing new models API patterns (`models("gpt-4", "prompt")`)

2. **Documentation Issues**
   - References to "poetry run" instead of "uv run"
   - Outdated import statements in docstrings
   - Missing or incomplete example outputs

3. **Consistency Issues**
   - Different patterns used across similar examples
   - Inconsistent error handling
   - Variable naming conventions

## Improvement Plan

### Phase 1: API Alignment (Priority: Critical)

#### 1.1 Merge/Rebase Strategy
- **Action**: Determine correct branch strategy
- **Options**:
  a. Merge `models-deep-review-phase1` into current branch
  b. Rebase current work onto `models-deep-review-phase1`
  c. Cherry-pick specific API improvements
- **Decision Criteria**: Minimize conflicts while getting latest API

#### 1.2 Update Models Examples
Transform all model examples to use new patterns:

```python
# OLD Pattern
from ember.core.registry.model import initialize_registry
registry = initialize_registry()
model = registry.get_model("gpt-4")

# NEW Pattern
from ember.api import models
response = models("gpt-4", "What is 2+2?")
```

**Files to Update**:
- `models/register_models_directly.py`
- `models/model_registry_direct.py`
- `models/list_models.py`
- `models/function_style_api.py`
- `models/model_registry_example.py`

#### 1.3 Update Import Patterns
Replace deep imports with API imports across all examples:

```python
# OLD
from ember.core.registry.operator.base import Operator
from ember.core.utils.data import load_dataset

# NEW
from ember.api.operators import Operator
from ember.api.data import load_dataset
```

### Phase 2: Example Categories Update (Priority: High)

#### 2.1 Basic Examples
- ✅ `minimal_example.py` - Already good
- 🔧 `minimal_operator_example.py` - Minor updates needed
- 🔧 `compact_notation_example.py` - Update mock handling
- ✅ `context_example.py` - Working
- 🔧 `simple_jit_demo.py` - Fix imports
- 🔧 `check_env.py` - Update for new environment

#### 2.2 Models Examples
- 🔧 All files need simplified API updates
- Add examples showing:
  - Error handling with new exceptions
  - Model binding for performance
  - Provider-specific features

#### 2.3 Operators Examples
- 🔧 Update deep imports to API imports
- Add examples for:
  - Operator composition with new API
  - Async operator patterns
  - Integration with models API

#### 2.4 Data Examples
- 🔧 Ensure compatibility with data API
- Add examples for:
  - Streaming datasets
  - Custom transformations
  - Integration with evaluation

#### 2.5 XCS Examples
- 🔧 Update for latest JIT improvements
- Add examples for:
  - Performance comparisons
  - Different scheduling strategies
  - Transform combinations

#### 2.6 Advanced Examples
- 🔧 Major updates needed for API alignment
- Focus on real-world patterns:
  - Multi-model ensembles
  - Evaluation pipelines
  - Production patterns

### Phase 3: Quality Assurance (Priority: High)

#### 3.1 Golden Test Enhancement
- Fix mock issues in test fixtures
- Add output validation for all examples
- Create snapshot tests for complex outputs

#### 3.2 Documentation Standards
- Create example template with sections:
  - Purpose and use case
  - Prerequisites
  - Code walkthrough
  - Expected output
  - Common variations

#### 3.3 CI/CD Integration
- Add golden tests to GitHub Actions
- Create example validation workflow
- Auto-generate example gallery

### Phase 4: New Examples (Priority: Medium)

#### 4.1 Missing Patterns
- **Structured Output**: Using models with typed responses
- **Streaming**: Real-time model responses
- **Batch Processing**: Efficient multi-prompt handling
- **Cost Optimization**: Usage tracking and optimization
- **Error Recovery**: Handling API failures gracefully

#### 4.2 Integration Examples
- Model + Data pipeline
- Operator + XCS optimization
- Full evaluation workflow
- Production deployment patterns

### Phase 5: Developer Experience (Priority: Medium)

#### 4.1 Interactive Examples
- Jupyter notebook versions
- Gradio/Streamlit demos
- CLI interactive mode

#### 4.2 Example Discovery
- Categorized example index
- Search by use case
- Difficulty levels (beginner/intermediate/advanced)

## Implementation Timeline

### Week 1: Foundation
- [ ] Resolve branch strategy
- [ ] Update all models examples
- [ ] Fix golden test infrastructure

### Week 2: Core Updates  
- [ ] Update basic examples
- [ ] Update operator examples
- [ ] Update data examples

### Week 3: Advanced Updates
- [ ] Update XCS examples
- [ ] Update advanced examples
- [ ] Update integration examples

### Week 4: Quality & Polish
- [ ] Complete golden test coverage
- [ ] Documentation review
- [ ] Add new examples for gaps

## Success Metrics

1. **Test Coverage**: 100% of examples have golden tests
2. **Pass Rate**: 95%+ golden tests passing
3. **API Consistency**: All examples use simplified APIs
4. **Documentation**: Every example has clear purpose and output
5. **Discoverability**: Examples easy to find and understand

## Maintenance Strategy

1. **Automated Checks**
   - Pre-commit hooks for example validation
   - CI checks for API compatibility
   - Regular dependency updates

2. **Review Process**
   - All new examples require golden tests
   - API changes trigger example updates
   - Quarterly example audit

3. **Community Contribution**
   - Example contribution guide
   - Template for new examples
   - Review checklist

## Next Steps

1. **Immediate Actions**
   - Fix mock_lm fixture in conftest.py
   - Update model_api_example.py to test
   - Create PR for branch merge strategy

2. **This Week**
   - Update 5 highest-priority examples
   - Fix failing golden tests
   - Document patterns guide

3. **Ongoing**
   - Track progress in GitHub project
   - Weekly example updates
   - Community feedback integration