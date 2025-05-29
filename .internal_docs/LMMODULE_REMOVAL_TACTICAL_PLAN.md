# LMModule Removal - Tactical Execution Plan

## Overview

This document breaks down the LMModule removal into tactical work slices with detailed checklists. Each slice is designed to be completed in 1-2 days by a single developer.

## Work Organization

- **Total Duration**: 4 weeks
- **Work Slices**: 20 slices (1-2 days each)
- **Dependencies**: Clearly marked between slices
- **Risk Level**: ⚠️ Low | ⚠️⚠️ Medium | ⚠️⚠️⚠️ High

---

## Phase 1: Foundation (Week 1)

### Slice 1.1: Analysis and Documentation (Day 1)
**Owner**: Lead Developer  
**Risk**: ⚠️ Low  
**Dependencies**: None

#### Tasks:
- [ ] Run dependency analysis script to find all LMModule usage
  ```bash
  grep -r "LMModule\|LMModuleConfig" src/ --include="*.py" > lmmodule_usage.txt
  ```
- [ ] Document all files that import LMModule
- [ ] Create dependency graph of affected components
- [ ] Identify high-risk areas (operators with complex LMModule usage)
- [ ] Review and update migration plan based on findings
- [ ] Create tracking spreadsheet for migration progress

#### Deliverables:
- [ ] `lmmodule_usage.txt` - Complete list of usage
- [ ] `dependency_graph.md` - Visual/text representation
- [ ] `risk_assessment.md` - Areas of concern

### Slice 1.2: Setup Test Infrastructure (Day 2)
**Owner**: Test Engineer  
**Risk**: ⚠️ Low  
**Dependencies**: None

#### Tasks:
- [ ] Create test file: `tests/migration/test_lmmodule_compatibility.py`
- [ ] Write baseline tests for current LMModule behavior
  ```python
  def test_lmmodule_basic_invocation():
      """Baseline test for LMModule behavior."""
      config = LMModuleConfig(id="gpt-3.5-turbo", temperature=0.7)
      lm = LMModule(config=config)
      response = lm(prompt="Hello")
      assert isinstance(response, str)
  ```
- [ ] Create performance benchmark suite
  ```python
  def benchmark_lmmodule_vs_modelbinding():
      """Compare performance of both approaches."""
      # Implementation
  ```
- [ ] Set up CI job for migration tests
- [ ] Create test data fixtures for migration validation

#### Deliverables:
- [ ] Test suite with 20+ baseline tests
- [ ] Performance benchmark results
- [ ] CI configuration for migration tests

### Slice 1.3: Create Compatibility Layer (Day 3-4)
**Owner**: Senior Developer  
**Risk**: ⚠️⚠️ Medium  
**Dependencies**: Slice 1.1, 1.2

#### Tasks:
- [ ] Create `src/ember/core/registry/model/model_module/lm_compat.py`
- [ ] Implement compatibility wrapper:
  ```python
  import warnings
  from ember.api import models, ModelBinding
  
  class LMModule:
      """Compatibility wrapper for LMModule -> ModelBinding migration."""
      
      def __init__(self, config=None, model_service=None):
          warnings.warn(
              "LMModule is deprecated and will be removed in v2.0. "
              "Use models.bind() instead. "
              "See LMMODULE_MIGRATION_GUIDE.md for details.",
              DeprecationWarning,
              stacklevel=2
          )
          # Implementation
  ```
- [ ] Add logging for migration tracking
- [ ] Implement all LMModule methods using ModelBinding
- [ ] Add compatibility for LMModuleConfig
- [ ] Write comprehensive tests for compatibility layer

#### Deliverables:
- [ ] Compatibility module fully tested
- [ ] All existing LMModule tests passing with wrapper
- [ ] Migration logging implemented

### Slice 1.4: Update Base Operator (Day 5)
**Owner**: Core Team Developer  
**Risk**: ⚠️⚠️ Medium  
**Dependencies**: Slice 1.3

#### Tasks:
- [ ] Update `src/ember/core/registry/operator/base/operator_base.py`
- [ ] Add `_normalize_models` method:
  ```python
  def _normalize_models(self, model_specs, default_params=None):
      """Convert model specifications to ModelBinding instances."""
      # Implementation from spec
  ```
- [ ] Add backward compatibility checks
- [ ] Update operator initialization to handle both patterns
- [ ] Add deprecation warnings for LMModule usage
- [ ] Update base operator tests
- [ ] Document new methods in docstrings

#### Deliverables:
- [ ] Updated BaseOperator with dual support
- [ ] All base operator tests passing
- [ ] Documentation updated

---

## Phase 2: Core Operator Migration (Week 2)

### Slice 2.1: Migrate VerifierOperator (Day 6)
**Owner**: Backend Developer  
**Risk**: ⚠️⚠️ Medium  
**Dependencies**: Slice 1.4

#### Pre-migration Checklist:
- [ ] Review current VerifierOperator implementation
- [ ] Identify all test files using VerifierOperator
- [ ] Create migration branch: `feature/migrate-verifier-operator`

#### Migration Tasks:
- [ ] Update constructor signature:
  ```python
  def __init__(self, model: Union[str, ModelBinding], **kwargs):
  ```
- [ ] Replace LMModule usage with ModelBinding
- [ ] Update `forward` method to use Response object
- [ ] Add proper error handling for new exceptions
- [ ] Maintain backward compatibility during transition

#### Test Updates:
- [ ] Update `test_verifier_operator.py`
- [ ] Ensure all integration tests pass
- [ ] Add new tests for ModelBinding usage
- [ ] Performance comparison test

#### Post-migration:
- [ ] Run full test suite
- [ ] Update operator documentation
- [ ] Create PR with migration changes
- [ ] Get code review approval

### Slice 2.2: Migrate EnsembleOperator (Day 7-8)
**Owner**: Senior Developer  
**Risk**: ⚠️⚠️⚠️ High (Most complex operator)  
**Dependencies**: Slice 2.1

#### Pre-migration Checklist:
- [ ] Analyze EnsembleOperator's multiple model handling
- [ ] Review aggregation methods
- [ ] Create detailed migration plan for this operator
- [ ] Set up integration test environment

#### Migration Tasks:
- [ ] Update constructor for multiple models:
  ```python
  def __init__(self, models: List[Union[str, ModelBinding]], **kwargs):
  ```
- [ ] Refactor `_initialize_models` method
- [ ] Update response aggregation for Response objects
- [ ] Handle mixed model types (strings and bindings)
- [ ] Preserve all aggregation methods
- [ ] Add comprehensive error handling

#### Complex Cases:
- [ ] Handle dynamic model addition/removal
- [ ] Ensure thread safety for concurrent execution
- [ ] Maintain performance for large ensembles
- [ ] Test with 10+ models

#### Test Updates:
- [ ] Update all ensemble tests
- [ ] Add edge case tests (empty ensemble, single model)
- [ ] Performance benchmarks with various ensemble sizes
- [ ] Integration tests with real models

### Slice 2.3: Migrate MostCommonOperator (Day 9)
**Owner**: Developer  
**Risk**: ⚠️ Low  
**Dependencies**: Slice 2.2

#### Tasks:
- [ ] Update to use migrated EnsembleOperator
- [ ] Adjust response counting for Response objects
- [ ] Update threshold checking logic
- [ ] Migrate all tests
- [ ] Verify consensus logic unchanged

### Slice 2.4: Migrate SynthesisJudgeOperator (Day 10)
**Owner**: ML Engineer  
**Risk**: ⚠️⚠️ Medium  
**Dependencies**: Slice 2.1

#### Tasks:
- [ ] Update for single model usage
- [ ] Migrate prompt construction
- [ ] Handle structured output parsing
- [ ] Update judgment criteria handling
- [ ] Migrate all related tests

---

## Phase 3: Examples and Documentation (Week 3)

### Slice 3.1: Update Basic Examples (Day 11)
**Owner**: Developer Advocate  
**Risk**: ⚠️ Low  
**Dependencies**: Phase 2 completion

#### Example Categories:
- [ ] `examples/basic/minimal_operator_example.py`
- [ ] `examples/basic/context_example.py`
- [ ] `examples/operators/container_operator_example.py`
- [ ] `examples/operators/simplified_ensemble_example.py`

#### For Each Example:
- [ ] Replace LMModule imports with models API
- [ ] Update instantiation patterns
- [ ] Test example runs successfully
- [ ] Update inline comments
- [ ] Verify output matches expected

### Slice 3.2: Update Advanced Examples (Day 12)
**Owner**: Senior Developer  
**Risk**: ⚠️⚠️ Medium  
**Dependencies**: Slice 3.1

#### Advanced Examples:
- [ ] `examples/advanced/ensemble_judge_mmlu.py`
- [ ] `examples/advanced/reasoning_system.py`
- [ ] `examples/advanced/parallel_pipeline_example.py`

#### Tasks:
- [ ] Update complex operator compositions
- [ ] Migrate custom operator implementations
- [ ] Test performance characteristics
- [ ] Update documentation strings

### Slice 3.3: Update Notebooks (Day 13)
**Owner**: Data Scientist  
**Risk**: ⚠️ Low  
**Dependencies**: Slice 3.1

#### Notebooks to Update:
- [ ] `examples/notebooks/operator_tutorial.ipynb`
- [ ] `examples/notebooks/model_usage_tutorial.ipynb`

#### For Each Notebook:
- [ ] Update code cells
- [ ] Re-run all cells
- [ ] Update markdown explanations
- [ ] Verify outputs are correct
- [ ] Export clean version

### Slice 3.4: API Documentation (Day 14)
**Owner**: Technical Writer  
**Risk**: ⚠️ Low  
**Dependencies**: Slices 3.1-3.3

#### Documentation Updates:
- [ ] Update `docs/operators/README.md`
- [ ] Update `docs/models/getting_started.md`
- [ ] Create migration guide in docs
- [ ] Update API reference
- [ ] Add deprecation notices

---

## Phase 4: Testing and Validation (Week 3-4)

### Slice 4.1: Integration Test Suite (Day 15)
**Owner**: QA Engineer  
**Risk**: ⚠️⚠️ Medium  
**Dependencies**: Phase 3 completion

#### Test Categories:
- [ ] End-to-end operator pipelines
- [ ] Multi-operator compositions
- [ ] Error handling scenarios
- [ ] Performance regression tests

#### Specific Tests:
- [ ] Test operator chaining with new pattern
- [ ] Verify error propagation
- [ ] Test model fallback scenarios
- [ ] Concurrent execution tests

### Slice 4.2: Performance Validation (Day 16)
**Owner**: Performance Engineer  
**Risk**: ⚠️⚠️ Medium  
**Dependencies**: Slice 4.1

#### Benchmarks:
- [ ] Single operator invocation latency
- [ ] Ensemble operator with 10+ models
- [ ] Memory usage comparison
- [ ] Concurrent execution throughput

#### Performance Targets:
- [ ] 50% reduction in invocation overhead
- [ ] No regression in throughput
- [ ] 20% memory usage reduction

### Slice 4.3: Backward Compatibility Testing (Day 17)
**Owner**: Senior QA  
**Risk**: ⚠️⚠️⚠️ High  
**Dependencies**: Slice 4.1

#### Compatibility Tests:
- [ ] Mixed usage (LMModule + ModelBinding)
- [ ] Deprecation warnings appear correctly
- [ ] No breaking changes for existing code
- [ ] Serialization/deserialization works

### Slice 4.4: User Acceptance Testing (Day 18)
**Owner**: Product Manager  
**Risk**: ⚠️ Low  
**Dependencies**: Slice 4.3

#### UAT Tasks:
- [ ] Run through migration guide as a user
- [ ] Test common use cases
- [ ] Validate error messages are helpful
- [ ] Check documentation clarity

---

## Phase 5: Cleanup and Release (Week 4)

### Slice 5.1: Remove Deprecations (Day 19)
**Owner**: Lead Developer  
**Risk**: ⚠️⚠️⚠️ High  
**Dependencies**: All previous phases

#### Removal Tasks:
- [ ] Delete `src/ember/core/registry/model/model_module/lm.py`
- [ ] Remove compatibility wrapper
- [ ] Clean up imports across codebase
- [ ] Remove LMModule from `__init__.py` files
- [ ] Update setup.py if needed

#### Verification:
- [ ] No remaining LMModule imports
- [ ] All tests still passing
- [ ] No circular dependencies

### Slice 5.2: Final Validation (Day 20)
**Owner**: Release Manager  
**Risk**: ⚠️⚠️ Medium  
**Dependencies**: Slice 5.1

#### Final Checks:
- [ ] Full test suite passes
- [ ] Performance benchmarks meet targets
- [ ] Documentation is complete
- [ ] Migration guide tested by team
- [ ] No TODO comments left

#### Release Preparation:
- [ ] Update CHANGELOG.md
- [ ] Create release notes
- [ ] Tag release candidate
- [ ] Final security scan

---

## Daily Standup Template

```markdown
## Date: [DATE]
## Slice: [SLICE_NUMBER]

### Completed Yesterday:
- [ ] Task 1
- [ ] Task 2

### Today's Focus:
- [ ] Task 1
- [ ] Task 2

### Blockers:
- None / Description

### Risk Updates:
- Any changes to risk assessment
```

## Progress Tracking

### Week 1 Checklist:
- [ ] Slice 1.1: Analysis ✓
- [ ] Slice 1.2: Test Infrastructure ✓
- [ ] Slice 1.3: Compatibility Layer ✓
- [ ] Slice 1.4: Base Operator ✓

### Week 2 Checklist:
- [ ] Slice 2.1: VerifierOperator
- [ ] Slice 2.2: EnsembleOperator
- [ ] Slice 2.3: MostCommonOperator
- [ ] Slice 2.4: SynthesisJudgeOperator

### Week 3 Checklist:
- [ ] Slice 3.1: Basic Examples
- [ ] Slice 3.2: Advanced Examples
- [ ] Slice 3.3: Notebooks
- [ ] Slice 3.4: Documentation
- [ ] Slice 4.1: Integration Tests
- [ ] Slice 4.2: Performance Tests

### Week 4 Checklist:
- [ ] Slice 4.3: Compatibility Tests
- [ ] Slice 4.4: UAT
- [ ] Slice 5.1: Remove Deprecations
- [ ] Slice 5.2: Final Validation

## Risk Mitigation Strategies

### High Risk Slices:
1. **Slice 2.2 (EnsembleOperator)**: 
   - Pair programming recommended
   - Extra code review required
   - Feature flag for rollback

2. **Slice 5.1 (Remove Deprecations)**:
   - Create full backup branch
   - Run removal in stages
   - Have rollback script ready

### Rollback Procedures:

#### Quick Rollback (< 5 minutes):
```bash
git checkout main
git branch -D feature/remove-lmmodule
git push origin --delete feature/remove-lmmodule
```

#### Compatibility Layer Restoration:
```bash
git checkout [LAST_STABLE_COMMIT] -- src/ember/core/registry/model/model_module/
git commit -m "Restore LMModule for compatibility"
```

## Success Criteria

### Technical Metrics:
- [ ] 100% test coverage maintained
- [ ] < 1ms operator invocation overhead
- [ ] Zero breaking changes for users
- [ ] 500+ lines of code removed

### Business Metrics:
- [ ] Zero customer complaints
- [ ] Positive developer feedback
- [ ] Reduced onboarding time for new devs

## Communication Plan

### Week 1:
- Monday: Announce migration start
- Friday: Progress update email

### Week 2:
- Daily: Slack updates on operator migration
- Friday: Demo migrated operators

### Week 3:
- Monday: Call for early testing
- Wednesday: Documentation review
- Friday: Performance results

### Week 4:
- Monday: Final testing call
- Wednesday: Go/No-go decision
- Friday: Release announcement