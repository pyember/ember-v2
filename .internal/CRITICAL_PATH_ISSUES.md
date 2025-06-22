# Critical Path Issues

*GitHub issue templates for the must-fix items*

## Issue #1: Remove Public Registry Access

**Title**: Remove `ModelsAPI.get_registry()` public method

**Labels**: `tech-debt`, `must-have`, `breaking-change`

**Description**:
The models API exposes internal registry details through `get_registry()`. This violates our principle of hiding complexity.

**Acceptance Criteria**:
- [ ] Method renamed to `_get_registry()` or removed
- [ ] No public API exposes registry internals
- [ ] Contract test verifies registry not in `dir(ember.api.models)`

**Time Estimate**: 1 hour

---

## Issue #2: Deprecate Duplicate Operators API

**Title**: Deprecate `operators.py` in favor of `operators_v2.py`

**Labels**: `tech-debt`, `must-have`, `api-cleanup`

**Description**:
We have two operator APIs. This confuses users and maintains unnecessary code.

**Acceptance Criteria**:
- [ ] `operators.py` shows deprecation warning
- [ ] All exports redirect to `operators_v2`
- [ ] Migration guide created
- [ ] Deletion date set (2 weeks from now)

**Time Estimate**: 2 hours

---

## Issue #3: Scaffold IR System

**Title**: Create minimal IR package for XCS

**Labels**: `enhancement`, `must-have`, `foundation`

**Description**:
XCS improvements are blocked on missing IR system. Need minimal implementation.

**Files to Create**:
```
src/ember/ir/
  __init__.py
  ops.py      # OpType enum, Operation dataclass
  graph.py    # Graph class with optimize() stub
  builder.py  # GraphBuilder for constructing IR
```

**Acceptance Criteria**:
- [ ] Can create Operation with LLM_CALL type
- [ ] Can build Graph from Operations
- [ ] `optimize()` exists (can be no-op)
- [ ] Basic tests pass

**Time Estimate**: 4 hours

---

## Issue #4: Implement DatasetMetadata

**Title**: Replace complex DatasetInfo with minimal DatasetMetadata

**Labels**: `enhancement`, `simplification`

**Description**:
Current DatasetInfo is overengineered. Implement the 8-field version from philosophy doc.

**Acceptance Criteria**:
- [ ] `DatasetMetadata` class created with exactly 8 fields
- [ ] DataContext returns new metadata
- [ ] Old DatasetInfo deprecated
- [ ] Tests updated

**Time Estimate**: 3 hours

---

## Issue #5: CI/CD Module Size Enforcement

**Title**: Add automated module size checks

**Labels**: `ci/cd`, `must-have`, `quality`

**Description**:
Without automated enforcement, modules will grow beyond 1000 lines.

**Acceptance Criteria**:
- [ ] Script counts lines per module
- [ ] CI fails if any module > 1000 LOC
- [ ] Pre-commit hook warns locally
- [ ] Current violations documented

**Time Estimate**: 2 hours

---

## Issue #6: Contract Tests for Public API

**Title**: Add regression tests for public API surface

**Labels**: `testing`, `must-have`, `quality`

**Description**:
Need tests that fail if public API grows or leaks internals.

**Test Cases**:
```python
def test_models_api_minimal():
    api = dir(ember.api.models)
    assert len(api) < 10
    assert "get_registry" not in api

def test_no_duplicate_operators():
    # Should not import without warning
    with pytest.warns(DeprecationWarning):
        import ember.api.operators
```

**Acceptance Criteria**:
- [ ] Test file created
- [ ] Tests for all 4 modules
- [ ] CI runs tests
- [ ] Tests currently pass

**Time Estimate**: 2 hours

---

## Issue #7: Export Missing Decorators

**Title**: Export `validate` decorator from operators_v2

**Labels**: `bug`, `api-completion`

**Description**:
Documentation mentions `@validate` but it's not exported.

**Acceptance Criteria**:
- [ ] `validate` decorator implemented
- [ ] Exported in `__all__`
- [ ] Example in docstring
- [ ] Test coverage

**Time Estimate**: 1 hour

---

## Issue #8: Fix or Remove Async Methods

**Title**: Implement or remove `async_call` methods

**Labels**: `bug`, `api-completion`

**Description**:
Several async methods raise `NotImplementedError`. Either implement or remove.

**Acceptance Criteria**:
- [ ] All async methods either work or don't exist
- [ ] No `NotImplementedError` in public API
- [ ] Tests for any implemented async methods

**Time Estimate**: 3 hours

---

## Execution Order

1. **Blocking Issues** (Day 0 Morning):
   - #1: Remove public registry (1h)
   - #2: Deprecate operators.py (2h)
   - #8: Fix async methods (1h of 3h)

2. **Foundation** (Day 0 Afternoon):
   - #3: Scaffold IR (4h)
   - #7: Export validate (1h)

3. **Quality Gates** (Day 0.5):
   - #5: Module size CI (2h)
   - #6: Contract tests (2h)
   - #4: DatasetMetadata (3h)
   - #8: Complete async fixes (2h)

Total: ~20 hours of focused work before Week 1 can properly begin.

---

## The North Star Test

For each issue ask:
- Does fixing this remove confusion? ✓
- Does it make the codebase smaller? ✓
- Would the masters approve? ✓

All issues pass. Ship them.