# Reality Check: Immediate Actions Before Simplification

*What the code review revealed and what we must do first*

## Critical Insights from the Review

### 1. We're Not Starting from a Clean Slate
- Legacy exports still exist (`operators.py`, `get_registry()`)
- Backward compatibility code contradicts our "clean break" decision
- Missing foundational pieces (IR system, validation decorator)

### 2. The Gap Between Plan and Reality
- **Plan**: "Hide provider registry"
- **Reality**: `ModelsAPI.get_registry()` is public
- **Plan**: "IR system with optimize()"
- **Reality**: No IR implementation exists yet
- **Plan**: "Each module < 1000 lines"
- **Reality**: Deep dependency trees make this hard

### 3. Technical Debt is Blocking Progress
- Duplicate APIs (`operators.py` vs `operators_v2.py`)
- No CI enforcement of our principles
- Missing regression tests for simplification

## Immediate Actions (Before Week 1)

### Day 0: Foundation Fixes (Must Do First)

#### Morning: Clean Up Obvious Issues (4 hours)
- [ ] **Fix ModelsAPI.get_registry()** 
  ```python
  # Change from:
  def get_registry(self):
  # To:
  def _get_registry(self):  # or remove entirely
  ```

- [ ] **Deprecate operators.py**
  ```python
  # operators.py
  """DEPRECATED: Use ember.api.operators_v2 instead."""
  import warnings
  warnings.warn("operators.py is deprecated", DeprecationWarning)
  from ember.api.operators_v2 import *
  ```

- [ ] **Remove NotImplementedError for async**
  ```python
  # Either implement basic async or remove the method
  # Don't leave landmines
  ```

#### Afternoon: Scaffold Critical Missing Pieces (4 hours)
- [ ] **Create minimal IR package**
  ```python
  # src/ember/ir/__init__.py
  # src/ember/ir/ops.py - OpType, Operation
  # src/ember/ir/graph.py - Graph class
  # src/ember/ir/optimizer.py - stub optimize()
  ```

- [ ] **Export validation decorator**
  ```python
  # operators_v2/__init__.py
  from .validate import validate
  __all__ = [..., "validate"]
  ```

- [ ] **Implement DatasetMetadata**
  ```python
  # src/ember/data/metadata.py
  @dataclass
  class DatasetMetadata:
      # The 8 fields from philosophy doc
  ```

### Day 0.5: CI/CD Infrastructure (Critical)

#### Morning: Automated Quality Gates (4 hours)
- [ ] **Set up pre-commit hooks**
  ```yaml
  # .pre-commit-config.yaml
  - repo: mypy
    hooks:
      - id: mypy
        args: [--strict]
  - repo: local
    hooks:
      - id: loc-check
        name: LOC Check
        entry: ./scripts/check_module_size.sh
  ```

- [ ] **Create LOC checker script**
  ```bash
  #!/bin/bash
  # scripts/check_module_size.sh
  # Fail if any module > 1000 lines
  ```

- [ ] **Add contract tests**
  ```python
  # tests/test_public_api_contract.py
  def test_models_api_surface():
      api = dir(ember.api.models)
      assert "get_registry" not in api  # No leakage
      assert len(api) < 10  # Minimal surface
  ```

#### Afternoon: Documentation Reality Check (4 hours)
- [ ] **Update QUICK_REFERENCE.md**
  - Fix examples to match actual API
  - Add "Where to look" section
  - Highlight the 4 questions in a box

- [ ] **Create DEPRECATION.md**
  - List everything we're removing
  - Show migration path for each
  - Set deletion date

## Revised Week 1 Plan

### Now We Can Actually Start

With the foundation fixes done, Week 1 can focus on true simplification rather than fighting existing complexity.

#### Key Changes to Original Plan:
1. **Day 1 includes**: Verify registry is actually hidden
2. **Day 2 includes**: Remove legacy code identified in Day 0
3. **Day 3 includes**: Use the IR system we scaffolded
4. **Day 5 includes**: Run the contract tests we created

## New Success Metrics

### Immediate (Day 0-0.5)
- [ ] No public registry access
- [ ] No duplicate operator APIs
- [ ] IR package exists (even if minimal)
- [ ] CI enforces our principles

### Week 1
- [ ] Models API surface < 10 public methods
- [ ] All async methods work or don't exist
- [ ] Legacy operators.py deleted

## The Masters' Perspective on This Reality Check

**Carmack**: "You found broken windows. Fix them before adding features."

**Ritchie**: "Delete the duplicates. Now."

**Dean & Ghemawat**: "Measure module size in CI or it won't stay small."

**Martin**: "Legacy code is a broken window. Every day it stays, quality degrades."

**Brockman**: "Users see get_registry() and think they should use it. Hide it."

**Jobs**: "Ship nothing until the old mess is gone."

**Knuth**: "Premature deprecation is the root of all confusion. Do it now."

## Action Items for Right Now

1. **Stop all feature work**
2. **Execute Day 0 and Day 0.5**
3. **Only then begin Week 1**

The review revealed we were planning to build on quicksand. These actions create solid ground.

## Tracking

Create a GitHub issue for each checkbox above. Tag them with `tech-debt` and `must-have`. Nothing else ships until these are green.

---

*"First make it work, then make it right, then make it fast." - Kent Beck*

*We're still at "make it work" - let's fix that.*