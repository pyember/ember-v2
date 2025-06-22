# Consolidated Review Insights & Actions

*Merging the paired review findings with our plans*

## Critical Gaps Found

The review revealed we were planning to build on quicksand. These must be fixed first:

1. **ModelsAPI.get_registry()** - Still public, violates hiding principle
2. **No IR implementation** - Only decorators exist, blocking XCS work  
3. **operators.py exports legacy API** - Confuses users
4. **No CI enforcement** - Modules will grow without limits
5. **Async methods broken** - raise NotImplementedError
6. **validate decorator missing** - Mentioned in docs but not exported
7. **DatasetMetadata not implemented** - Still using heavy DatasetInfo
8. **No contract tests** - API leakage will grow

## Day 0 Consolidated Actions (8 hours)

### Morning: Fix Public API (4 hours)
```python
# 1. Hide registry (1h)
class ModelsAPI:
    def _get_registry(self):  # was: get_registry()
        ...

# 2. Deprecate legacy operators (2h)  
# operators.py
"""DEPRECATED: Use ember.api.operators_v2"""
import warnings
warnings.warn("operators.py is deprecated", DeprecationWarning)
from ember.api.operators_v2 import *

# 3. Fix async methods (1h)
# Either implement or remove entirely
```

### Afternoon: Scaffold Foundations (4 hours)
```python
# 4. Create IR package (2h)
# src/ember/ir/ops.py
from enum import Enum
from dataclasses import dataclass

class OpType(Enum):
    LLM_CALL = "llm_call"
    ENSEMBLE = "ensemble"
    # ... from design

@dataclass
class Operation:
    op_type: OpType
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]

# 5. Export validate (1h)
# operators_v2/__init__.py
from .validate import validate
__all__ = [..., "validate"]

# 6. Implement DatasetMetadata (1h)
# As designed in philosophy doc
```

## Day 0.5 Infrastructure (8 hours)

### Morning: CI/CD Gates (4 hours)
```bash
# scripts/check_module_size.py
import os
import sys

def check_module_sizes(max_lines=1000):
    violations = []
    for root, dirs, files in os.walk("src/ember"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path) as f:
                    lines = len(f.readlines())
                if lines > max_lines:
                    violations.append((path, lines))
    
    if violations:
        print("Module size violations:")
        for path, lines in violations:
            print(f"  {path}: {lines} lines")
        sys.exit(1)

# Add to .github/workflows/ci.yml
```

### Contract Tests
```python
# tests/test_public_api_contract.py
def test_models_api_minimal():
    api = dir(ember.api.models)
    assert len(api) < 10  # Minimal surface
    assert "get_registry" not in api  # No leakage

def test_operators_deprecation():
    with pytest.warns(DeprecationWarning):
        import ember.api.operators
```

## Updated Week 1 Plan

Now that foundations are solid:

### Day 1 Adjustments
- Verify registry actually hidden ✓
- Remove any legacy code found ✓
- Focus on true simplification

### Day 2 Adjustments  
- Use the DatasetMetadata we created ✓
- Remove DataItem complexity ✓
- Delete legacy imports

### Day 3 Adjustments
- Use the IR system we scaffolded ✓
- Delete operators.py after deprecation ✓
- Focus on natural API

## Success Criteria

### Immediate (Day 0-0.5)
- [ ] `dir(ember.api)` shows < 20 items
- [ ] No NotImplementedError in public API
- [ ] IR package exists and imports work
- [ ] CI fails on large modules

### Week 1
- [ ] Models API surface < 10 methods
- [ ] operators.py deleted
- [ ] All async methods work
- [ ] Contract tests green

## The Masters on This Approach

**Carmack**: "Fix the broken windows first. Everything else is built on sand."

**Ritchie**: "Delete the duplicates immediately. Every day they exist adds confusion."

**Dean & Ghemawat**: "Measure in CI or it won't happen."

**Martin**: "Technical debt compounds. Pay it down before adding features."

## Next Steps

1. Execute Day 0 (all 8 checkboxes)
2. Execute Day 0.5 (all 8 checkboxes)  
3. Only then begin Week 1 Models work

The review showed our plans were good but assumed a cleaner starting point than reality. These actions create that clean foundation.