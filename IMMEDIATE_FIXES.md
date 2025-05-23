# Immediate Fixes for Examples

## Priority 1: Fix Test Infrastructure

### Fix mock_lm fixture
The mock_lm fixture in `tests/golden/conftest.py` has an issue:

```python
# Current (broken)
lm.__call__.side_effect = mock_invoke  # AttributeError

# Fixed
lm = MagicMock(side_effect=mock_invoke)
```

### Fix models API mock
Update mocks to match new simplified API:

```python
# Mock for new models() function
mock_models = MagicMock()
mock_models.side_effect = lambda model, prompt, **kwargs: Response("Mocked response")
```

## Priority 2: Update Critical Examples

### 1. models/list_models.py
```python
# OLD
from ember.api.models import initialize_registry
registry = initialize_registry()
models = registry.list_models()

# NEW
from ember.api import models
available_models = models.list()
```

### 2. models/model_registry_direct.py
```python
# OLD
from ember.core.registry.model.base.registry import ModelRegistry
from ember.core.registry.model.base.services import ModelService

# NEW
from ember.api.models import get_registry, get_model_service
# Or better: just use models() directly
```

### 3. operators/custom_prompt_example_caravan.py
```python
# OLD
from ember.core.registry.model.model_module import lm

# NEW
from ember.api import models
# Or if it needs the lm pattern:
from ember.api.non import lm
```

### 4. advanced/test_auto_discovery.py
```python
# OLD
registry = initialize_registry(auto_discover=True)

# NEW
from ember.api.models import get_default_context
context = get_default_context()
registry = context.registry
```

## Priority 3: Fix Common Patterns

### Import Updates Required

Replace these patterns everywhere:

| Old Import | New Import |
|------------|------------|
| `from ember.core.registry.model import ...` | `from ember.api.models import ...` |
| `from ember.core.registry.operator import ...` | `from ember.api.operators import ...` |
| `from ember.core.utils.data import ...` | `from ember.api.data import ...` |
| `from ember.xcs import ...` | `from ember.api.xcs import ...` |

### Documentation Updates

Replace in all docstrings:
- `poetry run` → `uv run`
- Update example outputs to match new API

## Testing Strategy

1. **Fix one category at a time**
2. **Run golden tests after each fix**
3. **Commit working examples immediately**

## Quick Test Commands

```bash
# Test single category
uv run pytest tests/golden/test_models_examples.py -v

# Test single example
uv run python src/ember/examples/models/model_api_example.py

# Run all golden tests
uv run python tests/golden/run_golden_tests.py
```

## Branch Strategy Decision

**Recommendation**: Create a new branch that merges both:
```bash
git checkout -b examples-update-combined
git merge origin/models-deep-review-phase1
# Fix conflicts favoring the simplified API
# Then apply our golden test improvements
```

This ensures we have:
1. Latest simplified APIs
2. Golden test framework
3. Clean history