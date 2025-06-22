# LMModule to ModelBinding Migration Guide

## Overview

We're simplifying how models are used in Ember operators. The `LMModule` and `LMModuleConfig` classes are being replaced with the simpler and more powerful `ModelBinding` pattern from the models API.

## Why This Change?

1. **Simpler**: One clear way to use models instead of multiple patterns
2. **Faster**: Removes unnecessary abstraction layers
3. **Cleaner**: Less code, fewer bugs, easier to understand

## Quick Migration Examples

### Basic Model Usage

**Before (LMModule)**:
```python
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig

# Create model
lm_module = LMModule(config=LMModuleConfig(
    id="gpt-4",
    temperature=0.7,
    max_tokens=1000
))

# Use model
response = lm_module(prompt="What is the capital of France?")
print(response)  # "Paris"
```

**After (ModelBinding)**:
```python
from ember.api import models

# Create model binding
model = models.bind("gpt-4", temperature=0.7, max_tokens=1000)

# Use model
response = model("What is the capital of France?")
print(response.text)  # "Paris"
```

### Using Models in Operators

**Before**:
```python
from ember.core.registry.operator.core.ensemble import EnsembleOperator

# Create operator with LMModules
lm_modules = [
    LMModule(config=LMModuleConfig(id="gpt-4")),
    LMModule(config=LMModuleConfig(id="claude-3")),
    LMModule(config=LMModuleConfig(id="gpt-3.5-turbo"))
]

operator = EnsembleOperator(lm_modules=lm_modules)
result = operator.forward("Explain quantum computing")
```

**After**:
```python
from ember.core.registry.operator.core.ensemble import EnsembleOperator

# Create operator with model names or bindings
operator = EnsembleOperator(
    models=["gpt-4", "claude-3", "gpt-3.5-turbo"],
    temperature=0.7  # Applied to all models
)

result = operator.forward("Explain quantum computing")
```

### Advanced Model Configuration

**Before**:
```python
# Different settings per model
configs = [
    LMModuleConfig(id="gpt-4", temperature=0.5, max_tokens=1000),
    LMModuleConfig(id="claude-3", temperature=0.9, max_tokens=2000),
]
lm_modules = [LMModule(config=cfg) for cfg in configs]
```

**After**:
```python
# Different settings per model
models = [
    models.bind("gpt-4", temperature=0.5, max_tokens=1000),
    models.bind("claude-3", temperature=0.9, max_tokens=2000),
]
```

## Operator-Specific Migration

### EnsembleOperator

**Before**:
```python
operator = EnsembleOperator(
    lm_modules=[
        LMModule(config=LMModuleConfig(id="gpt-4")),
        LMModule(config=LMModuleConfig(id="claude-3"))
    ],
    aggregation_method="most_common"
)
```

**After**:
```python
operator = EnsembleOperator(
    models=["gpt-4", "claude-3"],
    aggregation_method="most_common",
    temperature=0.7  # Optional: applies to all models
)
```

### VerifierOperator

**Before**:
```python
verifier = VerifierOperator(
    lm_module=LMModule(config=LMModuleConfig(
        id="gpt-4",
        temperature=0.3
    ))
)
```

**After**:
```python
verifier = VerifierOperator(
    model="gpt-4",
    temperature=0.3
)
```

## Key Differences

### 1. Response Access

**Before**:
```python
response = lm_module(prompt="Hello")
# response is a string
```

**After**:
```python
response = model("Hello")
# response is a Response object
text = response.text  # Access the text
tokens = response.usage['total_tokens']  # Access usage info
```

### 2. Error Handling

**Before**:
```python
try:
    response = lm_module(prompt="Hello")
except Exception as e:
    # Generic exception handling
    print(f"Error: {e}")
```

**After**:
```python
from ember.api.models import ModelError, RateLimitError

try:
    response = model("Hello")
except RateLimitError:
    # Handle rate limits specifically
    time.sleep(60)
except ModelError as e:
    # Handle other model errors
    print(f"Model error: {e}")
```

### 3. Model Service Access

**Before**:
```python
# LMModule creates its own model service
lm_module = LMModule(config=config, model_service=custom_service)
```

**After**:
```python
# Models API uses shared model service
# For custom services, use ModelContext
from ember.core.registry.model.base.context import ModelContext

context = ModelContext(model_service=custom_service)
model = context.models.bind("gpt-4")
```

## Testing Your Code

### Update Test Mocks

**Before**:
```python
mock_lm = MagicMock()
mock_lm.return_value = "test response"
```

**After**:
```python
from unittest.mock import MagicMock
from ember.api import Response

mock_model = MagicMock()
mock_response = MagicMock(spec=Response)
mock_response.text = "test response"
mock_model.return_value = mock_response
```

### Integration Tests

```python
# Test with real models (be mindful of API costs)
def test_my_operator():
    operator = MyOperator(model="gpt-3.5-turbo")
    result = operator.forward("Test prompt")
    assert result is not None
```

## Gradual Migration

During the transition period, you can use both patterns:

```python
# Temporary compatibility
from ember.core.registry.model.model_module.lm import LMModule

# This will show a deprecation warning but still work
lm_module = LMModule(config=config)

# Start migrating to this
model = models.bind(config.id, temperature=config.temperature)
```

## Common Patterns

### Pattern 1: Multiple Models with Same Config

```python
# Create a base configuration
base_model = lambda model_id: models.bind(model_id, temperature=0.7, max_tokens=1000)

models_list = [
    base_model("gpt-4"),
    base_model("claude-3"),
    base_model("gpt-3.5-turbo")
]
```

### Pattern 2: Dynamic Model Selection

```python
def get_model_for_task(task_type: str) -> ModelBinding:
    model_map = {
        "creative": models.bind("gpt-4", temperature=0.9),
        "analytical": models.bind("claude-3", temperature=0.3),
        "fast": models.bind("gpt-3.5-turbo", temperature=0.5)
    }
    return model_map.get(task_type, model_map["fast"])
```

### Pattern 3: Model with Retry Logic

```python
from ember.api.models import RateLimitError
import time

def invoke_with_retry(model: ModelBinding, prompt: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return model(prompt)
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

## Timeline

- **Week 1**: Deprecation warnings added to LMModule
- **Week 2-3**: Examples and documentation updated
- **Week 4**: LMModule removed from codebase

## Getting Help

If you encounter issues during migration:

1. Check this guide for examples
2. Look at updated operator examples in `examples/operators/`
3. Review the models API documentation
4. Report issues in the GitHub repository

## Summary

The migration from LMModule to ModelBinding is straightforward:

1. Replace `LMModule(config=LMModuleConfig(...))` with `models.bind(...)`
2. Access response text with `.text` property
3. Use specific error types for better error handling
4. Enjoy simpler, faster code!

The new pattern is cleaner, more intuitive, and aligns with the principle of having "one obvious way to do things."