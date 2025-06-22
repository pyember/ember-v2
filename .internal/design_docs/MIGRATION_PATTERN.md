# Operator Migration Pattern

## Design Decision

After careful consideration, operators that use models will:
1. Accept both strings and ModelBinding instances
2. Handle their own normalization (no base class changes)
3. Optimize for the common case (just strings)

## Implementation Pattern

```python
from typing import List, Union
from ember.api import models, ModelBinding
from ember.core.registry.operator.base import Operator

class ModelUsingOperator(Operator):
    """Example operator that uses language models."""
    
    def __init__(
        self, 
        # Accept both for flexibility
        model_or_models: Union[str, ModelBinding, List[Union[str, ModelBinding]]],
        # Common parameters applied to string models
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Normalize to list for uniform handling
        model_list = (
            [model_or_models] 
            if not isinstance(model_or_models, list) 
            else model_or_models
        )
        
        # Convert strings to bindings with operator-level params
        self.models = [
            models.bind(m, temperature=temperature, max_tokens=max_tokens) 
            if isinstance(m, str) else m
            for m in model_list
        ]
    
    def forward(self, inputs):
        # Use models
        responses = []
        for model in self.models:
            response = model(inputs.prompt)
            responses.append(response.text)
        return {"responses": responses}
```

## Usage Examples

### Simple Case (90% of usage)
```python
# Single model with defaults
verifier = VerifierOperator("gpt-4")

# Multiple models with shared config
ensemble = EnsembleOperator(
    ["gpt-4", "claude-3", "gpt-3.5-turbo"],
    temperature=0.7
)
```

### Advanced Case (10% of usage)
```python
# Custom configuration per model
ensemble = EnsembleOperator([
    models.bind("gpt-4", temperature=0.3),       # Analytical
    models.bind("claude-3", temperature=0.9),     # Creative
    models.bind("gpt-3.5-turbo", temperature=0.5) # Balanced
])
```

## Benefits

1. **Simple Common Case**: Just pass model names as strings
2. **Flexible When Needed**: Accept pre-configured models
3. **Explicit**: No magic, clear what's happening
4. **Operator Control**: Each operator manages its own models
5. **No Base Class Pollution**: Operator base stays pure

## Migration Checklist

For each operator:
- [ ] Change `lm_modules: List[LMModule]` to `models: List[Union[str, ModelBinding]]`
- [ ] Add model parameters to `__init__` (temperature, max_tokens, etc.)
- [ ] Add normalization logic in `__init__`
- [ ] Update `forward` to use `response.text` instead of raw string
- [ ] Update tests to use new pattern
- [ ] Update documentation