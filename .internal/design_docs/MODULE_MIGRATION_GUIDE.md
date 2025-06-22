# Module System Migration Guide

## Overview

The Ember module system has been simplified to follow Dean/Ghemawat/Martin/Jobs principles:
- Radical simplicity
- Explicit over implicit
- No hidden behavior
- Performance by default

## Migration Path

### From module_v2, module_v3, or module_v4

```python
# Old (v2/v3)
from ember.core.module_v2 import EmberModule, static_field, Chain, Ensemble

class MyModule(EmberModule):
    value: int
    config: dict = static_field(default_factory=dict)

# New
from ember.core.module import module, static_field, chain, ensemble

@module
class MyModule:
    value: int
    config: dict = static_field(default_factory=dict)
```

### Key Changes

1. **Decorator instead of inheritance**
   - Old: `class MyModule(EmberModule):`
   - New: `@module class MyModule:`

2. **Function composition instead of classes**
   - Old: `Chain(operators=(op1, op2))`
   - New: `chain(op1, op2)`
   - Old: `Ensemble(operators=(op1, op2))`
   - New: `ensemble(op1, op2)`

3. **No tracing by default**
   - Old: Automatic tracing with metadata
   - New: No tracing (moved to experimental if needed)

4. **Simpler imports**
   - Old: Various module versions with different features
   - New: Single `ember.core.module` with everything you need

## Complete Example

### Before (module_v4)
```python
from ember.core.module_v4 import EmberModule, Operator, static_field

class ChainOfThought(Operator[Dict, Dict]):
    model: Any
    temperature: float = 0.7
    config: dict = static_field(default_factory=dict)
    
    def forward(self, *, inputs: Dict) -> Dict:
        reasoning = self.model(inputs["question"])
        return {"answer": reasoning}
    
    def __post_init__(self):
        super().__post_init__()
        # Metadata initialization

# Usage with tracing
cot = ChainOfThought(model=model).with_metadata(
    id="cot1",
    stage="reasoning"
)

# Composition
ensemble = EnsembleOperator(
    operators=[cot1, cot2, cot3]
)
```

### After (new module)
```python
from ember.core.module import module, static_field, chain, ensemble

@module
class ChainOfThought:
    model: Any
    temperature: float = 0.7
    config: dict = static_field(default_factory=dict)
    
    def __call__(self, question: str) -> str:
        reasoning = self.model(question)
        return reasoning

# Usage - simple and direct
cot = ChainOfThought(model=model)

# Composition with functions
ensemble_cot = ensemble(
    ChainOfThought(model=model1),
    ChainOfThought(model=model2),
    ChainOfThought(model=model3)
)
```

## Benefits of Migration

1. **Simpler code**: 50-70% less boilerplate
2. **Better performance**: No tracing overhead
3. **Easier debugging**: Standard Python, no metaclass magic
4. **Clearer intent**: What you write is what happens

## Gradual Migration

You can migrate gradually:

1. Add deprecation warnings are already in place
2. Update imports one file at a time
3. Test thoroughly
4. Remove old imports

## Need Help?

- The old modules will remain available (with deprecation warnings) for several releases
- Tracing functionality has moved to `.internal_docs/experimental/tracing.py`
- For complex migrations, consider creating an adapter:

```python
# Temporary adapter for old-style operators
def adapt_old_operator(old_operator_class):
    @module
    class Adapted:
        # ... adapt fields and methods
    return Adapted
```

## Timeline

- **Current**: Deprecation warnings added
- **Next minor version**: Old modules moved to `ember.legacy`
- **Next major version**: Old modules removed

Plan your migration accordingly.