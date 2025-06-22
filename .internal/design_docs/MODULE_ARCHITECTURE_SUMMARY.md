# Ember Module Architecture (Simplified)

## Core Design Principles

Following Dean/Ghemawat/Martin/Jobs:
1. **Simple things are simple**: `@module` decorator and you're done
2. **No hidden behavior**: What you see is what happens
3. **Performance by default**: Zero overhead for common cases
4. **One way to do things**: Clear, opinionated design

## Architecture Overview

```
ember.core.module                 # Main module system
├── @module decorator            # Makes classes immutable & transformable
├── static_field()              # Mark fields as non-transformable
├── chain()                     # Sequential composition
└── ensemble()                  # Parallel composition

ember.core.operators_v2         # Operator protocols
└── Operator[T, S] Protocol    # Any callable T -> S

ember.api                       # Public API
├── operators                   # Legacy operator system
└── operators_v2               # New protocol-based system
```

## Usage Examples

### 1. Simple Module
```python
from ember.core.module import module

@module
class MultiplyAdd:
    multiply: float
    add: float = 0.0
    
    def __call__(self, x: float) -> float:
        return x * self.multiply + self.add

# Usage
op = MultiplyAdd(multiply=2.0, add=1.0)
result = op(5)  # 11.0
```

### 2. Composition
```python
from ember.core.module import chain, ensemble

# Sequential
pipeline = chain(
    MultiplyAdd(2.0, 1.0),   # 2x + 1
    MultiplyAdd(3.0, -2.0),  # 3(2x + 1) - 2
)

# Parallel
parallel = ensemble(
    lambda x: x,      # Identity
    lambda x: x * 2,  # Double
    lambda x: x ** 2  # Square
)

results = parallel(5)  # [5, 10, 25]
```

### 3. With Transformations
```python
from ember.xcs import jit, vmap

# JIT compile for speed
fast_pipeline = jit(pipeline)

# Vectorize for batches
batch_op = vmap(MultiplyAdd(2.0, 1.0))
results = batch_op([1, 2, 3, 4, 5])
```

## What's Different

### Old Way (module_v2/v3/v4)
- Inheritance from EmberModule base class
- Complex metaclass magic
- Automatic tracing and metadata
- Hidden behavior
- Performance overhead

### New Way (module)
- Simple decorator
- Explicit behavior
- No hidden costs
- Composable functions
- Clean separation of concerns

## Migration

1. **Deprecation warnings** guide you
2. **Simple changes**: Replace inheritance with decorator
3. **Better performance**: No tracing overhead
4. **Cleaner code**: Less boilerplate

## Benefits

- **67% less code** than module_v4
- **Zero overhead** for common cases
- **Standard Python** debugging
- **Clear semantics** - no surprises
- **Composable** - functions all the way down

## Philosophy

> "Perfection is achieved not when there is nothing more to add,
> but when there is nothing left to take away."
> - Antoine de Saint-Exupéry

The new module system embodies this principle - it does exactly what's
needed for immutable, transformable operators, and nothing more.