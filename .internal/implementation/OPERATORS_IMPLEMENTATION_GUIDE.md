# Operators Implementation Guide

## Overview

This document explains the new operator system implementation, where everything lives, and the rationale behind each decision.

## Directory Structure

```
src/ember/
├── api/
│   ├── operators.py          # Public API (was operators_v2.py)
│   └── operator.py           # Backward compatibility (singular)
│
└── core/
    └── operators/            # Core implementation (was operators_v2/)
        ├── __init__.py       # Module exports
        ├── protocols.py      # Protocol definitions
        ├── validate.py       # @validate decorator
        ├── composition.py    # chain, parallel, ensemble
        ├── capabilities.py   # add_batching, add_cost_tracking
        └── specification.py  # EmberModel/Specification support
```

## What We Created and Why

### 1. **protocols.py** - Clean Contracts Without Inheritance
```python
@runtime_checkable
class Operator(Protocol[T, S]):
    """Something that transforms T to S. That's all."""
    def __call__(self, input: T) -> S: ...
```

**Why**: 
- Protocols over base classes (no forced inheritance)
- Any callable is an operator
- Type-safe without complexity
- What Dean & Ghemawat would do: minimal interface

### 2. **validate.py** - Progressive Type Validation
```python
@validate(input=str, output=int)
def count_words(text: str) -> int:
    return len(text.split())
```

**Why**:
- Optional runtime validation (YAGNI principle)
- Clean decorator syntax
- Only validates when you ask it to
- No Pydantic models required for simple types

### 3. **composition.py** - Natural Function Composition
```python
pipeline = chain(extract, transform, load)
results = parallel(analyzer1, analyzer2, analyzer3)
consensus = ensemble(model1, model2, model3, reducer=majority_vote)
```

**Why**:
- Composition is just function composition
- No complex operator classes needed
- Reads like natural Python
- What Ritchie would appreciate: simplicity

### 4. **capabilities.py** - Add Features When Needed
```python
# Start simple
def process(x): return x * 2

# Add capabilities progressively
batch_op = add_batching(process, batch_size=32)
cost_op = add_cost_tracking(batch_op, cost_per_call=0.02)
```

**Why**:
- Progressive enhancement pattern
- Don't pay for what you don't use
- Clean adapter pattern
- What Carmack would do: performance when needed

### 5. **specification.py** - Full Power When Required
```python
@with_specification(ComplexSpec)
def analyze(inputs: ComplexInput) -> ComplexOutput:
    # Full EmberModel validation and structure
```

**Why**:
- Backward compatibility preserved
- Complex cases still supported
- Progressive disclosure: simple default, complex possible
- What Jobs would want: power without complexity

## Migration Path

### Old System (Removed)
- `src/ember/core/registry/operator/` → `.internal_docs/backup/old_operators_v1/`
- Complex `Operator` base class with forced inheritance
- Required EmberModule, Specification, Input/Output models
- ~180 lines for minimal operator

### New System (Current)
- `src/ember/core/operators/` - Clean, simple implementation
- Functions are operators
- Progressive complexity
- 3 lines for minimal operator

## Key Design Decisions

### 1. **Functions First**
```python
# This is a valid operator
def add(a, b): return a + b
```
- No base class required
- No boilerplate
- Just write Python

### 2. **Progressive Disclosure**
- Level 1: Functions (90%) - No imports needed
- Level 2: @validate (9%) - Simple type checking
- Level 3: Specification (1%) - Full structure when needed

### 3. **Explicit Over Magic**
- No metaclasses
- No hidden behavior
- Clear function names
- What you see is what you get

### 4. **Composition Over Inheritance**
- Use `chain()` not complex class hierarchies
- Use `parallel()` not thread management
- Use `ensemble()` not voting operators

### 5. **Protocols Over Base Classes**
- Define behavior, not structure
- Duck typing with type safety
- No forced inheritance
- Maximum flexibility

## Examples

### Before (Old System)
```python
class MyOperator(Operator[InputModel, OutputModel]):
    specification = Specification(...)
    
    def forward(self, *, inputs: InputModel) -> OutputModel:
        # 20+ lines of boilerplate above this
        return process(inputs)
```

### After (New System)
```python
def my_operator(inputs):
    return process(inputs)
```

## Testing

All operator tests that tested the old system have been moved to:
`.internal_docs/backup/old_operators_v1/`

New tests should:
- Test functions directly
- Use protocols for type checking
- Verify composition works
- Check progressive enhancement

## Future Considerations

1. **Async Support**: Can add `async def` operators naturally
2. **Streaming**: Can return generators/iterators
3. **Error Handling**: Can use standard Python try/except
4. **Metrics**: Can add via capabilities without changing operators

## Summary

The new operator system embodies the principles from CLAUDE.md and what our mentors (Dean, Ghemawat, Jobs, Brockman, Ritchie, Knuth, Carmack) would appreciate:

- **Radical simplicity**: 97% less code for common cases
- **No magic**: Explicit, clear behavior
- **Progressive disclosure**: Complexity only when needed
- **Natural Python**: Works like developers expect
- **Powerful when needed**: Full EmberModel support remains

This is a principled, root-node fix that eliminates accidental complexity while preserving essential complexity.