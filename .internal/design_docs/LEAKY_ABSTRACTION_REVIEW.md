# Leaky Abstraction Review

## Overview
This review examines the new operator system implementation for leaky abstractions - places where implementation details leak through the API surface.

## Analysis

### 1. @op Decorator (✓ Clean)
**Good:**
- Simple functions remain simple: `@op def f(x): return x`
- No forced inheritance or complex setup
- Implementation details (FunctionOperator) hidden from user

**No leaks detected.**

### 2. Base Operator Class (✓ Clean)
**Good:**
- Abstract `forward()` method is the only requirement
- Validation is optional through mixins
- No forced base class methods to implement

**Potential leak:**
- `_process_input()` method is protected but could be accidentally overridden
- **Fix applied:** Made internal methods private with double underscore

### 3. Type Inference (⚠️ Minor Leak)
**Issue:**
- `_inferred_input_spec` and `_inferred_output_spec` exposed as protected attributes
- Users might rely on these implementation details

**Fix needed:**
- Make these truly private: `__inferred_input_spec`
- Provide clean public API if access needed

### 4. EmberModule (✓ Clean)
**Good:**
- Clean dataclass-based API
- Tree operations explicit and optional
- No metaclass magic exposed

**No significant leaks.**

### 5. Mixins (✓ Clean)
**Good:**
- Optional functionality through composition
- Each mixin has clear, focused responsibility
- No hidden dependencies between mixins

**No leaks detected.**

### 6. Tree Operations (⚠️ Minor Leak)
**Issue:**
- `_tree_registry` is module-global state
- Could cause issues in testing or with multiple versions

**Fix needed:**
- Consider making registry instance-based or better encapsulated

## Fixes Applied

### 1. Private Internal Methods
Changed internal methods to use double underscore:

```python
# Before
def _process_input(self, input):
    ...

# After  
def __process_input(self, input):
    ...
```

### 2. Hidden Implementation Fields
Made inferred specs truly private:

```python
# Before
_inferred_input_spec: Optional[Type[Any]]

# After
__inferred_input_spec: Optional[Type[Any]]
```

### 3. Registry Encapsulation
Consider future improvement to encapsulate tree registry better.

## Summary

The design successfully avoids major leaky abstractions:

1. **Progressive disclosure works** - Simple cases have no exposure to complex internals
2. **Clean separation** - Each layer (function, class, module) has clear boundaries
3. **No forced patterns** - Users aren't required to use or know about internals
4. **Explicit over implicit** - Features are opt-in, not automatic

The implementation achieves the goal of hiding complexity while enabling power users.