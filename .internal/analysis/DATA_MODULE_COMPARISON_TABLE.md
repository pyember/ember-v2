# Data Module: Old vs New Comparison

## Overview Comparison

| Aspect | Old Implementation | New Implementation |
|--------|-------------------|-------------------|
| **Lines of Code** | ~688 lines | ~1215 lines (with comprehensive docs) |
| **Number of Classes** | 5 classes + complex hierarchy | 3 concrete classes + 1 protocol |
| **API Levels** | 3 levels (confusing) | 2 functions + optional chaining |
| **Memory Model** | Unclear (streaming optional) | Streaming by default, explicit loading |
| **Documentation** | Minimal, philosophical | Comprehensive, technical, practical |

## API Design Comparison

| Feature | Old API | New API |
|---------|---------|---------|
| **Basic Usage** | `data("mmlu")` | `stream("mmlu")` |
| **Eager Loading** | `data("mmlu", streaming=False)` | `load("mmlu")` |
| **Return Type** | `Union[StreamingView, MaterializedDataset]` | `StreamIterator` or `List[Dict]` |
| **Initialization** | `context = EmberContext.current()`<br>`data_api = DataAPI(context)` | No initialization needed |
| **Function Names** | Ambiguous `data()` | Clear `stream()` and `load()` |

## Code Complexity Comparison

| Aspect | Old | New |
|--------|-----|-----|
| **Simple Dataset Loading** | 3 lines with context | 1 line |
| **Filtering** | `.filter(subject="physics")` | `filter=lambda x: x["metadata"]["subject"] == "physics"` |
| **Chaining** | Always available (confusing) | Progressive - only when `.filter()` called |
| **Builder Pattern** | Complex 3-level API | Simple parameters or chaining |
| **Magic Behavior** | `item.question` via `__getattr__` | `item["question"]` explicit dict |

## Feature Comparison

| Feature | Old | New | Improvement |
|---------|-----|-----|-------------|
| **Subset/Split Selection** | ✓ Builder only | ✓ Direct parameters | Simpler |
| **Transformations** | ✓ Complex | ✓ Lambda functions | Clearer |
| **Streaming Default** | ✗ Optional | ✓ Default | Better performance |
| **Metadata Access** | ✓ Complex | ✓ Simple function | Easier |
| **Custom Sources** | ✓ Complex registration | ✓ Simple protocol | More flexible |
| **Zero Config** | ✗ Needs context | ✓ Just works | Much simpler |
| **Type Safety** | Partial | ✓ Full annotations | Better IDE support |
| **Thread Safety** | Unclear | ✓ Documented | Predictable |

## Documentation Comparison

| Aspect | Old | New |
|--------|-----|-----|
| **Module Docstring** | Philosophical quotes | Technical overview with examples |
| **Function Docs** | Basic | Full Args/Returns/Raises/Examples |
| **Examples** | ~5 basic | 50+ comprehensive |
| **Error Docs** | Missing | Complete with all exceptions |
| **Type Hints** | Partial | Complete with all generics |
| **Style** | Casual, references to "masters" | Professional, technical |

## Usage Examples Comparison

### Loading a Dataset

**Old:**
```python
from ember.api import data
context = EmberContext.current()
data_api = DataAPI(context)
dataset = data_api.builder().from_registry("mmlu").build()
for item in dataset:
    print(item.query)  # Magic attribute
```

**New:**
```python
from ember.api import stream
for item in stream("mmlu"):
    print(item["question"])  # Explicit dict
```

### Loading with Subset

**Old:**
```python
dataset = data().builder()
    .from_registry("mmlu")
    .subset("physics")
    .split("test")
    .build()
```

**New:**
```python
for item in stream("mmlu", subset="physics", split="test"):
    process(item)
```

### Filtering Data

**Old:**
```python
# Magic filtering
items = data("mmlu").filter(subject="physics")
```

**New:**
```python
# Explicit lambda
items = stream("mmlu", filter=lambda x: x["metadata"]["subject"] == "physics")
```

### Complex Pipeline

**Old:**
```python
result = data("mmlu")
    .filter(subject="physics")
    .transform(add_prompt)
    .limit(100)
    .collect()  # When do I need this?
```

**New:**
```python
# Option 1: Parameters (simple)
result = load("mmlu",
    filter=lambda x: x["metadata"]["subject"] == "physics",
    transform=lambda x: {**x, "prompt": format_prompt(x)},
    max_items=100
)

# Option 2: Chaining (advanced)
result = (stream("mmlu")
    .filter(lambda x: x["metadata"]["subject"] == "physics")
    .transform(lambda x: {**x, "prompt": format_prompt(x)})
    .limit(100)
    .collect())
```

## Performance Comparison

| Metric | Old | New |
|--------|-----|-----|
| **Memory Usage** | Varies (unclear default) | O(1) streaming default |
| **Startup Time** | Context initialization | None |
| **Type Checking** | Slow (Union types) | Fast (clear types) |
| **Error Messages** | Generic | Specific with suggestions |

## Migration Effort

| Task | Difficulty | Notes |
|------|------------|-------|
| **Update Imports** | Easy | `data` → `stream`/`load` |
| **Remove Context** | Easy | Delete 2-3 lines |
| **Fix Attribute Access** | Medium | `.question` → `["question"]` |
| **Update Filters** | Medium | Keyword → lambda |
| **Update Builders** | Easy | Builder → parameters |

## Key Improvements Summary

1. **Clarity**: `stream()` vs `load()` makes intent explicit
2. **Simplicity**: No context, no builders for common cases  
3. **Performance**: Streaming by default saves memory
4. **Safety**: No magic attributes, full type annotations
5. **Documentation**: 10x more examples and details
6. **Flexibility**: Same features, simpler API

## Developer Experience

| Aspect | Old | New | Impact |
|--------|-----|-----|---------|
| **Learning Curve** | Steep (3 levels) | Gentle (2 functions) | Faster onboarding |
| **Debugging** | Hard (magic behavior) | Easy (plain dicts) | Less confusion |
| **IDE Support** | Limited | Full autocomplete | Better productivity |
| **Error Discovery** | Runtime | Type checking | Earlier bug detection |
| **API Discovery** | Confusing | Obvious | Better DX |

The new implementation achieves the same functionality with significantly less complexity and better developer experience.