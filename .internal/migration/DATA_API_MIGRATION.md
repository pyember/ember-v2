# Data API Migration Guide

## Clean Break - No Backward Compatibility

Following the principles of Dean, Ghemawat, Jobs, Brockman, Ritchie, Knuth, and Carmack, we've made a clean, decisive break with the old API.

## What Changed

### Old API (Complex)
```python
# Required context management
context = EmberContext.current()
data_api = DataAPI(context)

# Complex builder pattern
dataset = (
    data_api.builder()
    .from_registry("mmlu")
    .subset("physics")
    .split("test")
    .sample(100)
    .build()
)

# Complex return types
for entry in dataset:
    print(entry.query)  # DatasetEntry object
```

### New API (Simple)
```python
# Zero config
from ember.api import data

# Direct usage - 80% of cases
for item in data("mmlu"):
    print(item.question)  # Clean DataItem with normalized access

# Chaining - 15% of cases
physics = data("mmlu").filter(subject="physics").limit(100)

# Builder for complex cases - 5% of cases  
dataset = data().builder()
    .from_registry("mmlu")
    .subset("high_school_physics")
    .split("test")
    .sample(1000, seed=42)
    .build()
```

## Migration Steps

1. **Change imports**:
   ```python
   # Old
   from ember.api import DataAPI, DatasetBuilder
   
   # New
   from ember.api import data
   ```

2. **Remove context management**:
   ```python
   # Old
   context = EmberContext.current()
   data_api = DataAPI(context)
   
   # New - nothing needed
   ```

3. **Update simple loading**:
   ```python
   # Old
   dataset = data_api.builder().from_registry("mmlu").build()
   
   # New
   dataset = data("mmlu")
   ```

4. **Update filtering**:
   ```python
   # Old - custom transformer classes
   
   # New - simple lambdas
   filtered = data("mmlu").filter(lambda x: x["score"] > 0.5)
   ```

5. **Update data access**:
   ```python
   # Old
   entry.query  # Maybe exists, maybe not
   
   # New  
   item.question  # Normalized access
   item.options   # Always a dict
   item.answer    # Clear property
   ```

## Key Improvements

1. **Progressive Disclosure**
   - Simple: `data("dataset")`
   - Intermediate: `.filter().transform()`
   - Advanced: `.builder()` for full control

2. **Streaming by Default**
   - O(1) memory usage
   - Explicit materialization when needed

3. **Clean Abstractions**
   - DataItem with normalized access
   - No leaky DatasetEntry internals
   - Clear StreamingView vs MaterializedDataset

4. **No Hidden Complexity**
   - No context management
   - No interface classes
   - Direct function calls

## Removed Classes

These classes no longer exist:
- `Dataset` - Use `StreamingView` or `MaterializedDataset`
- Old `DatasetBuilder` - Now just `builder()` method
- Complex validation schemas

## New Classes

- `DataItem` - Clean normalized access to any dataset item
- `StreamingView` - Efficient chainable operations
- `MaterializedDataset` - When you need random access
- `DatasetBuilder` - Clean builder for complex scenarios

## Philosophy

As Carmack would say: "Just load the data and get out of the way."

No backward compatibility cruft. No migration helpers. Just a clean, simple API that does what you need.