# Data API Migration Guide - V3

## Overview

We've completely redesigned the data API following CLAUDE.md principles and the philosophies of Dean, Ghemawat, Jobs, Brockman, Ritchie, Knuth, Carmack, and Martin. The new API is simpler, more explicit, and more powerful.

## Key Changes

### 1. No More Magic Behavior
- ❌ **Removed**: `DataItem` class with `__getattr__` magic
- ✅ **Now**: Plain dictionaries with clear schema

### 2. Two Clear Functions
- ❌ **Removed**: Complex 3-level API with `data()` doing everything
- ✅ **Now**: `stream()` for streaming, `load()` for eager loading

### 3. Explicit Over Implicit
- ❌ **Removed**: Auto-normalization guessing
- ✅ **Now**: Explicit `normalize=True/False` parameter

### 4. Progressive Disclosure
- Simple parameters for 90% of cases
- Chainable methods for advanced usage (10%)
- No hidden complexity

## Migration Examples

### Basic Loading

**Old API**:
```python
from ember.api import data

# Confusing - is this streaming or loaded?
for item in data("mmlu"):
    print(item.question)  # Magic attribute
```

**New API**:
```python
from ember.api import stream

# Clear - this is streaming
for item in stream("mmlu"):
    print(item["question"])  # Explicit dict access
```

### Loading with Subset/Split

**Old API**:
```python
dataset = data().builder()
    .from_registry("mmlu")
    .subset("physics")
    .split("test")
    .build()
```

**New API**:
```python
# Simple parameters
for item in stream("mmlu", subset="physics", split="test"):
    process(item)
```

### Filtering and Transformation

**Old API**:
```python
# Complex chaining with special classes
items = data("mmlu")
    .filter(subject="physics")
    .transform(add_prompt)
    .collect()  # When do I need this?
```

**New API**:
```python
# Option 1: Inline parameters (simple)
items = load("mmlu", 
    filter=lambda x: x["metadata"].get("subject") == "physics",
    transform=lambda x: {**x, "prompt": f"Q: {x['question']}"}
)

# Option 2: Chaining (advanced)
items = stream("mmlu")
    .filter(lambda x: x["metadata"].get("subject") == "physics")
    .transform(lambda x: {**x, "prompt": f"Q: {x['question']}"})
    .collect()
```

### Eager Loading

**Old API**:
```python
# Unclear when this loads into memory
dataset = data("mmlu", streaming=False)
# or
dataset = data("mmlu").collect()
```

**New API**:
```python
# Explicit function name
items = load("mmlu")  # Clear: loads into memory
```

### Custom Data Sources

**Old API**:
```python
from ember.api import DataAPI, register_dataset
context = EmberContext.current()
data_api = DataAPI(context)
# Complex registration...
```

**New API**:
```python
from ember.api import register, FileSource

# Simple registration
register("my_data", FileSource("data.jsonl"))

# Or implement the protocol
class MySource:
    def read_batches(self, batch_size=32):
        yield [{"question": "Q1", "answer": "A1"}]

register("custom", MySource())
```

### File Loading

**Old API**:
```python
# Had to go through registry or complex builders
```

**New API**:
```python
from ember.api import from_file, load_file

# Stream from file
for item in from_file("data.jsonl"):
    process(item)

# Load file into memory
data = load_file("data.csv")
```

## API Reference

### Main Functions

```python
stream(source, *, subset=None, split=None, filter=None, 
       transform=None, batch_size=32, max_items=None, 
       normalize=True) -> StreamIterator
```
Stream data with optional filtering and transformation.

```python
load(source, *, subset=None, split=None, filter=None,
     transform=None, max_items=None, normalize=True) -> List[Dict]
```
Load data into memory (same parameters as stream).

```python
metadata(dataset: str) -> DatasetInfo
```
Get dataset metadata including size and example.

```python
list_datasets() -> List[str]
```
List available dataset names.

```python
register(name: str, source: DataSource, metadata=None) -> None
```
Register a custom data source.

### Normalized Schema

All items are normalized to this schema by default:
```python
{
    "question": str,  # The question/prompt/query
    "answer": str,    # The answer/target/label
    "choices": dict,  # Multiple choice options (if any)
    "metadata": dict  # All other fields
}
```

Set `normalize=False` to get raw data.

### Chaining Methods (Advanced)

The `StreamIterator` returned by `stream()` supports:
- `.filter(predicate)` - Add a filter
- `.transform(fn)` - Add a transformation  
- `.limit(n)` - Limit to n items
- `.first(n)` - Get first n as list
- `.collect()` - Collect all into list

## Complete Migration Checklist

1. **Update imports**:
   ```python
   # Old
   from ember.api import data, DataAPI
   
   # New
   from ember.api import stream, load
   ```

2. **Replace magic attribute access**:
   ```python
   # Old
   item.question
   
   # New
   item["question"]
   ```

3. **Use explicit functions**:
   ```python
   # Old
   data("mmlu")  # Ambiguous
   
   # New
   stream("mmlu")  # Streaming
   load("mmlu")    # In memory
   ```

4. **Update filtering**:
   ```python
   # Old
   .filter(subject="physics")
   
   # New
   filter=lambda x: x["metadata"].get("subject") == "physics"
   ```

5. **Remove context management**:
   ```python
   # Old
   context = EmberContext.current()
   data_api = DataAPI(context)
   
   # New - not needed!
   ```

## Benefits of New Design

1. **No UX Confusion**: Function names match behavior exactly
2. **No Magic**: Plain dicts, no `__getattr__` tricks
3. **Explicit**: Clear when streaming vs loading
4. **SOLID**: Extensible via DataSource protocol only
5. **Efficient**: Streaming by default, O(1) memory
6. **Simple**: 2 functions cover 90% of use cases

## Need Help?

The new API is designed to be self-explanatory. If you're unsure:
- `stream()` when you want to process data item by item
- `load()` when you need all data in memory
- Add parameters for filtering/transformation
- Chain methods only for complex pipelines