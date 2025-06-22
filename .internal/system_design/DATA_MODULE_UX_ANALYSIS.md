# UX Confusion Analysis: Data Module Design

## Common Sources of UX Confusion in Data APIs

### 1. **Multiple Ways to Do the Same Thing** ❌
**Current Design (3 levels)**:
```python
# Confusion: Which one should I use?
data("mmlu")                                    # Level 1
data("mmlu").filter(subject="physics")          # Level 2  
data().builder().from_registry("mmlu").build()  # Level 3
```

**Our Design**: ✅
```python
# Only one way to stream
stream("mmlu")

# Only one way to load eagerly
load("mmlu")
```

### 2. **Unclear Return Types** ❌
**Current Design**:
```python
# What type is returned? StreamingView? MaterializedDataset?
dataset = data("mmlu")  # Union[StreamingView, MaterializedDataset]

# Do I need to call collect()?
items = data("mmlu").collect()  # When is this needed?
```

**Our Design**: ✅
```python
# Clear: stream() returns Iterator
for item in stream("mmlu"):  # Iterator[Dict]
    pass

# Clear: load() returns List  
items = load("mmlu")  # List[Dict]
```

### 3. **Magic Behavior** ❌
**Current Design**:
```python
# Magic attribute access
item.question  # Where did this come from?
item.my_custom_field  # Will this work?

# Magic normalization
data("mmlu")  # Normalized? Not normalized?
```

**Our Design**: ✅
```python
# Explicit dictionary access
item["question"]  # Clear: it's a dict

# Explicit normalization control
stream("mmlu", normalize=True)   # Normalized
stream("mmlu", normalize=False)  # Raw data
```

### 4. **Confusing Parameter Names** ❌
**Common Confusion**:
```python
# What's the difference?
.limit(100)   # Stop after 100?
.sample(100)  # Random 100?
.head(100)    # First 100?

# Unclear naming
batch_size=32  # For what?
```

**Our Design**: ✅
```python
# Clear, single parameter
stream("mmlu", max_items=100)  # Stop after 100 items

# Clear purpose
batch_size=32  # For efficiency, not functionality
```

### 5. **Hidden State/Context** ❌
**Current Design**:
```python
# What context? Where does it come from?
context = EmberContext.current()
data_api = DataAPI(context)

# Hidden global state
data()  # What initialization happened?
```

**Our Design**: ✅
```python
# No context needed
stream("mmlu")  # Just works

# Registration is explicit when needed
register("my_data", FileSource("data.json"))
```

### 6. **Streaming vs Loading Confusion** ❌
**Common Confusion**:
```python
# Is this streaming or loaded?
dataset = load_dataset("mmlu")
for item in dataset:  # Streaming? In memory?
    pass

# When do I use which?
dataset.to_list()
dataset.materialize()
dataset.collect()
```

**Our Design**: ✅
```python
# Function name tells you everything
stream("mmlu")  # Returns Iterator (streaming)
load("mmlu")    # Returns List (in memory)

# No conversion methods needed
```

## Potential Remaining Confusion Points

### 1. **When to Normalize?** ⚠️
```python
# Users might not know when they need this
stream("mmlu", normalize=True)  # Default
stream("mmlu", normalize=False) # When?
```

**Solution**: Clear documentation
```python
def stream(source, *, normalize: bool = True):
    """
    Args:
        normalize: Convert to standard schema (question/answer/choices).
                  Set False only when you need original field names.
    """
```

### 2. **Custom Source Protocol** ⚠️
```python
# How do I implement this?
class MySource:
    def read_batches(self, batch_size=32):
        # What should this return?
```

**Solution**: Clear example in docs
```python
class MySource:
    def read_batches(self, batch_size=32):
        """Yield lists of dictionaries."""
        yield [{"question": "Q1", "answer": "A1"}]
```

### 3. **File Type Detection** ⚠️
```python
# Will this work?
from_file("data.xlsx")  # Not supported
```

**Solution**: Clear error messages
```python
raise ValueError(
    f"Unsupported file type: {path.suffix}\n"
    f"Supported: .json, .jsonl, .csv"
)
```

## UX Confusion Score

### Current Design: 7/10 Confusion
- Multiple API levels
- Unclear return types  
- Magic behavior
- Complex initialization
- Hidden state

### Our Design: 2/10 Confusion
- Two clear functions
- Predictable types
- Explicit behavior
- No initialization needed
- No hidden state

## The "Pit of Success" Test

Good UX design leads users into the "pit of success" - doing the right thing should be the easiest thing.

### ✅ **Success Paths in Our Design**

1. **Default Usage = Best Practice**
   ```python
   # This default is memory-efficient streaming
   for item in stream("mmlu"):
       process(item)
   ```

2. **Explicit Choice for Memory Usage**
   ```python
   # User must explicitly choose to load all
   all_items = load("mmlu")  # Clear memory implication
   ```

3. **No Accidental Complexity**
   ```python
   # Can't accidentally create complex chains
   # No: data().filter().transform().limit().materialize()
   # Just: stream() with simple parameters
   ```

## Comparison with Popular Libraries

### Pandas (Confusing)
```python
# Multiple ways, unclear memory usage
pd.read_csv("file.csv")
pd.read_csv("file.csv", chunksize=1000)
pd.read_csv("file.csv", iterator=True)
```

### Our Design (Clear)
```python
# Streaming (default)
stream(FileSource("file.csv"))

# Load all (explicit)
load(FileSource("file.csv"))
```

### HuggingFace Datasets (Confusing)
```python
# What's the difference?
dataset = load_dataset("squad")
dataset = load_dataset("squad", streaming=True)
dataset = dataset.map(fn)  # Lazy? Eager?
```

### Our Design (Clear)
```python
# Function name = behavior
stream("squad")  # Streaming
load("squad")    # Load all
```

## Final UX Assessment

### ✅ **No UX Confusion Because:**

1. **Function names describe behavior** (stream vs load)
2. **Return types are predictable** (Iterator vs List)
3. **No magic or hidden behavior**
4. **Parameters have clear names** (max_items, not limit)
5. **No multiple ways to do the same thing**
6. **Errors are helpful and specific**
7. **Defaults are the best practice**

### ⚠️ **Minimal Confusion Points:**

1. **When to use normalize=False** → Solved by documentation
2. **How to implement custom sources** → Solved by examples
3. **Supported file types** → Solved by error messages

## The Ultimate Test

Can a new user understand this in 30 seconds?

```python
from ember.api import stream, load

# Stream data (memory efficient)
for item in stream("mmlu"):
    print(item["question"])

# Load all data (uses memory)
data = load("mmlu", max_items=1000)
```

**Yes.** No confusion. No questions. It just makes sense.

## Conclusion

This design achieves what Jobs would call "intuitive" - it works the way users expect. There's no UX confusion because:

1. **Names match behavior** exactly
2. **No abstractions to learn**
3. **No choices to make**
4. **No magic to discover**

The confusion score is near zero. Users fall into the "pit of success" by default.