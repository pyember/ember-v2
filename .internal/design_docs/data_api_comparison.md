# DataAPI v1 vs v2 Comparison

## Key Differences

### 1. **API Design Philosophy**

**DataAPI v1 (existing)**:
- Complex, feature-rich API with multiple abstraction layers
- Exposes many internal types (DatasetEntry, DatasetInfo, TaskType, etc.)
- Requires understanding of EmberContext and DataContext
- Builder pattern with many configuration options
- Supports both streaming and materialized datasets

**DataAPI v2 (new)**:
- Minimalist API following "one obvious way" principle
- Hides internal complexity (no DatasetEntry, DatasetInfo exposed)
- No context management required by users
- Streaming-first design with simple transformations
- Focuses on developer ergonomics

### 2. **Core Components**

**v1 Components**:
- `DataAPI` - Main API class requiring context injection
- `DatasetBuilder` - Complex builder with many methods
- `Dataset` - Container class for materialized data
- `DataItem` - Normalized representation with attribute access
- `DatasetEntry` - Internal data structure exposed to users
- Multiple transformer interfaces and adapters

**v2 Components**:
- `DataAPI` - Simple class with hidden context
- `StreamingDataset` - Single unified dataset type
- `DatasetMetadata` - Minimal metadata (replaces DatasetInfo)
- No exposed internal types or complex hierarchies

### 3. **Usage Patterns**

**v1 Usage**:
```python
# Complex initialization
data_api = DataAPI(EmberContext.current())

# Builder pattern
dataset = (
    data_api.builder()
    .from_registry("mmlu")
    .subset("physics")
    .split("test")
    .sample(100)
    .transform(lambda x: {"query": f"Question: {x['question']}"})
    .build()
)

# Direct call with kwargs
items = data_api("mmlu", streaming=True, limit=100)
```

**v2 Usage**:
```python
# Simple initialization
data = DataAPI()

# Direct loading with chaining
items = data.load("mmlu").filter(
    lambda x: x["subject"] == "physics"
).limit(100)

# Minimal metadata
info = data.metadata("mmlu")
```

### 4. **Features to Preserve from v1**

1. **Registry Integration**: v1 has robust dataset registry with discovery
2. **DataContext Integration**: Proper separation of concerns
3. **Transform Pipeline**: v1's transformer interface is more flexible
4. **Prepper System**: v1's dataset preppers handle complex data formats
5. **Validation**: v1 includes dataset validation
6. **Batch Configuration**: v1 allows fine-grained batch control
7. **Error Messages**: v1 provides helpful "available datasets" messages
8. **Type Safety**: v1 exposes types for better IDE support

### 5. **Features Missing in v2**

1. **Non-streaming mode**: v2 is streaming-only, v1 supports materialization
2. **Complex transformers**: v2 only supports simple functions
3. **Dataset splits**: v2 doesn't handle train/test/val splits
4. **Sampling with seed**: v2 lacks reproducible sampling
5. **HuggingFace config**: v1 handles subset configurations
6. **Custom loaders**: v1 supports pluggable loaders
7. **Progress tracking**: v1 can integrate with progress bars
8. **Context management**: v1 allows explicit context control

### 6. **Benefits of v2**

1. **Simplicity**: Much easier to understand and use
2. **Streaming-first**: Better memory efficiency by default
3. **Functional style**: Clean method chaining
4. **Hidden complexity**: Users don't see internal details
5. **Minimal imports**: Fewer types to import

## Recommendations

### Keep from v1:
1. Registry system with dataset discovery
2. Flexible transformer pipeline (but simplify the interface)
3. Prepper system for data normalization
4. Context-based configuration
5. Helpful error messages with available datasets
6. Support for both streaming and materialized modes

### Adopt from v2:
1. Simple API surface without exposed internals
2. Streaming-first design
3. Method chaining for transformations
4. Minimal metadata approach
5. Hidden context management

### Hybrid Approach:
```python
# Simple API by default
data = DataAPI()  # Auto-creates context
for item in data("mmlu").limit(100):
    process(item)

# Advanced API when needed
dataset = data.builder()
    .dataset("mmlu") 
    .split("test")
    .streaming(False)  # Get materialized dataset
    .seed(42)
    .build()

# Minimal but useful metadata
info = data.info("mmlu")
print(f"Size: {info.size_mb} MB, Examples: {info.count}")
```

This would give us the best of both worlds - simple for common cases, powerful when needed.