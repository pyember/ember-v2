# Data Module

Two functions. Stream or load.

```python
from ember.api import stream, load

# Stream - O(1) memory
for item in stream("mmlu"):
    process(item)

# Load - O(n) memory  
data = load("mmlu")
```

## API

### `stream(dataset, **kwargs) -> Iterator[Dict]`

Returns iterator. Memory constant regardless of dataset size.

```python
stream("mmlu")                              # Basic
stream("mmlu", split="test")                # Split
stream("mmlu", filter=lambda x: x["id"]>10) # Filter
stream("mmlu", transform=lambda x: x|{"new": 1}) # Transform
stream("mmlu", batch_size=64)               # Batch size
stream("mmlu", max_items=1000)              # Limit
```

Chaining for 10% of cases:
```python
(stream("mmlu")
 .filter(lambda x: x["valid"])
 .transform(normalize)
 .batch(32))
```

### `load(dataset, **kwargs) -> List[Dict]`

Returns list. Loads entire dataset into memory. Same kwargs as stream.

```python
data = load("mmlu", split="validation")
```

### Additional Functions

```python
metadata("mmlu")          # -> DatasetMetadata(size_bytes=..., estimated_examples=...)
list_datasets()           # -> ["mmlu", "gpqa", "aime", ...]
register("custom", src)   # Register DataSource
load_file("data.json")    # -> List[Dict]
from_file("data.jsonl")   # -> Iterator[Dict]
```

## Performance

| Operation | Memory | Time | Notes |
|-----------|--------|------|-------|
| stream()  | O(1)   | O(n) | Constant memory, linear time |
| load()    | O(n)   | O(n) | Linear memory, includes parse time |
| .batch(k) | O(k)   | O(n) | Batch size k items in memory |
| .filter() | O(1)   | O(n) | No memory overhead |
| .transform() | O(1) | O(n) | Applied per item |

Streaming processes 1GB/s on NVMe SSD. Network adds 10-50ms/MB.

## Implementation

```python
class StreamIterator:
    def __iter__(self): return self
    def __next__(self): return self._next_item()
    def filter(self, pred): return FilterIterator(self, pred)
    def transform(self, fn): return TransformIterator(self, fn)
    def batch(self, size): return BatchIterator(self, size)
```

DataSource protocol:
```python
def read_batches(self, batch_size: int) -> Iterator[List[Dict]]:
    """Yield batches of items."""
```

## Examples

### Basic Usage

```python
# Training loop
for epoch in range(10):
    for batch in stream("train_data").batch(32):
        loss = model.train_step(batch)
        
# Evaluation
val_data = load("dataset", split="validation")
accuracy = evaluate(model, val_data)

# Data analysis
lengths = [len(item["text"]) for item in stream("dataset")]
print(f"Average: {sum(lengths) / len(lengths)}")
```

### Advanced Patterns

```python
# Process 1TB dataset with 100MB RAM
for item in stream("huge_dataset", batch_size=1000):
    embedding = model.embed(item["text"])
    index.add(item["id"], embedding)

# Parallel processing
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for batch in stream("dataset").batch(100):
        futures.append(executor.submit(process_batch, batch))

# Export with progress
total = metadata("dataset").estimated_examples
for i, item in enumerate(stream("dataset")):
    if i % 1000 == 0:
        print(f"{i}/{total}")
    write_to_db(item)

# Custom data source
class LiveDataSource:
    def read_batches(self, batch_size):
        while True:
            batch = fetch_from_api(batch_size)
            if not batch:
                break
            yield batch

register("live_feed", LiveDataSource())
for item in stream("live_feed"):
    if urgent(item):
        alert(item)
```

### Common Workflows

```python
# Filter and transform pipeline
pipeline = (stream("raw_data")
    .filter(lambda x: x["quality"] > 0.8)
    .transform(lambda x: {
        "text": clean_text(x["text"]),
        "label": x["label"],
        "metadata": extract_metadata(x)
    })
    .batch(64))

for batch in pipeline:
    processed_results = model.process_batch(batch)
    save_results(processed_results)

# Multi-dataset training
datasets = ["dataset1", "dataset2", "dataset3"]
for dataset_name in datasets:
    for item in stream(dataset_name, split="train"):
        model.update(item)

# Export to different formats
import json
import csv

# JSON Lines
with open("output.jsonl", "w") as f:
    for item in stream("dataset"):
        f.write(json.dumps(item) + "\n")

# CSV
with open("output.csv", "w") as f:
    writer = None
    for item in stream("dataset"):
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=item.keys())
            writer.writeheader()
        writer.writerow(item)
```

## Design Rationale

1. **Two functions**: 90% use stream(), 10% use load()
2. **Iterator protocol**: Composes with Python ecosystem
3. **Explicit batching**: User controls memory usage
4. **No hidden state**: Each iteration independent
5. **Protocol not base class**: extend via `read_batches()`

## Migration

```python
# Before: 7 lines, 3 imports, context management
from ember import EmberContext
from ember.data import DataAPI
ctx = EmberContext()
datasets = DataAPI(ctx).datasets()
loader = datasets["mmlu"].subset("train").build()
for batch in loader:
    for item in batch:
        process(item)

# After: 2 lines, 1 import
from ember.api import stream
for item in stream("mmlu", subset="train"):
    process(item)
```

## See Also

- [data.py](data.py) - Implementation
- [STREAMING_REQUIREMENTS.md](../core/utils/data/STREAMING_REQUIREMENTS.md) - Design constraints