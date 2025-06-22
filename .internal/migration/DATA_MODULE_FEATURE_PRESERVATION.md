# Feature Preservation Analysis: New Data Design

## Features We Must Preserve

### 1. **Registry Support** ✅
**Current**: Registry with named datasets
**New Design**: ✅ Preserved in `_Registry` class
```python
# Still works
stream("mmlu")  # Named dataset from registry
```

### 2. **Transformations** ⚠️ 
**Current**: Chain `.filter().transform()`
**New Design**: ⚠️ Need to add this capability
```python
# Current design allows
data("mmlu").filter(subject="physics").transform(add_prompt)

# Our design needs
stream("mmlu", filter=lambda x: x["subject"] == "physics")
# But no transform chaining!
```

### 3. **Subset/Split Selection** ❌
**Current**: `.subset("physics").split("test")`
**New Design**: ❌ Missing this important feature
```python
# Need to support
stream("mmlu", subset="physics", split="test")
```

### 4. **Custom Datasets** ✅
**Current**: Can register custom datasets
**New Design**: ✅ Preserved with `register()`
```python
register("my_data", FileSource("data.json"))
```

### 5. **Chainable Operations** ⚠️
**Current**: Fluent interface for complex pipelines
**New Design**: ⚠️ Lost the ability to chain operations
```python
# Can't do this anymore
stream("mmlu").filter(...).transform(...).limit(...)
```

### 6. **Metadata Access** ❌
**Current**: Can get dataset metadata
**New Design**: ❌ No metadata API
```python
# Missing
metadata = get_metadata("mmlu")
print(f"Size: {metadata.size_bytes}")
```

## Revised Design: Best of Both Worlds

```python
"""Data module - Simple by default, powerful when needed."""

from typing import Dict, Iterator, List, Optional, Protocol, Callable, Any
from dataclasses import dataclass
import threading


# Core protocols
@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources."""
    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches of dictionaries."""
        ...


# Metadata (important for users to know dataset characteristics)
@dataclass(frozen=True)
class DatasetInfo:
    """Essential dataset information."""
    name: str
    description: str
    size_bytes: int
    example_count: int
    example_item: Dict[str, Any]
    streaming_supported: bool = True
    
    
# Core streaming function with all needed parameters
def stream(
    source: str | DataSource,
    *,
    # Dataset selection
    subset: Optional[str] = None,
    split: Optional[str] = None,
    # Processing
    filter: Optional[Callable[[Dict], bool]] = None,
    transform: Optional[Callable[[Dict], Dict]] = None,
    # Control
    batch_size: int = 32,
    max_items: Optional[int] = None,
    normalize: bool = True
) -> 'StreamIterator':
    """Stream data with optional filtering and transformation.
    
    Args:
        source: Dataset name or custom DataSource.
        subset: Dataset subset (e.g., "physics" for MMLU).
        split: Dataset split (train/validation/test).
        filter: Function to filter items.
        transform: Function to transform items.
        batch_size: Items per batch.
        max_items: Maximum items to yield.
        normalize: Apply standard normalization.
        
    Returns:
        StreamIterator that supports chaining.
        
    Examples:
        # Simple (90% of cases)
        for item in stream("mmlu"):
            print(item["question"])
            
        # With selection
        for item in stream("mmlu", subset="physics", split="test"):
            print(item["question"])
            
        # With inline processing
        for item in stream("mmlu", 
                          filter=lambda x: x["score"] > 0.5,
                          transform=lambda x: {...x, "prompt": f"Q: {x['question']}"}):
            process(item)
            
        # With chaining (advanced)
        stream("mmlu").filter(has_math).transform(add_prompt).first(100)
    """
    # Get source
    if isinstance(source, str):
        data_source = _registry.get_source(source, subset, split)
    else:
        data_source = source
        
    # Create iterator with all parameters
    return StreamIterator(
        data_source, 
        filter=filter,
        transform=transform,
        batch_size=batch_size,
        max_items=max_items,
        normalize=normalize
    )


class StreamIterator:
    """Iterator with chainable operations for advanced usage."""
    
    def __init__(self, source: DataSource, **params):
        self._source = source
        self._filter = params.get('filter')
        self._transform = params.get('transform')
        self._batch_size = params.get('batch_size', 32)
        self._max_items = params.get('max_items')
        self._normalize = params.get('normalize', True)
        self._count = 0
        
    def __iter__(self):
        """Iterate with all transformations applied."""
        for batch in self._source.read_batches(self._batch_size):
            for item in batch:
                # Check limit
                if self._max_items and self._count >= self._max_items:
                    return
                    
                # Normalize if requested
                if self._normalize:
                    item = _normalize(item)
                    
                # Apply filter
                if self._filter and not self._filter(item):
                    continue
                    
                # Apply transform
                if self._transform:
                    item = self._transform(item)
                    
                yield item
                self._count += 1
    
    # Chainable methods for advanced usage
    def filter(self, predicate: Callable[[Dict], bool]) -> 'StreamIterator':
        """Add or chain a filter."""
        # Combine with existing filter
        if self._filter:
            old_filter = self._filter
            new_filter = lambda x: old_filter(x) and predicate(x)
        else:
            new_filter = predicate
            
        return StreamIterator(
            self._source,
            filter=new_filter,
            transform=self._transform,
            batch_size=self._batch_size,
            max_items=self._max_items,
            normalize=self._normalize
        )
    
    def transform(self, fn: Callable[[Dict], Dict]) -> 'StreamIterator':
        """Add or chain a transformation."""
        # Combine with existing transform
        if self._transform:
            old_transform = self._transform
            new_transform = lambda x: fn(old_transform(x))
        else:
            new_transform = fn
            
        return StreamIterator(
            self._source,
            filter=self._filter,
            transform=new_transform,
            batch_size=self._batch_size,
            max_items=self._max_items,
            normalize=self._normalize
        )
    
    def limit(self, n: int) -> 'StreamIterator':
        """Limit to n items."""
        return StreamIterator(
            self._source,
            filter=self._filter,
            transform=self._transform,
            batch_size=self._batch_size,
            max_items=n,
            normalize=self._normalize
        )
        
    def first(self, n: int) -> List[Dict[str, Any]]:
        """Get first n items as a list."""
        return list(self.limit(n))
        
    def collect(self) -> List[Dict[str, Any]]:
        """Collect all items into a list."""
        return list(self)


def load(
    source: str | DataSource,
    *,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    filter: Optional[Callable[[Dict], bool]] = None,
    transform: Optional[Callable[[Dict], Dict]] = None,
    max_items: Optional[int] = None,
    normalize: bool = True
) -> List[Dict[str, Any]]:
    """Load data into memory.
    
    Same parameters as stream(), but returns a list.
    
    Examples:
        # Load with filters
        data = load("mmlu", subset="physics", max_items=1000)
    """
    return list(stream(
        source, 
        subset=subset,
        split=split,
        filter=filter,
        transform=transform,
        max_items=max_items,
        normalize=normalize
    ))


def metadata(dataset: str) -> DatasetInfo:
    """Get dataset metadata.
    
    Examples:
        info = metadata("mmlu")
        print(f"Dataset size: {info.size_bytes / 1e9:.1f} GB")
        print(f"Example count: {info.example_count:,}")
    """
    return _registry.get_metadata(dataset)


# Registry with subset/split support
class _Registry:
    """Internal registry for datasets."""
    
    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
        self._metadata: Dict[str, DatasetInfo] = {}
        self._lock = threading.Lock()
        
    def get_source(self, name: str, subset: Optional[str] = None, 
                   split: Optional[str] = None) -> DataSource:
        """Get source with optional subset/split."""
        with self._lock:
            # For HuggingFace datasets, create with subset/split
            if name in self._sources:
                source = self._sources[name]
                if hasattr(source, 'with_config'):
                    return source.with_config(subset=subset, split=split)
                return source
            else:
                # Try as HuggingFace dataset
                return HuggingFaceSource(name, split=split, config=subset)
                
    def get_metadata(self, name: str) -> DatasetInfo:
        """Get dataset metadata."""
        with self._lock:
            if name not in self._metadata:
                # Load metadata on demand
                self._metadata[name] = self._load_metadata(name)
            return self._metadata[name]


# Enhanced HuggingFace source with subset/split
class HuggingFaceSource:
    """HuggingFace dataset source."""
    
    def __init__(self, name: str, split: Optional[str] = None, 
                 config: Optional[str] = None):
        self.name = name
        self.split = split or "train"
        self.config = config
        
    def with_config(self, subset: Optional[str] = None, 
                    split: Optional[str] = None) -> 'HuggingFaceSource':
        """Create new source with different config."""
        return HuggingFaceSource(
            self.name,
            split=split or self.split,
            config=subset or self.config
        )
        
    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Read batches from HuggingFace."""
        # Implementation as before
        pass
```

## Feature Comparison: Final Design

| Feature | Original | Current Proposal | Final Design |
|---------|----------|------------------|--------------|
| Registry support | ✓ | ✓ | ✓ |
| Subset/split selection | ✓ | ❌ | ✓ |
| Transformations | ✓ | ❌ | ✓ |
| Filter chaining | ✓ | ❌ | ✓ |
| Streaming default | ✗ | ✓ | ✓ |
| Batching | ✓ | ✓ | ✓ |
| Custom datasets | ✓ | ✓ | ✓ |
| Zero-config | ✗ | ✓ | ✓ |
| Metadata access | ✓ | ❌ | ✓ |
| Normalized access | ✗ | ✓ | ✓ |

## Usage Examples: Progressive Disclosure

### Level 1: Simple (80% of users)
```python
# Just load data
for item in stream("mmlu"):
    print(item["question"])
```

### Level 2: Common Options (15% of users)
```python
# With subset and split
for item in stream("mmlu", subset="physics", split="test"):
    print(item["question"])

# With inline filter
for item in stream("mmlu", filter=lambda x: x["score"] > 0.5):
    print(item["question"])
```

### Level 3: Advanced Chaining (5% of users)
```python
# Complex pipeline with chaining
results = (stream("mmlu", subset="physics")
    .filter(lambda x: "quantum" in x["question"].lower())
    .transform(lambda x: {...x, "prompt": f"Explain: {x['question']}"})
    .first(100))
```

## Key Design Decisions

1. **Parameters vs Chaining**: Both! Parameters for simple cases, chaining for complex
2. **Subset/Split**: Critical for ML datasets, must be first-class
3. **Metadata**: Users need to know dataset characteristics
4. **Transform/Filter**: Both inline (simple) and chainable (advanced)

This preserves ALL important features while maintaining simplicity for basic usage.