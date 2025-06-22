# SOLID Analysis: Data Module Redesign

## Robert C. Martin's Perspective

Uncle Bob would likely have mixed feelings about our pure functional approach. Let's analyze against SOLID principles:

## SOLID Principle Analysis

### 1. **Single Responsibility Principle (SRP)** ✅
**Current Design**: Each function has one clear responsibility
- `load_dataset()`: Stream data from registry
- `load_dataset_eagerly()`: Load data into memory
- `_normalize_item()`: Normalize to standard schema

**Martin's View**: "A module should have one, and only one, reason to change."
- ✅ Our functions change only when data loading logic changes
- ❌ But `load_dataset()` does multiple things: registry lookup, source creation, normalization, streaming

### 2. **Open/Closed Principle (OCP)** ❌ 
**Current Design**: Functions with hard-coded behavior
- Adding new data sources requires modifying `_create_source()`
- Adding new normalization rules requires modifying `_normalize_item()`

**Martin's View**: "Software entities should be open for extension, closed for modification."
- ❌ Our design requires modification to add new behaviors
- ❌ No abstraction for data sources or normalization strategies

### 3. **Liskov Substitution Principle (LSP)** ⚠️
**Current Design**: Protocol for DataSource
```python
class DataSource(Protocol):
    def iter_batches(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
```

**Martin's View**: "Derived classes must be substitutable for their base classes."
- ✅ Protocol ensures substitutability
- ⚠️ But we're avoiding inheritance altogether

### 4. **Interface Segregation Principle (ISP)** ✅
**Current Design**: Minimal Protocol interface
- Only one method required: `iter_batches()`

**Martin's View**: "Clients should not be forced to depend on interfaces they don't use."
- ✅ DataSource protocol is minimal
- ✅ No fat interfaces

### 5. **Dependency Inversion Principle (DIP)** ❌
**Current Design**: Direct dependencies
- `load_dataset()` directly creates sources
- `_normalize_item()` has hard-coded normalization logic

**Martin's View**: "Depend on abstractions, not concretions."
- ❌ Functions depend on concrete implementations
- ❌ No dependency injection

## What Robert C. Martin Would Actually Build

Martin would likely propose a more object-oriented design:

```python
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Any
import threading


# Domain Entities (following DDD)
class DatasetItem:
    """Value object representing a dataset item."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = self._normalize(data)
    
    @property
    def question(self) -> str:
        return self._data.get("question", "")
    
    @property
    def answer(self) -> str:
        return self._data.get("answer", "")
    
    @property
    def options(self) -> Dict[str, str]:
        return self._data.get("options", {})
    
    def _normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize to standard schema."""
        return {
            "question": data.get("question") or data.get("query", ""),
            "answer": data.get("answer") or data.get("correct_answer", ""),
            "options": data.get("options") or data.get("choices", {}),
            "metadata": {k: v for k, v in data.items() 
                        if k not in ["question", "query", "answer", 
                                    "correct_answer", "options", "choices"]}
        }


# Abstractions (DIP)
class DataSource(ABC):
    """Abstract data source."""
    
    @abstractmethod
    def stream(self, batch_size: int) -> Iterator[DatasetItem]:
        """Stream dataset items."""
        pass


class DatasetRepository(ABC):
    """Abstract repository for dataset metadata."""
    
    @abstractmethod
    def find(self, name: str) -> Optional['DatasetInfo']:
        """Find dataset by name."""
        pass
    
    @abstractmethod
    def list_available(self) -> List[str]:
        """List available dataset names."""
        pass


class DataSourceFactory(ABC):
    """Abstract factory for creating data sources."""
    
    @abstractmethod
    def create(self, info: 'DatasetInfo') -> DataSource:
        """Create appropriate data source."""
        pass


# Concrete Implementations
class HuggingFaceDataSource(DataSource):
    """Data source for HuggingFace datasets."""
    
    def __init__(self, dataset_name: str, config: Optional[str] = None):
        self.dataset_name = dataset_name
        self.config = config
    
    def stream(self, batch_size: int) -> Iterator[DatasetItem]:
        # Implementation details
        pass


class RegistryDatasetRepository(DatasetRepository):
    """Thread-safe dataset repository."""
    
    def __init__(self):
        self._datasets: Dict[str, DatasetInfo] = {}
        self._lock = threading.Lock()
    
    def find(self, name: str) -> Optional['DatasetInfo']:
        with self._lock:
            return self._datasets.get(name)
    
    def list_available(self) -> List[str]:
        with self._lock:
            return list(self._datasets.keys())


# Use Case (following Clean Architecture)
class LoadDatasetUseCase:
    """Use case for loading datasets."""
    
    def __init__(
        self,
        repository: DatasetRepository,
        source_factory: DataSourceFactory
    ):
        self.repository = repository
        self.source_factory = source_factory
    
    def execute(
        self,
        name: str,
        batch_size: int = 32,
        limit: Optional[int] = None
    ) -> Iterator[DatasetItem]:
        """Load dataset by name."""
        info = self.repository.find(name)
        
        if not info:
            available = self.repository.list_available()[:5]
            raise ValueError(
                f"Dataset '{name}' not found. "
                f"Available: {', '.join(available)}"
            )
        
        source = self.source_factory.create(info)
        count = 0
        
        for item in source.stream(batch_size):
            if limit and count >= limit:
                return
            yield item
            count += 1


# Facade for Simple API (maintaining ease of use)
class DataLoader:
    """Simple facade for data loading."""
    
    def __init__(self):
        # Dependency injection container setup
        self._repository = RegistryDatasetRepository()
        self._factory = DefaultDataSourceFactory()
        self._use_case = LoadDatasetUseCase(self._repository, self._factory)
    
    def load(
        self,
        name: str,
        *,
        batch_size: int = 32,
        limit: Optional[int] = None
    ) -> Iterator[DatasetItem]:
        """Load dataset as a stream."""
        return self._use_case.execute(name, batch_size, limit)
    
    def load_all(
        self,
        name: str,
        *,
        limit: Optional[int] = None
    ) -> List[DatasetItem]:
        """Load entire dataset into memory."""
        return list(self.load(name, limit=limit))


# Global instance for convenience
_loader = DataLoader()
load_dataset = _loader.load
load_dataset_eagerly = _loader.load_all
```

## Comparison: SOLID vs Simplicity

### Martin's Approach
**Pros**:
- ✅ Fully SOLID compliant
- ✅ Extensible without modification
- ✅ Testable with mocks/stubs
- ✅ Clear separation of concerns
- ✅ Follows Clean Architecture

**Cons**:
- ❌ 4x more code
- ❌ Multiple classes for simple data loading
- ❌ Cognitive overhead for users
- ❌ Violates YAGNI (You Aren't Gonna Need It)

### Our Simple Approach
**Pros**:
- ✅ Dead simple to understand
- ✅ Minimal code
- ✅ Direct and efficient
- ✅ What Carmack would write

**Cons**:
- ❌ Not fully SOLID
- ❌ Requires modification for extension
- ❌ Harder to unit test in isolation

## The Tension: Martin vs The Masters

**Robert C. Martin** emphasizes:
- Abstraction and indirection
- Dependency injection
- Interface-based design
- Extensibility

**Dean/Ghemawat/Carmack** emphasize:
- Simplicity and directness
- Minimal abstraction
- Performance
- Getting things done

## Hybrid Approach: Best of Both Worlds

```python
"""Data loading with pluggable sources."""

from typing import Protocol, Dict, Iterator, Optional, Any


class DataSource(Protocol):
    """Protocol for data sources."""
    def load(self, info: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Load data from source."""
        ...


class DataNormalizer(Protocol):
    """Protocol for data normalization."""
    def normalize(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize item to standard schema."""
        ...


def load_dataset(
    name: str,
    *,
    source: Optional[DataSource] = None,
    normalizer: Optional[DataNormalizer] = None,
    batch_size: int = 32,
    limit: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """Load dataset with optional custom source and normalizer.
    
    Args:
        name: Dataset name.
        source: Custom data source (default: auto-detect).
        normalizer: Custom normalizer (default: standard).
        batch_size: Batch size for streaming.
        limit: Maximum items.
        
    Yields:
        Normalized dictionary items.
    """
    # Use defaults if not provided
    if source is None:
        source = _default_source()
    if normalizer is None:
        normalizer = _default_normalizer()
    
    info = _get_dataset_info(name)
    count = 0
    
    for item in source.load(info):
        if limit and count >= limit:
            return
        yield normalizer.normalize(item)
        count += 1
```

## Recommendation

For Ember's data module, I recommend:

1. **Start Simple**: Use the functional approach
2. **Add Protocols**: For extension points (DataSource, DataNormalizer)
3. **Avoid Over-Engineering**: No abstract factories or use cases
4. **Document Extension Points**: Show how to customize when needed

This satisfies:
- Dean/Ghemawat/Carmack: Simple and direct
- Martin: Extensible through protocols
- Users: Easy to use, hard to misuse

The key insight: **SOLID principles are tools, not rules**. Use them when they add value, not dogmatically. For a data loader, simplicity trumps architectural purity.