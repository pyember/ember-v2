"""Memory-efficient streaming dataset implementation.

Processes datasets with O(1) memory usage regardless of dataset size,
using Python's iterator protocol for lazy evaluation.
"""

import logging
import threading
from itertools import islice
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Union

from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.context.data_context import DataContext

logger = logging.getLogger(__name__)


class StreamingTransformer(Protocol):
    """Protocol for streaming-compatible transformers."""

    def transform_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform a single item.

        Args:
            item: Dataset item to transform

        Returns:
            Transformed item or None to filter it out
        """
        ...


class StreamingDataset:
    """Memory-efficient dataset iterator with O(batch_size) memory footprint.

    This implementation:
    1. Uses lazy initialization to defer dataset loading
    2. Processes items one at a time with minimal memory usage
    3. Supports transformations with streaming semantics
    4. Handles both dataset names and raw iterators

    Attributes:
        _source: Dataset name or iterator of items
        _transformers: Optional transformers to apply to items
        _batch_size: Number of items to process at once
        _prepper: Optional dataset prepper to create entries
    """

    def __init__(
        self,
        source: Union[str, Iterator[Dict[str, Any]], List[Dict[str, Any]]],
        *,
        transformers: Optional[List[StreamingTransformer]] = None,
        batch_size: int = 32,
        prepper: Optional[Any] = None,
        _limit: Optional[int] = None,
        _data_context: Optional[DataContext] = None,
    ):
        """Initialize with dataset source.

        Args:
            source: Dataset name or iterator of items
            transformers: Optional transformers to apply to items
            batch_size: Number of items to process at once
            prepper: Optional dataset prepper to create entries
            _limit: Internal parameter for limiting results
            _data_context: Optional data context for registry/loader access
        """
        self._source = source
        self._transformers = transformers or []
        self._batch_size = max(1, batch_size)
        self._prepper = prepper
        self._limit = _limit
        self._data_context = _data_context

        # Lazy initialization state
        self._initialized = False
        self._source_iter = None

        # Thread safety for initialization
        self._init_lock = threading.RLock()

    def __iter__(self) -> Iterator[DatasetEntry]:
        """Return self as iterator."""
        self._ensure_initialized()

        # Create basic iterator
        iterator = self._create_iterator()

        # Apply limit if specified
        if self._limit is not None:
            return islice(iterator, self._limit)

        return iterator

    def _create_iterator(self) -> Iterator[DatasetEntry]:
        """Create base iterator that processes items.

        Returns:
            Iterator over dataset entries
        """
        while True:
            try:
                # Get next item
                item = next(self._source_iter)

                # Apply transformations (may return None to filter out)
                item = self._transform(item)

                # Skip None items (filtered out)
                if item is None:
                    continue

                # Create dataset entry
                yield self._prepare(item)
            except StopIteration:
                self._initialized = False  # Reset for potential reuse
                break

    def __next__(self) -> DatasetEntry:
        """Get next dataset entry.

        This method is called when directly iterating over the dataset.
        It reuses the iterator created by __iter__.

        Returns:
            Next dataset entry

        Raises:
            StopIteration: When no more items
        """
        # Initialize on first call
        if not hasattr(self, "_iter"):
            self._iter = iter(self)

        # Get next item
        return next(self._iter)

    def _transform(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply transformations to item.

        Args:
            item: Item to transform

        Returns:
            Transformed item or None if filtered out
        """
        result = item
        for transformer in self._transformers:
            # Skip None items (previous transformer filtered it out)
            if result is None:
                return None
            result = transformer.transform_item(result)
        return result

    def _prepare(self, item: Dict[str, Any]) -> DatasetEntry:
        """Create dataset entry from transformed item.

        Args:
            item: Transformed item

        Returns:
            Dataset entry
        """
        # Try preppers in order of specificity
        prepper = self._prepper or self._get_dataset_prepper()

        if prepper and hasattr(prepper, "create_dataset_entries"):
            entries = prepper.create_dataset_entries(item=item)
            if entries and entries[0]:
                return entries[0]

        # Simple fallback with direct conversion
        return self._create_entry(item)

    def _get_dataset_prepper(self):
        """Get dataset prepper from registry if available.

        Returns:
            Dataset prepper or None if not available.
        """
        # For named datasets, look up the prepper in the registry
        if isinstance(self._source, str):
            # Get registry from data context
            registry = self._data_context.registry
            dataset_entry = registry.get(name=self._source)
            if dataset_entry and hasattr(dataset_entry, "prepper"):
                return dataset_entry.prepper
        return None

    def _ensure_initialized(self) -> None:
        """Initialize source iterator if needed in a thread-safe way."""
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    # Require context if loading by name
                    if isinstance(self._source, str) and self._data_context is None:
                        raise ValueError(
                            "DataContext must be provided to StreamingDataset when loading by name."
                        )
                    self._initialize_source()
                    self._initialized = True

    def _initialize_source(self) -> None:
        """Initialize source iterator based on source type."""
        if isinstance(self._source, str):
            # Use the provided data context
            data_context = self._data_context
            if (
                data_context is None
            ):  # Should not happen due to check in _ensure_initialized
                raise RuntimeError(
                    "DataContext is missing during source initialization."
                )

            registry = data_context.registry

            # First check the registry for the dataset
            dataset_entry = registry.get(name=self._source)

            # If not found, ensure registry is initialized
            if not dataset_entry or not dataset_entry.info:
                # Initialize registry
                from ember.core.utils.data.initialization import (
                    initialize_dataset_registry,
                )

                initialize_dataset_registry(metadata_registry=registry)

                # Try again
                dataset_entry = registry.get(name=self._source)

            # Final check with helpful error message
            if not dataset_entry or not dataset_entry.info:
                available = registry.list_datasets()
                raise ValueError(
                    f"Dataset '{self._source}' not found in registry. Available datasets: {available}"
                )

            # Get dataset source name
            source_name = dataset_entry.info.source

            # Create or get a cached loader using the data context
            loader = self._get_or_create_loader(data_context, source_name)

            # Load dataset
            data = loader.load(dataset_name=source_name)
            self._source_iter = iter(data)
        elif isinstance(self._source, list):
            # Use list as iterator (convert once)
            self._source_iter = iter(self._source)
        elif hasattr(self._source, "__iter__"):
            # Use existing iterator
            self._source_iter = iter(self._source)
        else:
            # Invalid source type
            raise TypeError(f"Unsupported source type: {type(self._source)}")

    def _get_or_create_loader(self, data_context, source_name: str) -> Any:
        """Get or create a dataset loader with caching.

        Args:
            data_context: The data context
            source_name: Name of the dataset source

        Returns:
            A dataset loader instance
        """
        # Import here to avoid circular imports
        from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader

        # Use the dataset service loader if available
        if hasattr(data_context, "dataset_service") and data_context.dataset_service:
            return data_context.dataset_service.loader

        # Get configuration from the context
        config = data_context.config

        # Create loader with appropriate configuration
        return HuggingFaceDatasetLoader(
            cache_dir=config.cache_dir,
            use_auth_token=getattr(config, "huggingface_token", None),
        )

    def _create_entry(self, item: Dict[str, Any]) -> DatasetEntry:
        """Create dataset entry from item with flexible field mapping.

        Args:
            item: Dataset item

        Returns:
            Dataset entry
        """
        # Extract mandatory and optional fields with sensible defaults
        item_id = str(item.get("id", id(item)))

        # Items often have different schemas - support both mapping forms
        if "query" in item:
            query = item["query"]
        elif "question" in item:
            query = item["question"]
        else:
            # Fallback to string representation
            query = str(item)

        # Get choices with multiple field names
        choices = None
        for field in ["choices", "options", "answers"]:
            if field in item:
                choices = item[field]
                break

        # Create dataset entry with defaults
        entry = DatasetEntry(
            query=query,
            choices=choices or {},
            metadata={"id": item_id, **item.get("metadata", {})},
        )

        # Add additional fields from item
        for key, value in item.items():
            if key not in ["query", "choices", "metadata", "id"]:
                entry.metadata[key] = value

        return entry

    def limit(self, count: int) -> "StreamingDataset":
        """Limit number of items to process.

        Args:
            count: Maximum number of items

        Returns:
            New streaming dataset with item limit
        """
        if count <= 0:
            raise ValueError(f"Count must be positive: {count}")

        # Create new dataset with same parameters but with limit
        return StreamingDataset(
            source=self._source,
            transformers=self._transformers.copy(),
            batch_size=self._batch_size,
            prepper=self._prepper,
            _limit=count,
            _data_context=self._data_context,
        )

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "StreamingDataset":
        """Filter items based on predicate.

        Args:
            predicate: Function that returns True for items to keep

        Returns:
            New streaming dataset with filter applied
        """

        # Create filter transformer adapter
        class FilterTransformer:
            def __init__(self, predicate_fn):
                self.predicate_fn = predicate_fn

            def transform_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                # Skip filtered items
                if item is None:
                    return None

                # Apply predicate
                return item if self.predicate_fn(item) else None

        # Add to transformers list
        new_transformers = self._transformers.copy()
        new_transformers.append(FilterTransformer(predicate))

        # Create new instance with updated transformers
        return StreamingDataset(
            source=self._source,
            transformers=new_transformers,
            batch_size=self._batch_size,
            prepper=self._prepper,
            _limit=self._limit,
            _data_context=self._data_context,
        )

    def transform(
        self, transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> "StreamingDataset":
        """Apply transformation to each item.

        Args:
            transform_fn: Function to transform items

        Returns:
            New streaming dataset with transform applied
        """

        # Create transformer adapter
        class FunctionTransformerAdapter:
            def __init__(self, fn):
                self.transform_fn = fn

            def transform_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                # Skip filtered items
                if item is None:
                    return None

                # Apply transformation
                return self.transform_fn(item)

        # Add to transformers list
        new_transformers = self._transformers.copy()
        new_transformers.append(FunctionTransformerAdapter(transform_fn))

        # Create new instance with updated transformers
        return StreamingDataset(
            source=self._source,
            transformers=new_transformers,
            batch_size=self._batch_size,
            prepper=self._prepper,
            _limit=self._limit,
            _data_context=self._data_context,
        )

    def collect(self) -> List[DatasetEntry]:
        """Collect all dataset entries into a list.

        This is useful when you need to convert a streaming dataset to a list
        for operations that require random access.

        Warning: This loads the entire dataset into memory, so use with caution
        for large datasets.

        Returns:
            List of all dataset entries
        """
        return list(iter(self))

    def map(self, map_fn: Callable[[DatasetEntry], DatasetEntry]) -> "StreamingDataset":
        """Map function over dataset entries.

        Unlike transform, this operates on the DatasetEntry objects after
        they have been created, rather than on the raw dictionaries.

        Args:
            map_fn: Function to apply to each dataset entry

        Returns:
            New streaming dataset with mapping applied
        """
        # Create a generator that applies the map function
        source_iter = self

        class MappedIterator:
            def __iter__(self):
                for entry in source_iter:
                    yield map_fn(entry)

        # Create a new dataset with the mapped iterator as source
        return StreamingDataset(
            source=iter(MappedIterator()),
            batch_size=self._batch_size,
            _limit=self._limit,
            _data_context=self._data_context,
        )
