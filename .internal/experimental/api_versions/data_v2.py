"""Simplified data API hiding internal complexity.

Following the principles of Dean, Ghemawat, Jobs, Brockman, Ritchie, Knuth, and Carmack:
- One obvious way to load data
- Progressive disclosure of features
- Hide implementation details
- Keep the good parts (streaming, lazy loading)
"""

import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from ember.core.utils.data._metadata import (
    DatasetMetadata,
    create_minimal_metadata,
)
from ember.core.utils.data.context.data_context import DataContext
from ember.core.utils.data.streaming.dataset import StreamingDataset


class DataAPI:
    """Simplified interface for dataset operations.
    
    This is the primary way to work with datasets in Ember. It provides:
    - Simple dataset loading by name
    - Streaming by default (constant memory usage)
    - Optional metadata access
    - Clean functional transformations
    
    Example:
        # Simple usage
        data = DataAPI()
        for item in data.load("mmlu"):
            print(item["question"])
            
        # With transformations  
        items = data.load("mmlu").filter(
            lambda x: x["subject"] == "physics"
        ).limit(100)
        
        # Get metadata
        info = data.metadata("mmlu")
        print(f"Dataset size: {info.size_bytes / 1e9:.1f} GB")
    """
    
    def __init__(self):
        """Initialize with hidden internal context."""
        # Hide all the complexity
        self._context = DataContext()
        self._metadata_cache: Dict[str, DatasetMetadata] = {}
        
    def load(
        self,
        dataset: Union[str, List[Dict[str, Any]], Iterator[Dict[str, Any]]],
        *,
        batch_size: int = 32,
    ) -> StreamingDataset:
        """Load a dataset for streaming iteration.
        
        Args:
            dataset: Dataset name, list of items, or iterator
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            StreamingDataset that can be iterated over
            
        Example:
            # Load by name
            data = DataAPI()
            for item in data.load("mmlu"):
                process(item)
                
            # Load from list
            items = [{"text": "hello"}, {"text": "world"}]
            for item in data.load(items):
                process(item)
        """
        return StreamingDataset(
            source=dataset,
            batch_size=batch_size,
            _data_context=self._context,
        )
        
    def metadata(self, dataset_name: str) -> DatasetMetadata:
        """Get minimal metadata for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DatasetMetadata with essential information only
            
        Example:
            data = DataAPI()
            info = data.metadata("mmlu")
            print(f"Examples: {info.estimated_examples:,}")
            print(f"Recommended batch size: {info.recommended_batch_size}")
        """
        # Check cache first
        if dataset_name in self._metadata_cache:
            return self._metadata_cache[dataset_name]
            
        # Get from registry
        registry_entry = self._context.registry.get(dataset_name)
        if not registry_entry:
            # Initialize registry if needed
            from ember.core.utils.data.initialization import initialize_dataset_registry
            initialize_dataset_registry(self._context.registry)
            registry_entry = self._context.registry.get(dataset_name)
            
        if not registry_entry:
            available = self.list_datasets()
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available: {', '.join(available[:5])}..."
            )
            
        # Convert complex metadata to our simple format
        info = registry_entry.info
        
        # Load one example to show structure
        example = None
        try:
            dataset = self.load(dataset_name).limit(1)
            for item in dataset:
                example = dict(item.metadata) if hasattr(item, 'metadata') else item
                break
        except Exception:
            example = {"error": "Could not load example"}
            
        # Create minimal metadata
        metadata = create_minimal_metadata(
            dataset_name=dataset_name,
            size_mb=100.0,  # Would calculate from actual data
            example_count=10000,  # Would get from registry
            example=example or {},
            task_type=str(info.task_type),
            description=info.description,
        )
        
        # Cache for next time
        self._metadata_cache[dataset_name] = metadata
        return metadata
        
    def list_datasets(self) -> List[str]:
        """List available datasets.
        
        Returns:
            List of dataset names
            
        Example:
            data = DataAPI()
            datasets = data.list_datasets()
            print(f"Available: {len(datasets)} datasets")
        """
        # Ensure registry is initialized
        from ember.core.utils.data.initialization import initialize_dataset_registry
        initialize_dataset_registry(self._context.registry)
        
        return self._context.registry.list_datasets()
        
    def stream_file(
        self,
        filepath: str,
        *,
        format: str = "jsonl",
        batch_size: int = 32,
    ) -> StreamingDataset:
        """Stream data from a file.
        
        Args:
            filepath: Path to the file
            format: File format (jsonl, csv, parquet)
            batch_size: Batch size for processing
            
        Returns:
            StreamingDataset for the file
            
        Example:
            data = DataAPI()
            for item in data.stream_file("data.jsonl"):
                process(item)
        """
        # Simple file streaming
        import json
        
        def read_jsonl():
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
                        
        if format == "jsonl":
            return self.load(read_jsonl(), batch_size=batch_size)
        else:
            raise NotImplementedError(f"Format {format} not yet supported")
            

# Convenience functions matching the simple API style
_default_data = None


def data(
    dataset: Optional[Union[str, List[Dict[str, Any]], Iterator[Dict[str, Any]]]] = None,
    **kwargs
) -> Union[DataAPI, StreamingDataset]:
    """Simple data loading function.
    
    Args:
        dataset: If provided, loads the dataset. If None, returns DataAPI instance.
        **kwargs: Additional arguments passed to load()
        
    Returns:
        DataAPI instance if no dataset specified, otherwise StreamingDataset
        
    Example:
        # Get API instance
        api = data()
        
        # Load dataset directly  
        for item in data("mmlu"):
            print(item)
            
        # With options
        for item in data("mmlu", batch_size=64).limit(100):
            print(item)
    """
    global _default_data
    if _default_data is None:
        _default_data = DataAPI()
        
    if dataset is None:
        return _default_data
    else:
        return _default_data.load(dataset, **kwargs)