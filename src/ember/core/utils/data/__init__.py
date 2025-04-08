"""Data utilities for the Ember framework.

This package provides utilities for working with datasets, including:

- Dataset registry for managing dataset metadata
- Dataset loaders for loading datasets from various sources
- Dataset preppers for preparing datasets for use with Ember
- Dataset transformers for transforming datasets
- Dataset validation utilities
- Streaming dataset processing for memory efficiency
- Caching for improved performance
- Context integration for dependency management

Typical usage:

```python
# Basic data loading with streaming support
from ember.core.utils.data import load_dataset
entries = load_dataset("mmlu", limit=10)
for entry in entries:
    print(entry.query)

# Advanced usage with context integration
from ember.core.utils.data.context.data_context import get_default_context
data_context = get_default_context()
registry = data_context.registry
print(f"Available datasets: {registry.list_datasets()}")
```
"""

# Basic type imports
from typing import Any, Dict, Iterator, List, Optional, Type, Union

# Import context elements first
from ember.core.utils.data.context.data_context import (
    DataConfig,
    DataContext,
    get_default_context,
    reset_default_context,
    set_default_context,
)

# Load context integration to ensure EmberContext knows about the data system
import ember.core.utils.data.context_integration

# Import basic models needed for type annotations
from ember.core.utils.data.base.models import DatasetEntry


# Add unified loading function
def load_dataset(
    name: str,
    *,
    config: Optional[Any] = None,
    streaming: Optional[bool] = None,
    limit: Optional[int] = None,
    transformers: Optional[List[Any]] = None,
    context: Optional[DataContext] = None,
    **kwargs,
) -> Union[List[DatasetEntry], Iterator[Any]]:
    """Load dataset entries using the high-level API.

    This function provides a clean, thread-safe way to load datasets.

    Args:
        name: Unique identifier for the dataset.
        config: Optional dataset configuration
        streaming: Whether to use streaming mode (defaults to True)
        limit: Optional maximum number of samples to retrieve.
        transformers: Optional list of transformers to apply
        context: Optional data context to use (uses default if not provided)
        **kwargs: Additional dataset options

    Returns:
        Either a list of dataset entries (eager mode) or streaming iterator.

    Raises:
        ValueError: If the dataset is not found.

    Examples:
        # Basic streaming usage
        for entry in load_dataset("mmlu", limit=10):
            print(entry.query)

        # Non-streaming with specific configuration
        entries = load_dataset("mmlu", streaming=False, config={"subset": "physics"})
    """
    # Get or create DataContext
    data_context = context or get_default_context()

    # Use the context's load_dataset method
    return data_context.load_dataset(
        name=name,
        config=config,
        streaming=streaming,
        limit=limit,
        transformers=transformers,
        **kwargs,
    )


# Import basic types that don't have dependencies
from ember.core.utils.data.base.config import BaseDatasetConfig

# Import component interfaces
from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader, IDatasetLoader
from ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.base.samplers import DatasetSampler, IDatasetSampler
from ember.core.utils.data.base.transformers import IDatasetTransformer
from ember.core.utils.data.base.validators import DatasetValidator, IDatasetValidator

# Import utilities - dependencies on other components
from ember.core.utils.data.cache.cache_manager import DatasetCache

# Import core classes and functions
from ember.core.utils.data.initialization import initialize_dataset_registry
from ember.core.utils.data.loader_factory import DatasetLoaderFactory

# Import registry system - dependency on context
from ember.core.utils.data.registry import DATASET_REGISTRY, DatasetRegistry, register
from ember.core.utils.data.service import DatasetService
from ember.core.utils.data.streaming.dataset import StreamingDataset


def load_dataset_entries(
    *,
    dataset_name: str,
    config: Union[str, BaseDatasetConfig, None] = None,
    num_samples: Optional[int] = None,
    context: Optional[DataContext] = None,
) -> List[DatasetEntry]:
    """Load and prepare dataset entries using a high-level one-liner API.

    This function provides a fast-track method to load and process dataset entries
    with minimal boilerplate. The pipeline executes the following steps:
      1. Gets or creates a DataContext for dependency management.
      2. Retrieves the dataset metadata and its associated prepper class.
      3. Uses the DataContext's service to load and prepare dataset entries.

    Example:
        from ember.core.utils.data import load_dataset_entries
        entries = load_dataset_entries(
            dataset_name="mmlu",
            config={"config_name": "abstract_algebra", "split": "dev"},
            num_samples=5,
            context=data_context
        )

    Args:
        dataset_name: Unique identifier for the dataset.
        config: Optional configuration specifying dataset details.
        num_samples: Optional maximum number of samples to retrieve.
        context: DataContext to use (required).

    Returns:
        List of dataset entries after processing.

    Raises:
        ValueError: If context is None or dataset not found.
    """
    if context is None:
        raise ValueError(
            "DataContext must be explicitly provided to load_dataset_entries."
        )
    data_context = context

    # Get registry and loader factory from context
    registry = data_context.registry
    loader_factory = data_context.loader_factory

    # Retrieve dataset metadata from the registry
    dataset = registry.get(name=dataset_name)
    if not dataset or not dataset.info:
        available = registry.list_datasets()
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry. Available: {available}"
        )

    dataset_info = dataset.info

    # Use existing prepper if available, otherwise look up prepper class
    if dataset.prepper:
        prepper = dataset.prepper
    else:
        prepper_class: Optional[
            Type[IDatasetPrepper]
        ] = loader_factory.get_prepper_class(dataset_name=dataset_name)
        if prepper_class is None:
            raise ValueError(
                f"Prepper for dataset '{dataset_name}' not found in loader factory."
            )
        # Instantiate the prepper using the provided configuration
        prepper: IDatasetPrepper = prepper_class(config=config)

    # Use the DataContext's service to load and prepare dataset entries
    return data_context.dataset_service.load_and_prepare(
        dataset_info=dataset_info,
        prepper=prepper,
        config=config,
        num_samples=num_samples,
    )
