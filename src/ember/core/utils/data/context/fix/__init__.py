"""Thread-safe data context integration.

This module provides a clean API for loading datasets that integrates 
with the context system and properly handles thread isolation.
"""

import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Union

from ember.core.context.ember_context import EmberContext
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.context.data_context import DataContext, get_default_context

logger = logging.getLogger(__name__)


def load_dataset(
    name: str,
    *,
    streaming: bool = True,
    limit: Optional[int] = None,
    transformers: Optional[List[Any]] = None,
    config: Optional[Any] = None,
    context: Optional[Union[EmberContext, DataContext]] = None,
    **kwargs,
) -> Union[List[DatasetEntry], Iterator[Any]]:
    """Load a dataset with thread safety and context awareness.

    This provides a simple, unified interface for loading datasets
    that properly respects thread isolation.

    Args:
        name: Dataset name
        streaming: Whether to use streaming mode
        limit: Maximum number of samples
        transformers: Optional transformers to apply
        config: Optional dataset configuration
        context: Optional context (uses thread-local default if not provided)
        **kwargs: Additional dataset options

    Returns:
        Either a list of dataset entries or a streaming iterator

    Raises:
        ValueError: If dataset not found
    """
    # Resolve context
    data_context = None
    if context is None:
        # Use default context
        data_context = get_default_context()
    elif isinstance(context, EmberContext):
        # Create from EmberContext
        data_context = DataContext.create_from_ember_context(context)
    else:
        # Use provided DataContext
        data_context = context

    # Get dataset entry
    dataset_entry = data_context.registry.get(name=name)
    if not dataset_entry:
        available = data_context.registry.list_datasets()
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")

    # Use streaming mode (memory-efficient)
    if streaming:
        # Create streaming dataset
        stream = data_context.get_streaming_dataset(
            name=name, transformers=transformers, **kwargs
        )

        # Apply limit if specified
        if limit is not None:
            stream = stream.limit(limit)

        return stream

    # Use eager loading mode
    return data_context.dataset_service.load_and_prepare(
        dataset_info=dataset_entry.info,
        prepper=dataset_entry.prepper,
        config=config,
        num_samples=limit,
    )
