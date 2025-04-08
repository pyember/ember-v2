"""Streaming dataset implementation.

This module provides memory-efficient dataset iterators that process
datasets with O(1) memory usage regardless of dataset size.

Example:
    from ember.core.utils.data.streaming import StreamingDataset

    # Stream items with minimal memory usage
    for item in StreamingDataset("mmlu"):
        process(item)

    # Apply transformations
    from ember.core.utils.data.base.transformers import FunctionTransformer
    transformer = FunctionTransformer(lambda x: {"transformed": x})

    dataset = StreamingDataset(
        "mmlu",
        transformers=[transformer],
        batch_size=32,
    )

    # Limit the number of items
    for item in dataset.limit(10):
        print(item)
"""

from ember.core.utils.data.streaming.dataset import (
    StreamingDataset,
    StreamingTransformer,
)

__all__ = [
    "StreamingDataset",
    "StreamingTransformer",
]
