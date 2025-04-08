"""Dataset caching system.

This module provides multi-level caching for datasets to improve
performance when repeatedly accessing the same data.

Example:
    from ember.core.utils.data.cache import DatasetCache, cached

    # Create memory cache
    cache = DatasetCache()

    # Add items
    cache.set("key1", [1, 2, 3])

    # Get items
    items = cache.get("key1")

    # Use decorator for automatic caching
    @cached(ttl_seconds=3600)
    def get_items(dataset_name):
        # ... expensive loading ...
        return items
"""

from ember.core.utils.data.cache.cache_manager import DatasetCache, cached

__all__ = [
    "DatasetCache",
    "cached",
]
