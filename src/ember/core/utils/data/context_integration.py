"""Integration between data system and EmberContext.

Adds data-specific functionality to EmberContext to provide
unified access to dataset operations in a thread-safe manner.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, TypeVar, Union

from ember.core.context.ember_context import EmberContext
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.cache.cache_manager import DatasetCache
from ember.core.utils.data.context.data_context import DataContext
from ember.core.utils.data.context.registry_proxy import install_registry_proxy
from ember.core.utils.data.registry import DatasetRegistry

logger = logging.getLogger(__name__)
T = TypeVar("T")


def integrate_data_context() -> None:
    """Integrate data system with EmberContext.

    Adds data-specific components to EmberContext and installs
    backward compatibility proxies.

    This creates a bidirectional connection between EmberContext and
    DataContext, ensuring proper isolation and thread safety.
    """
    try:
        # Add data_context property to EmberContext
        def get_data_context(self) -> DataContext:
            """Get data context for EmberContext.

            Creates context on first access and caches it.

            Returns:
                Data context associated with this EmberContext
            """
            # Fast path: check if already cached using thread-safe access pattern
            # Use getattr with default to avoid AttributeError
            data_ctx = getattr(self, "_data_context", None)
            if data_ctx is None:
                # Only lock for initialization - single writer pattern
                with self._lock:
                    # Double-check after acquiring lock
                    if not hasattr(self, "_data_context") or self._data_context is None:
                        try:
                            # Create new context for this EmberContext
                            data_ctx = DataContext.create_from_ember_context(self)
                            # Store safely with atomic assignment - use setattr instead of direct assignment
                            setattr(self, "_data_context", data_ctx)
                        except Exception as e:
                            logger.error(f"Error creating data context: {e}")
                            # Create minimal context if creation fails
                            data_ctx = DataContext()
                            setattr(self, "_data_context", data_ctx)
                    else:
                        # Another thread created it while we were waiting
                        data_ctx = self._data_context
            else:
                # Fast path succeeded, use cached instance
                pass

            return data_ctx

        # Add thread-local registry access to EmberContext
        def get_dataset_registry(self) -> DatasetRegistry:
            """Get dataset registry for context.

            This provides access to the registry in the data context.

            Returns:
                Dataset registry
            """
            return self.data_context.registry

        # Add dataset cache to EmberContext
        def get_dataset_cache(self) -> DatasetCache:
            """Get dataset cache for context.

            Returns:
                Dataset cache
            """
            return self.data_context.cache_manager

        # Add convenience method for loading datasets with maximum flexibility
        def load_dataset(
            self,
            name: str,
            *,
            config: Optional[Any] = None,
            streaming: Optional[bool] = None,
            limit: Optional[int] = None,
            transformers: Optional[List[Any]] = None,
            **kwargs,
        ) -> Union[List[DatasetEntry], Iterator[DatasetEntry]]:
            """Load dataset with current context.

            Args:
                name: Name of dataset to load
                config: Optional dataset configuration
                streaming: Whether to use streaming mode
                limit: Optional sample limit
                transformers: Optional list of transformers to apply
                **kwargs: Additional dataset options

            Returns:
                Dataset entries or streaming iterator
            """
            # Simply forward to data_context's load_dataset method
            return self.data_context.load_dataset(
                name=name,
                config=config,
                streaming=streaming,
                limit=limit,
                transformers=transformers,
                **kwargs,
            )

        # Add properties to EmberContext using descriptor protocol
        EmberContext.data_context = property(get_data_context)
        EmberContext.dataset_registry = property(get_dataset_registry)
        EmberContext.dataset_cache = property(get_dataset_cache)
        EmberContext.load_dataset = load_dataset

        # Install registry proxy for backward compatibility
        install_registry_proxy()

        logger.info("Data context integration complete")
    except Exception as e:
        logger.error(f"Failed to integrate data context: {e}")


# Auto-integrate when imported
integrate_data_context()


# Import all context components at module scope for cleaner imports
__all__ = [
    "DataContext",
]
