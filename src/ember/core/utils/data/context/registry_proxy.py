"""Registry proxy for backward compatibility.

Provides transparent access to the DataContext through
the existing DATASET_REGISTRY global reference.
"""

import logging
import sys
import threading
from typing import Any, Dict, List, Optional, Type

from ember.core.utils.data.context.data_context import get_default_context

logger = logging.getLogger(__name__)


class DatasetRegistryProxy:
    """Thread-safe proxy for the dataset registry.

    Forwards attribute access to the registry in the current context
    providing transparent backward compatibility.

    This implements the full Registry interface but delegates to the
    context-specific registry, ensuring thread safety and correct
    context propagation.
    """

    # Thread-local storage for method cache
    _local = threading.local()

    def __init__(self):
        """Initialize thread-local cache."""
        if not hasattr(self._local, "cache"):
            self._local.cache = {}

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the registry in the current context."""
        # Fast path: check thread-local cache first
        cache = self._get_cache()
        if name in cache:
            return cache[name]

        try:
            # Slow path: get attribute from registry in current context
            registry = get_default_context().registry

            # Force initialization of the registry if empty and trying to list_datasets
            if name == "list_datasets" and callable(getattr(registry, name)):
                datasets = registry.list_datasets()
                if not datasets:
                    # Registry might be empty - ensure it's initialized
                    from ember.core.utils.data.initialization import (
                        initialize_dataset_registry)

                    initialize_dataset_registry(metadata_registry=registry)

            attr = getattr(registry, name)

            # Cache non-private callable attributes
            if callable(attr) and not name.startswith("_"):
                # Create dynamic methods that are bound to the current context's registry
                def proxy_method(*args, **kwargs):
                    current_registry = get_default_context().registry
                    method = getattr(current_registry, name)
                    return method(*args, **kwargs)

                cache[name] = proxy_method
                return proxy_method

            return attr
        except Exception as e:
            logger.debug(f"Error accessing registry method '{name}': {e}")

            # Fallback to legacy global registry to maintain compatibility
            try:
                from ember.core.utils.data.registry import (
                    DATASET_REGISTRY as global_registry)

                if hasattr(global_registry.__class__, name):
                    original_attr = getattr(global_registry.__class__, name)
                    if original_attr:
                        return original_attr.__get__(global_registry)
            except (ImportError, AttributeError):
                pass

            # If all else fails, re-raise the exception
            raise

    def _get_cache(self) -> Dict[str, Any]:
        """Get thread-local cache with safe initialization."""
        if not hasattr(self._local, "cache"):
            self._local.cache = {}
        return self._local.cache

    # Direct implementation of core registry methods for performance

    def register(self, **kwargs) -> None:
        """Register a dataset in the current context's registry."""
        registry = get_default_context().registry
        return registry.register(**kwargs)

    def get(self, *, name: str) -> Any:
        """Get a dataset from the current context's registry."""
        registry = get_default_context().registry
        return registry.get(name=name)

    def list_datasets(self) -> List[str]:
        """List datasets from the current context's registry."""
        registry = get_default_context().registry
        datasets = registry.list_datasets()

        # Force initialization if empty
        if not datasets:
            from ember.core.utils.data.initialization import initialize_dataset_registry

            initialize_dataset_registry(metadata_registry=registry)
            datasets = registry.list_datasets()

        return datasets

    def get_info(self, *, name: str) -> Optional[Any]:
        """Get dataset info from the current context's registry."""
        registry = get_default_context().registry
        return registry.get_info(name=name)

    def register_metadata(self, *, name: str, **kwargs) -> None:
        """Register dataset metadata in the current context's registry."""
        registry = get_default_context().registry
        return registry.register_metadata(name=name, **kwargs)

    def clear(self) -> None:
        """Clear the current context's registry."""
        registry = get_default_context().registry
        return registry.clear()


def install_registry_proxy() -> None:
    """Install registry proxy to replace global DATASET_REGISTRY.

    This makes existing code continue to work with the new architecture
    by forwarding calls to the current context's registry.
    """
    try:
        # Replace the global registry with our proxy
        module = sys.modules["ember.core.utils.data.registry"]
        original_registry = module.DATASET_REGISTRY

        # Create proxy that forwards to context
        proxy = DatasetRegistryProxy()
        module.DATASET_REGISTRY = proxy

        # Also update the global import for backward compatibility
        sys.modules["ember.core.utils.data"].DATASET_REGISTRY = proxy

        logger.info("Installed registry proxy for backward compatibility")
    except Exception as e:
        logger.error(f"Failed to install registry proxy: {e}")
