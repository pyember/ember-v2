"""Transparent proxy for thread-local registry access.

This module provides backward compatibility for code using the global
DATASET_REGISTRY by proxying to the thread-local registry.
"""

from typing import Any

from ember.core.context.ember_context import current_context


class RegistryProxy:
    """Transparent proxy to registry in current context.

    This class maintains backward compatibility with code that
    imports and uses the global DATASET_REGISTRY.

    Operations are proxied to the thread-local registry in the
    current EmberContext, ensuring thread safety and proper isolation.
    """

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to registry in current context.

        Args:
            name: Attribute name

        Returns:
            Attribute from current registry

        Raises:
            AttributeError: If attribute not found
        """
        # Get context registry
        try:
            context = current_context()
            registry = context.dataset_registry
            return getattr(registry, name)
        except (AttributeError, KeyError):
            # If context not available, fall back to original registry
            from ember.core.utils.data.registry import (
                DATASET_REGISTRY as original_registry,
            )

            return getattr(original_registry, name)


def install_registry_proxy() -> None:
    """Install registry proxy to replace global DATASET_REGISTRY.

    This function replaces the global DATASET_REGISTRY with a thread-local
    proxy that forwards to the registry in the current EmberContext.
    """
    import sys

    # Create registry proxy
    proxy = RegistryProxy()

    # Replace global registry with proxy
    sys.modules["ember.core.utils.data.registry"].DATASET_REGISTRY = proxy
