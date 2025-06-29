"""PyTree registration for Ember operators in XCS.

This module ensures that Ember operators are properly registered as JAX pytrees,
preventing static array warnings and enabling full JAX transformation support.
"""

from typing import Any

import jax.tree_util as tree_util

from ember._internal.module import Module


def register_ember_pytrees():
    """Register Ember operators as JAX pytrees.

    This is called automatically on import to ensure all Ember modules
    work seamlessly with JAX transformations like vmap, pmap, etc.
    """
    # Check if already registered
    if hasattr(register_ember_pytrees, "_registered"):
        return

    # Since EmberModule inherits from equinox.Module, it's already a pytree
    # But we need to ensure proper handling of nested structures

    # Register any custom operator types that don't inherit from Module
    # (Currently all operators should inherit from Module, but this is extensible)

    # Mark as registered
    register_ember_pytrees._registered = True


def ensure_pytree_compatible(obj: Any) -> Any:
    """Ensure an object is compatible with JAX pytree operations.

    This is a safety check for objects passed through XCS transformations.
    """
    if isinstance(obj, Module):
        # Already a pytree via equinox
        return obj

    # For other types, check if they're already registered
    try:
        tree_util.tree_flatten(obj)
        return obj
    except (TypeError, ValueError):
        # If not flattenable, wrap in a simple container
        return StaticWrapper(obj)


class StaticWrapper:
    """Wrapper for non-pytree objects to make them JAX-compatible."""

    def __init__(self, value):
        """Initialize StaticWrapper with a value.

        Args:
            value: The non-pytree object to wrap.
        """
        self.value = value

    def tree_flatten(self):
        """Flatten the wrapper for JAX pytree operations.

        Returns:
            Tuple of (leaves, treedef) where leaves is an empty list
            and treedef contains the static value.
        """
        return [], self.value

    @classmethod
    def tree_unflatten(cls, static, dynamic):
        """Reconstruct StaticWrapper from flattened representation.

        Args:
            static: The static value stored during flattening.
            dynamic: The dynamic leaves (empty for StaticWrapper).

        Returns:
            New StaticWrapper instance containing the static value.
        """
        return cls(static)


# Register StaticWrapper as a pytree
tree_util.register_pytree_node(
    StaticWrapper, StaticWrapper.tree_flatten, StaticWrapper.tree_unflatten
)


# Auto-register on import
register_ember_pytrees()
