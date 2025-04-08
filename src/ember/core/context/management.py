"""Context management utilities.

This module provides utilities for managing registry scope and isolation.
"""

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, TypeVar

from .registry import Registry

T = TypeVar("T")


@contextmanager
def scoped_registry() -> Iterator[Registry]:
    """Create temporary registry that restores previous on exit.

    This is useful for tests and isolated operations that shouldn't
    affect the rest of the application.

    Yields:
        Temporary registry for the scope
    """
    # Save current registry
    cls = Registry
    has_previous = hasattr(cls._thread_local, "registry")
    previous = getattr(cls._thread_local, "registry", None)

    try:
        # Create new registry
        cls._thread_local.registry = Registry()
        yield cls._thread_local.registry
    finally:
        # Restore previous registry
        if has_previous:
            cls._thread_local.registry = previous
        else:
            delattr(cls._thread_local, "registry")


def with_registry(f: Callable[..., T]) -> Callable[..., T]:
    """Decorator to run a function with its own registry.

    This decorator creates a temporary registry for the duration
    of the function call, then restores the previous registry.

    Args:
        f: Function to wrap

    Returns:
        Wrapped function
    """

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        with scoped_registry():
            return f(*args, **kwargs)

    return wrapper


@contextmanager
def temp_component(key: str, component: Any) -> Iterator[Any]:
    """Temporarily register a component in the current registry.

    Args:
        key: Component identifier
        component: Component instance

    Yields:
        The component instance
    """
    registry = Registry.current()
    previous = registry.get(key)
    had_previous = key in registry._components

    try:
        registry.register(key, component)
        yield component
    finally:
        if had_previous:
            registry.register(key, previous)
        else:
            registry.unregister(key)


def seed_registry(components: Dict[str, Any]) -> None:
    """Seed the current registry with multiple components.

    This is useful for testing and initialization.

    Args:
        components: Dictionary mapping keys to component instances
    """
    registry = Registry.current()
    for key, component in components.items():
        registry.register(key, component)
