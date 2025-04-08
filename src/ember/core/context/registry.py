"""Thread-local registry system for high-performance component discovery.

This registry forms the backbone of Ember's context system, providing a
zero-overhead mechanism for components to locate each other across the framework.
The design prioritizes performance, thread safety, and minimal memory overhead
while maintaining simplicity and robustness.

Implementation follows mechanical sympathy principles to minimize CPU cache misses
and ensure optimal performance in high-throughput scenarios.
"""

import threading
from typing import Any, Dict, List, Optional, TypeVar

T = TypeVar("T")  # Component type


class Registry:
    """Thread-isolated registry enabling zero-overhead component discovery.

    The Registry is the sole global abstraction in Ember, providing a
    thread-local store that isolates component registration between threads.
    This eliminates contention in multi-threaded environments while maintaining
    a simple API.

    Performance characteristics:
    - O(1) lookup complexity for component retrieval
    - Zero lock contention on read paths
    - Thread isolation prevents cross-thread interference
    - Cache-friendly flat structure minimizes pointer chasing
    - Reentrant locks ensure thread safety during modifications

    Usage example:
        # Get the registry for the current thread
        registry = Registry.current()

        # Register a component
        registry.register("config_manager", config_manager)

        # Retrieve a component
        config = registry.get("config_manager")
    """

    # Thread-local storage - each thread gets its own registry instance
    _thread_local = threading.local()

    @classmethod
    def current(cls) -> "Registry":
        """Retrieves the current thread's registry, creating it if needed.

        This is the primary entry point for accessing the registry. It ensures
        each thread has its own isolated registry instance without any cross-thread
        interference. Creates a new registry automatically if one doesn't exist.

        Returns:
            Registry: The current thread's registry instance.
        """
        if not hasattr(cls._thread_local, "registry"):
            cls._thread_local.registry = Registry()
        return cls._thread_local.registry

    @classmethod
    def clear(cls) -> None:
        """Removes the registry from the current thread's storage.

        This is primarily used in testing and during thread recycling to ensure
        a clean state. It completely removes the registry reference rather than
        just emptying it to minimize memory usage.
        """
        if hasattr(cls._thread_local, "registry"):
            delattr(cls._thread_local, "registry")

    def __init__(self) -> None:
        """Initializes an empty registry with thread-safety mechanisms."""
        # Using dict for O(1) component lookup
        self._components: Dict[str, Any] = {}
        # Reentrant lock allows nested calls from the same thread
        self._lock = threading.RLock()

    def register(self, key: str, component: Any) -> None:
        """Registers a component by key with thread-safe locking.

        This method safely adds or replaces a component in the registry.
        The lock scope is minimized to reduce contention in high-frequency
        registration scenarios.

        Args:
            key: Unique string identifier for the component.
            component: The component instance to register.
        """
        with self._lock:
            self._components[key] = component

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a component by key with zero lock overhead.

        This method is optimized for the read path, which is the most common
        operation. It performs a direct dictionary lookup without acquiring
        locks, as Python's dict implementation is thread-safe for reads.

        Args:
            key: The identifier for the component to retrieve.

        Returns:
            The registered component or None if not found.
        """
        return self._components.get(key)

    def unregister(self, key: str) -> bool:
        """Removes a component from the registry.

        Thread-safe removal with minimal lock contention. Useful for cleanup
        operations and dynamically replacing components.

        Args:
            key: The identifier of the component to remove.

        Returns:
            bool: True if the component was found and removed, False otherwise.
        """
        with self._lock:
            if key in self._components:
                del self._components[key]
                return True
            return False

    def keys(self) -> List[str]:
        """Returns all registered component keys.

        This method creates a new list to avoid exposing the internal dictionary
        keys view, which could lead to concurrent modification issues.

        Returns:
            List[str]: A list of all registered component identifiers.
        """
        return list(self._components.keys())
