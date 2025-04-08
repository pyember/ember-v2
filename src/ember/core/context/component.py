"""Base component abstraction for Ember's thread-safe dependency system.

This module defines the Component base class that all Ember components inherit from.
It implements an optimized lazy initialization pattern with thread-safe semantics,
self-registration, and automatic dependency resolution through the registry system.

The design reflects core principles of mechanical sympathy and SOLID design:
- Single Responsibility: Each component focuses on one aspect of functionality
- Open/Closed: Specialized via inheritance without modifying base behavior
- Dependency Inversion: Components depend on abstractions not implementations

Implementation ensures minimal contention in multi-threaded environments with
double-checked locking patterns and thread-local component isolation.
"""

import threading
from typing import ClassVar, Optional, Type, TypeVar

from .registry import Registry

# Type variable bound to Component for correct typing of factory methods
T = TypeVar("T", bound="Component")


class Component:
    """Thread-safe base class for self-registering components with lazy initialization.

    The Component abstraction provides a foundation for all Ember subsystems with
    these key characteristics:

    1. Registry integration: Each component registers in a thread-local registry
    2. Lazy initialization: Resources allocated only when first accessed
    3. Thread safety: Double-checked locking minimizes contention
    4. Self-discovery: Components find each other through the registry
    5. Singleton per registry: Only one instance per component type in a registry

    This pattern enables a highly modular system with zero-overhead component
    discovery and minimal resource consumption. Components initialize only when
    needed and only once per thread, providing optimal resource usage.

    Example usage:
        # Define custom component
        class DataLoader(Component):
            def _register(self) -> None:
                self._registry.register("data_loader", self)

            def _initialize(self) -> None:
                # Load resources, connect to databases, etc.
                self.connection = create_database_connection()

            def load_dataset(self, name: str) -> Dataset:
                self._ensure_initialized()  # Lazy initialization
                return self.connection.query_dataset(name)

        # Use component (auto-creates if needed)
        loader = DataLoader.get()
        dataset = loader.load_dataset("mnist")
    """

    # Class-level type registry to optimize memory usage
    _component_types: ClassVar[dict] = {}

    def __init__(self, registry: Optional[Registry] = None, register: bool = True):
        """Initializes component with thread-safe registration.

        Integrates with the registry system and sets up lazy initialization,
        but defers resource allocation until first use.

        Args:
            registry: Component registry to use. If None, uses the current
                thread's registry, ensuring thread isolation.
            register: Whether to automatically register this component in the
                registry. Set to False for temporary components or testing.
        """
        # Use provided registry or fallback to thread-local registry
        self._registry = registry or Registry.current()

        # Lazy initialization tracking
        self._initialized = False

        # Thread-safety for initialization
        self._lock = threading.RLock()

        # Auto-register in registry if requested
        if register:
            self._register()

    def _register(self) -> None:
        """Registers this component in the registry.

        This is a template method that subclasses must override to define
        their registration logic. Typically, components register themselves
        by type and name.

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError(
            f"Component subclass {self.__class__.__name__} must implement _register()"
        )

    def _ensure_initialized(self) -> None:
        """Ensures component is initialized with minimal locking overhead.

        Uses double-checked locking pattern to minimize contention:
        1. Fast path: Check initialization flag without lock
        2. Slow path: Acquire lock and check again before initializing

        This approach minimizes lock contention while ensuring thread safety.
        """
        # Fast path: Already initialized (most common case)
        if self._initialized:
            return

        # Slow path: Need to initialize with proper locking
        with self._lock:
            # Double-check after acquiring lock (another thread may have initialized)
            if not self._initialized:
                # Call initialization logic
                self._initialize()

                # Mark as initialized (atomic writes to booleans are thread-safe)
                self._initialized = True

    def _initialize(self) -> None:
        """Performs one-time initialization of component resources.

        This is a template method that subclasses must override to implement
        their specific initialization logic. It will be called exactly once
        per component instance, and only when the component is first used.

        Implementing lazy initialization improves startup time and reduces
        resource consumption for unused components.

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError(
            f"Component subclass {self.__class__.__name__} must implement _initialize()"
        )

    @classmethod
    def get(cls: Type[T]) -> T:
        """Retrieves or creates a component singleton from the registry.

        This factory method implements the service locator pattern:
        1. Checks if component exists in the registry
        2. Creates and registers a new instance if not found

        This provides a concise way to access components while ensuring
        only one instance exists per registry (typically per thread).

        Returns:
            Component instance of the requested type.
        """
        # Get current thread's registry
        registry = Registry.current()

        # Try to get existing component
        component = registry.get(cls.__name__)

        # Create new component if not found
        if component is None:
            component = cls(registry)

        # Return typed component
        return component
