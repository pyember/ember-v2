"""Thread-local execution context for computational graphs.

Provides thread-isolated configuration with zero-cost read paths and copy-on-write
semantics. Designed for high-performance operator execution in XCS pipelines.
"""

import contextlib
import threading
import types
from typing import Dict, Iterator, Optional, TypeVar

from ember.xcs.api.types import ContextDict, OptionValue

T = TypeVar("T")


class ExecutionContext:
    """Thread-local context system with hierarchical option inheritance.

    Provides a thread-safe configuration mechanism with the following properties:
    - Thread isolation using thread-local storage
    - Context nesting with proper option inheritance
    - Immutable context objects for safe sharing across threads
    - Zero-allocation on read paths
    """

    _thread_local = threading.local()
    _EMPTY_CONTEXT: ContextDict = types.MappingProxyType({})

    @classmethod
    def _initialize(cls) -> None:
        """Initialize thread-local state.

        Sets up the thread-local storage with default values.
        Called lazily only when needed.
        """
        cls._thread_local.stack = [cls._EMPTY_CONTEXT]
        cls._thread_local.buffer = {}

    @classmethod
    def current(cls) -> ContextDict:
        """Get current context dictionary.

        Returns:
            An immutable mapping for the current context.
            Creates default context if none exists in this thread.
        """
        if not hasattr(cls._thread_local, "stack"):
            cls._initialize()
        return cls._thread_local.stack[-1]

    @classmethod
    @contextlib.contextmanager
    def options(cls, **kwargs: OptionValue) -> Iterator[None]:
        """Set context options within a lexical scope.

        Creates a new context inheriting from the current one with the
        provided options added. Automatically restores the previous context
        when exiting the scope.

        Args:
            **kwargs: Options to set in this context scope

        Yields:
            None
        """
        # Fast path for empty options - avoid allocations
        if not kwargs:
            yield
            return

        # Initialize thread-local state if needed
        if not hasattr(cls._thread_local, "stack"):
            cls._initialize()

        # Use pre-allocated buffer to create new context
        buffer = cls._thread_local.buffer
        buffer.clear()
        buffer.update(cls.current())
        buffer.update(kwargs)

        # Create immutable view and push to context stack
        # Copy to ensure buffer can be modified while context is active
        immutable_context: ContextDict = types.MappingProxyType(buffer.copy())
        cls._thread_local.stack.append(immutable_context)

        try:
            yield
        finally:
            # Restore previous context
            if len(cls._thread_local.stack) > 1:
                cls._thread_local.stack.pop()

    @classmethod
    def get_option(cls, key: str, default: Optional[T] = None) -> OptionValue:
        """Get option value from current context.

        Retrieves a named option from the current context or returns
        the default value if not found.

        Args:
            key: Option name
            default: Default value if option not found

        Returns:
            Option value or default
        """
        return cls.current().get(key, default)

    @classmethod
    def get_all_options(cls) -> Dict[str, OptionValue]:
        """Get all options as a dictionary.

        Returns:
            Mutable copy of all current context options
        """
        return dict(cls.current())

    @classmethod
    def reset(cls) -> None:
        """Reset the thread-local context to initial state.

        Useful for testing and to ensure a clean context state.
        """
        if hasattr(cls._thread_local, "stack"):
            cls._thread_local.stack = [cls._EMPTY_CONTEXT]

        if hasattr(cls._thread_local, "buffer"):
            cls._thread_local.buffer = {}


# Compatibility layer
def get_execution_context() -> ContextDict:
    """Get current thread's execution context dictionary.

    Returns:
        Current execution context as an immutable mapping
    """
    return ExecutionContext.current()


def set_execution_options(**options: OptionValue) -> None:
    """Set options in current execution context.

    This is primarily for backward compatibility. Options are only active during
    the function call which immediately returns.

    Args:
        **options: Context options to set
    """
    with ExecutionContext.options(**options):
        pass


@contextlib.contextmanager
def execution_scope(**options: OptionValue) -> Iterator[None]:
    """Context manager for setting temporary execution options.

    Args:
        **options: Context options to set

    Yields:
        None
    """
    with ExecutionContext.options(**options):
        yield
