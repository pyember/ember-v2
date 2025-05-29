"""Resilient execution with configurable retry policies.

Provides type-safe retry strategies for handling transient failures.

Example:
    >>> from ember.core.utils.retry_utils import run_with_backoff
    >>> response = run_with_backoff(
    ...     requests.get,
    ...     "https://api.example.com/data",
    ...     headers={"Authorization": "Bearer token"}
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ImportError as err:
    raise ImportError(
        "tenacity is required for retry_utils. Install via Poetry: poetry add tenacity"
    ) from err

T = TypeVar("T")


class IRetryStrategy(ABC, Generic[T]):
    """Abstract base class for retry strategies.

    Defines contract for retry policy implementations with specific
    backoff algorithms and exception handling.
    """

    @abstractmethod
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a callable with retry policy.

        Args:
            func: Callable to execute.
            *args: Positional arguments for callable.
            **kwargs: Keyword arguments for callable.

        Returns:
            Result of callable execution.

        Raises:
            Exception: Last exception if all retries fail.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ExponentialBackoffStrategy(IRetryStrategy[T]):
    """Retry strategy using randomized exponential backoff.

    Implements jittered exponential backoff to avoid thundering herd.

    Attributes:
        min_wait: Minimum wait time in seconds.
        max_wait: Maximum wait time in seconds.
        max_attempts: Maximum retry attempts.
    """

    def __init__(
        self, min_wait: int = 1, max_wait: int = 60, max_attempts: int = 3
    ) -> None:
        """Initialize ExponentialBackoffStrategy.

        Args:
            min_wait: Minimum wait time in seconds.
            max_wait: Maximum wait time in seconds.
            max_attempts: Maximum retry attempts.
        """
        self.min_wait: int = min_wait
        self.max_wait: int = max_wait
        self.max_attempts: int = max_attempts

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute callable with exponential backoff retries.

        Args:
            func: Callable to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result from successful execution.

        Raises:
            Exception: Last exception if all retries fail.
        """

        # Creating a wrapped function with tenacity retry configuration
        @retry(
            wait=wait_random_exponential(min=self.min_wait, max=self.max_wait),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True)
        def wrapped() -> T:
            return func(*args, **kwargs)

        return wrapped()


# Singleton instance with default configuration for common use cases
_default_strategy: ExponentialBackoffStrategy[Any] = ExponentialBackoffStrategy()


def run_with_backoff(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute function with retry protection.

    Args:
        func: Function to execute.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of successful execution.

    Example:
        >>> response = run_with_backoff(
        ...     requests.get,
        ...     "https://api.example.com/data",
        ...     headers={"Authorization": "Bearer token"}
        ... )
    """
    return _default_strategy.execute(func, *args, **kwargs)


if __name__ == "__main__":

    def flaky_function(x: int) -> int:
        """Simulate a flaky function with random failures.

        Demonstrates retry behavior by randomly failing approximately
        half the time, simulating transient infrastructure issues.

        Args:
            x: An integer input for the computation.

        Returns:
            The computed result (x multiplied by 2).

        Raises:
            RuntimeError: On simulated random failure.
        """
        import random

        if random.random() < 0.5:
            raise RuntimeError("Simulated transient failure!")
        return x * 2

    # Implementation for demonstration purposes
