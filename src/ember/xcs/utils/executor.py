"""Adaptive execution system for computational graphs.

Provides a unified execution framework that automatically selects optimal
parallelization strategies based on workload characteristics. The system routes
operations to thread-based or async-based executors for maximum throughput.
"""

import asyncio
import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from ember.xcs.utils.execution_analyzer import ExecutionTracker

T = TypeVar("T")  # Input type
U = TypeVar("U")  # Output type
logger = logging.getLogger(__name__)


@runtime_checkable
class Executor(Protocol[T, U]):
    """Protocol for executors that run batched tasks."""

    def execute(self, fn: Callable[[T], U], inputs: List[T]) -> List[U]:
        """Execute function across inputs."""
        ...


class ThreadExecutor(Executor[Dict[str, Any], Any]):
    """Thread-based parallel executor optimized for mixed workloads.

    Efficiently executes tasks that combine computation and I/O operations
    using a managed thread pool. Designed for moderate concurrency needs with
    predictable resource utilization.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        fail_fast: bool = True,
    ):
        """Initialize thread executor with execution parameters.

        Args:
            max_workers: Maximum thread count (None uses CPU count)
            timeout: Operation timeout in seconds
            fail_fast: If True, propagates first error; otherwise continues
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.fail_fast = fail_fast
        self._executor = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """Create or return thread pool with lazy initialization."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def execute(self, fn: Callable, inputs: List[Dict[str, Any]]) -> List[Any]:
        """Execute function across inputs in parallel.

        Args:
            fn: Function to execute (takes inputs keyword argument)
            inputs: List of input dictionaries to process

        Returns:
            List of results in same order as inputs
        """
        executor = self._get_executor()

        # Submit all tasks with proper closure binding
        futures = [
            executor.submit(lambda input_dict=input_dict: fn(inputs=input_dict))
            for input_dict in inputs
        ]

        # Collect results preserving input order
        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.timeout)
                results.append(result)
            except Exception as e:
                if not self.fail_fast:
                    logger.warning("Thread execution error: %s", e)
                    results.append(None)
                else:
                    raise

        return results

    def close(self) -> None:
        """Release thread pool resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


class AsyncExecutor(Executor[Dict[str, Any], Any]):
    """Asyncio-based executor for high-concurrency I/O workloads.

    Optimized for handling hundreds of concurrent API calls with precise
    concurrency limits. Ideal for LLM API calls, network operations, and
    other I/O-bound tasks that benefit from non-blocking execution.
    """

    def __init__(
        self,
        max_concurrency: int = 20,
        timeout: Optional[float] = None,
        fail_fast: bool = True,
    ):
        """Initialize async executor with execution parameters.

        Args:
            max_concurrency: Maximum concurrent operations
            timeout: Operation timeout in seconds
            fail_fast: If True, propagates first error; otherwise continues
        """
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.fail_fast = fail_fast

    def execute(self, fn: Callable, inputs: List[Dict[str, Any]]) -> List[Any]:
        """Execute function across inputs using asyncio.

        Args:
            fn: Function to execute (coroutine or regular function)
            inputs: List of input dictionaries to process

        Returns:
            List of results in same order as inputs
        """
        return self._run_async_gather(fn, inputs)

    def _run_async_gather(
        self, fn: Callable, inputs: List[Dict[str, Any]]
    ) -> List[Any]:
        """Executes tasks concurrently with controlled parallelism."""

        async def _gather_with_concurrency_control():
            # Control concurrent operations with semaphore
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _process_single_item(input_dict: Dict[str, Any]) -> Any:
                """Process one item with concurrency limit."""
                async with semaphore:
                    # Handle both coroutine and regular functions
                    if inspect.iscoroutinefunction(fn):
                        return await fn(inputs=input_dict)

                    # Execute sync functions in thread pool
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None, lambda: fn(inputs=input_dict)
                    )

            # Create tasks for all inputs
            tasks = [_process_single_item(input_dict) for input_dict in inputs]

            # Execute with optional timeout
            try:
                if self.timeout:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.timeout,
                    )
                else:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                logger.error("Async execution timed out after %s seconds", self.timeout)
                raise

            # Process results with error handling
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    if self.fail_fast:
                        raise result
                    logger.warning("Async execution error: %s", result)
                    processed_results.append(None)
                else:
                    processed_results.append(result)

            return processed_results

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(_gather_with_concurrency_control())

    def close(self) -> None:
        """Clean up any async resources (none needed)."""
        pass


class Dispatcher:
    """Smart workload router for optimal parallel execution.

    Routes functions to the most appropriate execution engine based on
    their characteristics. Automatically selects between thread-based and
    async-based execution to maximize throughput and resource utilization.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        fail_fast: bool = True,
        executor: str = "auto",
    ):
        """Initialize dispatcher with execution configuration.

        Args:
            max_workers: Maximum parallel operations
            timeout: Operation timeout in seconds
            fail_fast: Whether to propagate errors immediately
            executor: Execution mode ("auto", "async", or "thread")
        """
        valid_executors = {"auto", "async", "thread"}
        if executor not in valid_executors:
            raise ValueError(
                f"Invalid executor: {executor}. "
                f"Must be one of: {', '.join(valid_executors)}"
            )

        self.max_workers = max_workers
        self.timeout = timeout
        self.fail_fast = fail_fast
        self.executor = executor

        # Lazy-initialized executors
        self._thread_executor = None
        self._async_executor = None

    def _get_thread_executor(self) -> ThreadExecutor:
        """Create thread executor on first use."""
        if self._thread_executor is None:
            self._thread_executor = ThreadExecutor(
                max_workers=self.max_workers,
                timeout=self.timeout,
                fail_fast=self.fail_fast,
            )
        return self._thread_executor

    def _get_async_executor(self) -> AsyncExecutor:
        """Create async executor on first use."""
        if self._async_executor is None:
            self._async_executor = AsyncExecutor(
                max_concurrency=self.max_workers or 20,
                timeout=self.timeout,
                fail_fast=self.fail_fast,
            )
        return self._async_executor

    def _select_executor(self, fn: Callable) -> Executor:
        """Select optimal executor for a function.

        Analyzes function characteristics to determine if it's better suited
        for thread-based or async-based execution. Uses both static analysis
        and runtime performance data for accurate decisions.

        Args:
            fn: Function to analyze

        Returns:
            Most appropriate executor implementation
        """
        # Honor explicit executor selection
        if self.executor == "async":
            return self._get_async_executor()
        if self.executor == "thread":
            return self._get_thread_executor()

        # Auto-selection based on function properties
        if inspect.iscoroutinefunction(fn):
            return self._get_async_executor()

        # Use runtime performance data to determine I/O vs CPU bound
        if ExecutionTracker.is_likely_io_bound(fn):
            return self._get_async_executor()

        # Default to thread executor for mixed and CPU-bound operations
        return self._get_thread_executor()

    def map(self, fn: Callable, inputs_list: List[Dict[str, Any]]) -> List[Any]:
        """Execute function across multiple inputs in parallel.

        Automatically routes execution to optimal engine based on function
        characteristics. Records performance metrics to improve future routing
        decisions.

        Args:
            fn: Function to execute
            inputs_list: List of input dictionaries

        Returns:
            List of results in same order as inputs
        """
        if not inputs_list:
            return []

        # Record execution metrics for adaptive optimization
        start_time = time.time()
        start_cpu = time.process_time()

        try:
            # Select and use appropriate executor
            executor = self._select_executor(fn)
            return executor.execute(fn, inputs_list)
        finally:
            # Update metrics to improve future executor selection
            cpu_time = time.process_time() - start_cpu
            wall_time = time.time() - start_time

            # Only update metrics when we have actual work
            if inputs_list:
                ExecutionTracker.update_metrics(fn, cpu_time, wall_time)

    def close(self) -> None:
        """Release all executor resources."""
        if self._thread_executor:
            self._thread_executor.close()
        if self._async_executor:
            self._async_executor.close()


# For compatibility with existing code that may have imported these names
TaskExecutor = Dispatcher
ExecutionCoordinator = Dispatcher
ThreadedStrategy = ThreadExecutor
AsyncStrategy = AsyncExecutor
