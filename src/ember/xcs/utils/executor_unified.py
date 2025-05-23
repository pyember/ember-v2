"""Next-generation unified executor for XCS - Simplified version.

This module provides the enhanced execution infrastructure that all XCS components
should migrate to. The learning is dead simple: remember what worked well before.

The goal: Make this so good that using anything else feels wrong.
"""

import asyncio
import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T")
U = TypeVar("U")
logger = logging.getLogger(__name__)

# Thread-local context for execution
_execution_context: ContextVar[Optional['ExecutionContext']] = ContextVar(
    'execution_context', 
    default=None
)


@dataclass
class ExecutionContext:
    """Context for execution decisions - keep it simple."""
    component: str  # e.g., "vmap", "mesh", "engine"
    pattern: Optional[str] = None  # e.g., "map", "ensemble", "wave"
    is_io_heavy: bool = False  # Hint: lots of API calls or I/O
    
    @classmethod
    def current(cls) -> Optional['ExecutionContext']:
        """Get the current execution context."""
        return _execution_context.get()
    
    def __enter__(self):
        """Set this as the current context."""
        self._token = _execution_context.set(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Reset the context."""
        _execution_context.reset(self._token)


class SimpleMemory:
    """Dead simple learning: remember what executor worked well for each function.
    
    This is the entire learning system. For each function, we remember:
    1. Which executor (thread vs async) was faster
    2. How confident we are in that choice
    
    That's it. No ML, no complex algorithms.
    """
    
    def __init__(self):
        # function_id -> (best_executor, confidence)
        self._memory: Dict[int, Tuple[str, float]] = {}
        
    def remember(self, fn: Callable, executor_type: str, was_fast: bool):
        """Remember if an executor choice was good for a function."""
        fn_id = id(fn)
        
        if fn_id not in self._memory:
            # First time seeing this function
            self._memory[fn_id] = (executor_type, 0.6 if was_fast else 0.4)
        else:
            # Update our confidence
            current_choice, confidence = self._memory[fn_id]
            
            if executor_type == current_choice:
                # Same choice - adjust confidence based on result
                if was_fast:
                    confidence = min(0.95, confidence + 0.1)
                else:
                    confidence = max(0.05, confidence - 0.2)
            else:
                # Different choice - maybe switch if this one is better
                if was_fast and confidence < 0.7:
                    self._memory[fn_id] = (executor_type, 0.6)
                elif not was_fast:
                    # Reinforce current choice
                    confidence = min(0.95, confidence + 0.05)
            
            self._memory[fn_id] = (current_choice, confidence)
    
    def suggest(self, fn: Callable) -> Optional[str]:
        """Suggest which executor to use, or None if we don't know yet."""
        fn_id = id(fn)
        if fn_id in self._memory:
            executor_type, confidence = self._memory[fn_id]
            if confidence > 0.6:  # Only suggest if we're reasonably confident
                return executor_type
        return None


# Global memory instance - shared across all dispatchers
_memory = SimpleMemory()


@runtime_checkable
class Executor(Protocol[T, U]):
    """Protocol for executors that run batched tasks."""

    def execute(self, fn: Callable[[T], U], inputs: List[T]) -> List[U]:
        """Execute function across inputs."""
        ...


class ThreadExecutor(Executor[Dict[str, Any], Any]):
    """Thread-based parallel executor for CPU-bound and mixed workloads."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers
        self._executor = None

    def execute(self, fn: Callable, inputs: List[Dict[str, Any]]) -> List[Any]:
        """Execute function across inputs in parallel."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        futures = [
            self._executor.submit(lambda inp=inp: fn(inputs=inp))
            for inp in inputs
        ]
        
        return [future.result() for future in futures]

    def close(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


class AsyncExecutor(Executor[Dict[str, Any], Any]):
    """Async executor for I/O-heavy workloads."""

    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers

    def execute(self, fn: Callable, inputs: List[Dict[str, Any]]) -> List[Any]:
        """Execute function across inputs using asyncio."""
        async def run_all():
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def run_one(inp):
                async with semaphore:
                    if inspect.iscoroutinefunction(fn):
                        return await fn(inputs=inp)
                    else:
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(None, lambda: fn(inputs=inp))
            
            tasks = [run_one(inp) for inp in inputs]
            return await asyncio.gather(*tasks)
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(run_all())

    def close(self):
        """Clean up resources."""
        pass


class UnifiedDispatcher:
    """The unified parallel execution dispatcher for XCS.
    
    Simple API: Just call map() and it figures out the best way to execute.
    
    The learning is dead simple:
    1. Try an executor (thread or async)
    2. If it's fast, remember that choice
    3. Use what worked well before
    
    That's it. No complex ML, just a simple cache of what worked.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        executor: Optional[str] = None,  # "thread", "async", or None for auto
        context: Optional[ExecutionContext] = None,
    ):
        self.max_workers = max_workers
        self.executor_override = executor
        self.context = context or ExecutionContext.current()
        
        # Lazy-initialized executors
        self._thread_executor = None
        self._async_executor = None
    
    def _choose_executor(self, fn: Callable) -> Tuple[Executor, str]:
        """Choose which executor to use. Simple rules + memory."""
        # 1. Manual override
        if self.executor_override:
            if self.executor_override == "async":
                if not self._async_executor:
                    self._async_executor = AsyncExecutor(self.max_workers or 20)
                return self._async_executor, "async"
            else:
                if not self._thread_executor:
                    self._thread_executor = ThreadExecutor(self.max_workers)
                return self._thread_executor, "thread"
        
        # 2. Check memory - do we know what works for this function?
        suggestion = _memory.suggest(fn)
        if suggestion:
            if suggestion == "async":
                if not self._async_executor:
                    self._async_executor = AsyncExecutor(self.max_workers or 20)
                return self._async_executor, "async"
            else:
                if not self._thread_executor:
                    self._thread_executor = ThreadExecutor(self.max_workers)
                return self._thread_executor, "thread"
        
        # 3. Simple heuristics
        # If it's async or has IO hints, use async
        if inspect.iscoroutinefunction(fn) or (self.context and self.context.is_io_heavy):
            if not self._async_executor:
                self._async_executor = AsyncExecutor(self.max_workers or 20)
            return self._async_executor, "async"
        
        # 4. Default to threads
        if not self._thread_executor:
            self._thread_executor = ThreadExecutor(self.max_workers)
        return self._thread_executor, "thread"
    
    def map(self, fn: Callable, inputs: List[Dict[str, Any]]) -> List[Any]:
        """Execute function across inputs. Simple and fast."""
        if not inputs:
            return []
        
        # Choose executor
        executor, executor_type = self._choose_executor(fn)
        
        # Execute and time it
        start_time = time.time()
        results = executor.execute(fn, inputs)
        duration = time.time() - start_time
        
        # Simple learning: was this fast?
        # "Fast" = less than 10ms per item on average
        avg_time_per_item = duration / len(inputs)
        was_fast = avg_time_per_item < 0.01
        
        # Remember what worked
        _memory.remember(fn, executor_type, was_fast)
        
        # Log if slow (helps debugging)
        if not was_fast and self.context:
            logger.debug(
                f"{self.context.component}: {executor_type} took "
                f"{avg_time_per_item*1000:.1f}ms per item"
            )
        
        return results
    
    def map_with_ids(
        self, 
        fn: Callable[[Dict], Any], 
        items: List[Tuple[str, Dict]]
    ) -> Dict[str, Any]:
        """Map function while preserving IDs. Useful for graph execution."""
        if not items:
            return {}
        
        # Simple wrapper to preserve IDs
        def wrapped(inputs):
            item_id = inputs['_id']
            data = inputs['_data']
            result = fn(data)
            return (item_id, result)
        
        # Execute
        wrapped_inputs = [{'_id': id_, '_data': data} for id_, data in items]
        results = self.map(wrapped, wrapped_inputs)
        
        # Convert to dict
        return dict(results)
    
    def close(self):
        """Clean up resources."""
        if self._thread_executor:
            self._thread_executor.close()
        if self._async_executor:
            self._async_executor.close()


# Simple global functions for one-off use

def parallel_map(fn: Callable, inputs: List[Dict[str, Any]], max_workers: Optional[int] = None) -> List[Any]:
    """Simple parallel map. No fuss."""
    dispatcher = UnifiedDispatcher(max_workers=max_workers)
    try:
        return dispatcher.map(fn, inputs)
    finally:
        dispatcher.close()


def parallel_map_with_context(
    fn: Callable, 
    inputs: List[Dict[str, Any]], 
    component: str,
    **kwargs
) -> List[Any]:
    """Parallel map with context for better optimization."""
    context = ExecutionContext(component=component, **kwargs)
    with context:
        dispatcher = UnifiedDispatcher()
        try:
            return dispatcher.map(fn, inputs)
        finally:
            dispatcher.close()