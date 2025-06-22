# Ember Simplified Core
# ~500 lines that do one thing well: make LLM calls fast

"""
The entire Ember framework in one file.
Following principles of Knuth, Ritchie, Carmack.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Iterator
import threading
from dataclasses import dataclass
import weakref
import hashlib
import json

# Type definitions
T = TypeVar('T')
R = TypeVar('R')

# Global state (minimal)
_providers: Dict[str, 'Provider'] = {}
_default_model = "gpt-3.5-turbo"
_metrics_collector = None

# Thread-local storage for JIT context
_jit_context = threading.local()

# Performance metrics
@dataclass
class Metrics:
    """Performance metrics for observability."""
    total_calls: int = 0
    total_time_ms: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_executions: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_time_ms / max(self.total_calls, 1)
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(total, 1)) * 100

# Provider interface
class Provider:
    """LLM provider interface."""
    
    def complete(self, prompt: str, model: str) -> str:
        """Complete a prompt."""
        raise NotImplementedError
    
    async def stream(self, prompt: str, model: str) -> Iterator[str]:
        """Stream a response."""
        # Default: convert complete to stream
        response = self.complete(prompt, model)
        for word in response.split():
            yield word

# Core API Implementation

def llm(prompt: str, model: Optional[str] = None) -> str:
    """
    Call an LLM. Simple as that.
    
    Args:
        prompt: The prompt to send
        model: Model to use (default: gpt-3.5-turbo)
        
    Returns:
        The model's response
    """
    if model is None:
        model = _default_model
    
    # Check if we're in JIT tracing mode
    if hasattr(_jit_context, 'tracing') and _jit_context.tracing:
        # Record the call instead of executing
        call_id = len(_jit_context.calls)
        placeholder = f"__llm_{call_id}__"
        _jit_context.calls.append({
            'id': placeholder,
            'prompt': prompt,
            'model': model
        })
        return placeholder
        
    provider = _providers.get(model)
    if not provider:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Available: {list(_providers.keys())}"
        )
    
    start = time.perf_counter()
    try:
        result = provider.complete(prompt, model)
        return result
    finally:
        if _metrics_collector:
            elapsed = (time.perf_counter() - start) * 1000
            _metrics_collector.total_calls += 1
            _metrics_collector.total_time_ms += elapsed

def jit(func: Callable[..., R]) -> Callable[..., R]:
    """
    Make function fast by parallelizing independent LLM calls.
    
    This is THE optimization that matters for LLM apps:
    turning sequential I/O into parallel I/O.
    
    Example:
        @jit
        def analyze(text):
            sentiment = llm(f"Sentiment: {text}")
            summary = llm(f"Summarize: {text}")
            return {"sentiment": sentiment, "summary": summary}
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Phase 1: Trace to find LLM calls
        _jit_context.tracing = True
        _jit_context.calls = []
        
        try:
            # Execute function in tracing mode
            template_result = func(*args, **kwargs)
        finally:
            _jit_context.tracing = False
        
        # Get traced calls
        calls = getattr(_jit_context, 'calls', [])
        
        # If no calls or just one, run normally
        if len(calls) <= 1:
            return func(*args, **kwargs)
        
        # Phase 2: Execute calls in parallel
        if _metrics_collector:
            _metrics_collector.parallel_executions += 1
        
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(calls), 10)) as executor:
            # Submit all LLM calls
            future_to_call = {}
            for call in calls:
                future = executor.submit(
                    llm, 
                    call['prompt'], 
                    call['model']
                )
                future_to_call[future] = call['id']
            
            # Collect results as they complete
            for future in as_completed(future_to_call):
                call_id = future_to_call[future]
                results[call_id] = future.result()
        
        # Phase 3: Replace placeholders with results
        def replace_placeholders(obj):
            """Recursively replace placeholders in the result."""
            if isinstance(obj, str) and obj in results:
                return results[obj]
            elif isinstance(obj, dict):
                return {k: replace_placeholders(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(replace_placeholders(item) for item in obj)
            else:
                return obj
        
        return replace_placeholders(template_result)
    
    return wrapper

def vmap(func: Callable[[T], R], batch_size: int = 10) -> Callable[[List[T]], List[R]]:
    """
    Vectorized map - process inputs in batches.
    
    Example:
        classify = vmap(lambda x: llm(f"Classify: {x}"))
        results = classify(["text1", "text2", ...])
    """
    @wraps(func)
    def wrapper(inputs: List[T]) -> List[R]:
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Parallel execution within batch
            with ThreadPoolExecutor(max_workers=min(len(batch), batch_size)) as executor:
                futures = [executor.submit(func, item) for item in batch]
                batch_results = [f.result() for f in futures]
                results.extend(batch_results)
        
        return results
    
    return wrapper

def pmap(func: Callable[[T], R], max_workers: Optional[int] = None) -> Callable[[List[T]], List[R]]:
    """
    Parallel map - process all inputs concurrently.
    
    Like vmap but processes everything at once.
    Use when you have fewer items or need maximum speed.
    """
    if max_workers is None:
        max_workers = 32  # Reasonable default
        
    return vmap(func, batch_size=max_workers)

def chain(*funcs: Callable) -> Callable:
    """
    Chain functions sequentially.
    
    Example:
        pipeline = chain(
            extract_text,
            clean_data,
            analyze_sentiment,
            format_output
        )
        result = pipeline(document)
    """
    def chained(x):
        for func in funcs:
            x = func(x)
        return x
    
    # Preserve function metadata
    if funcs:
        chained.__name__ = f"chain({', '.join(f.__name__ for f in funcs)})"
        
    return chained

def ensemble(*funcs: Callable, aggregator: Optional[Callable] = None) -> Callable:
    """
    Run functions in parallel and aggregate results.
    
    Example:
        sentiment = ensemble(
            lambda x: llm(f"Sentiment: {x}"),
            lambda x: llm(f"Emotion: {x}"),
            aggregator=majority_vote
        )
    """
    if aggregator is None:
        # Default: return list of results
        aggregator = lambda results: results
    
    def ensembled(x):
        # Execute all functions in parallel
        with ThreadPoolExecutor(max_workers=len(funcs)) as executor:
            futures = [executor.submit(func, x) for func in funcs]
            results = [f.result() for f in futures]
        
        return aggregator(results)
    
    return ensembled

def retry(
    func: Optional[Callable] = None,
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Retry function on failure with exponential backoff.
    
    Example:
        @retry(max_attempts=3, delay=1.0)
        def flaky_operation():
            return llm("Do something")
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        time.sleep(wait_time)
            
            # Re-raise the last exception
            raise last_exception
        
        return wrapper
    
    # Handle both @retry and @retry()
    if func is None:
        return decorator
    else:
        return decorator(func)

# Simple cache implementation
_cache_storage = {}
_cache_lock = threading.Lock()

def cache(
    func: Optional[Callable] = None,
    *,
    ttl: Optional[int] = 3600,
    key_func: Optional[Callable] = None
) -> Callable:
    """
    Cache function results.
    
    Example:
        @cache(ttl=3600)
        def expensive_analysis(text):
            return llm(f"Analyze: {text}")
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Simple key generation
                key_data = (f.__name__, args, tuple(sorted(kwargs.items())))
                cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Check cache
            with _cache_lock:
                if cache_key in _cache_storage:
                    entry = _cache_storage[cache_key]
                    if ttl is None or time.time() - entry['time'] < ttl:
                        if _metrics_collector:
                            _metrics_collector.cache_hits += 1
                        return entry['value']
            
            # Cache miss
            if _metrics_collector:
                _metrics_collector.cache_misses += 1
                
            # Execute function
            result = f(*args, **kwargs)
            
            # Store in cache
            with _cache_lock:
                _cache_storage[cache_key] = {
                    'value': result,
                    'time': time.time()
                }
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

def measure(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """
    Measure function performance.
    
    Example:
        @measure
        def my_function():
            return llm("Do something")
    """
    def decorator(f):
        nonlocal name
        if name is None:
            name = f.__name__
            
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                print(f"{name}: {elapsed:.1f}ms")
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

async def stream(prompt: str, model: Optional[str] = None) -> Iterator[str]:
    """
    Stream LLM responses for memory efficiency.
    
    Example:
        async for chunk in stream("Tell me a story"):
            print(chunk, end='')
    """
    if model is None:
        model = _default_model
        
    provider = _providers.get(model)
    if not provider:
        raise ValueError(f"Unknown model: {model}")
    
    async for chunk in provider.stream(prompt, model):
        yield chunk

# Provider registration
def register_provider(model: str, provider: Provider):
    """Register an LLM provider."""
    _providers[model] = provider

def set_default_model(model: str):
    """Set the default model."""
    global _default_model
    _default_model = model

# Metrics access
def get_metrics() -> Metrics:
    """Get performance metrics."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = Metrics()
    return _metrics_collector

def reset_metrics():
    """Reset performance metrics."""
    global _metrics_collector
    _metrics_collector = Metrics()

# Common aggregators for ensemble
def majority_vote(results: List[Any]) -> Any:
    """Return the most common result."""
    from collections import Counter
    if not results:
        return None
    counter = Counter(results)
    return counter.most_common(1)[0][0]

def mean_aggregator(results: List[Union[int, float]]) -> float:
    """Return mean of numeric results."""
    if not results:
        return 0.0
    return sum(results) / len(results)

def first_valid(results: List[Any]) -> Any:
    """Return first non-None result."""
    for result in results:
        if result is not None:
            return result
    return None

# Clear cache utility
def clear_cache():
    """Clear the cache."""
    global _cache_storage
    with _cache_lock:
        _cache_storage.clear()

# Context manager for metrics
class metrics_context:
    """Context manager for isolated metrics collection."""
    
    def __init__(self):
        self.metrics = Metrics()
        self.old_collector = None
        
    def __enter__(self):
        global _metrics_collector
        self.old_collector = _metrics_collector
        _metrics_collector = self.metrics
        return self.metrics
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _metrics_collector
        _metrics_collector = self.old_collector

# Export public API
__all__ = [
    # Core functions
    'llm',
    'jit', 
    'vmap',
    'pmap',
    'chain',
    'ensemble',
    'retry',
    'cache',
    'measure',
    'stream',
    
    # Configuration
    'register_provider',
    'set_default_model',
    
    # Metrics
    'get_metrics',
    'reset_metrics',
    'metrics_context',
    
    # Utilities
    'clear_cache',
    'majority_vote',
    'mean_aggregator',
    'first_valid',
    
    # Types
    'Provider',
    'Metrics'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Ember Team'

"""
Total lines: ~500
Concepts to learn: 3 (functions, decorators, parallelization)
No inheritance hierarchies
No configuration files
No framework magic

Just functions that make LLM calls fast.
"""