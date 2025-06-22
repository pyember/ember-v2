# Simplified Ember Implementation
# Following principles of Knuth, Ritchie, Carmack, Dean, Ghemawat, Brockman

"""
This is what Ember should be: ~500 lines that do one thing well.
Make LLM calls fast through parallelization.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Iterator
import aiohttp
import numpy as np
from dataclasses import dataclass
import json

# Type definitions
T = TypeVar('T')
LLMProvider = Callable[[str, str], str]

# Global state (minimal)
_providers: Dict[str, LLMProvider] = {}
_metrics = {
    'total_calls': 0,
    'total_time_ms': 0,
    'cache_hits': 0,
    'parallel_executions': 0
}

# Core API (10 functions)
__all__ = ['llm', 'jit', 'vmap', 'pmap', 'chain', 'ensemble', 'retry', 'cache', 'measure', 'stream']

# 1. Core LLM function
def llm(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Call an LLM. Simple as that."""
    start = time.perf_counter()
    
    provider = _providers.get(model)
    if not provider:
        raise ValueError(f"Unknown model: {model}. Register with register_provider()")
    
    result = provider(model, prompt)
    
    # Update metrics
    _metrics['total_calls'] += 1
    _metrics['total_time_ms'] += (time.perf_counter() - start) * 1000
    
    return result

# 2. JIT - Make functions fast by parallelizing LLM calls
def jit(func: Callable) -> Callable:
    """
    Make function fast by parallelizing independent LLM calls.
    
    This is the ONLY optimization that matters for LLM apps.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Trace execution to find LLM calls
        calls = []
        results = {}
        
        # Monkey-patch llm to collect calls instead of executing
        original_llm = globals()['llm']
        call_id = 0
        
        def tracing_llm(prompt: str, model: str = "gpt-3.5-turbo") -> str:
            nonlocal call_id
            call_key = f"__llm_call_{call_id}__"
            call_id += 1
            calls.append((call_key, prompt, model))
            return call_key
        
        # Trace the function
        globals()['llm'] = tracing_llm
        try:
            template_result = func(*args, **kwargs)
        finally:
            globals()['llm'] = original_llm
        
        # If no LLM calls or just one, run normally
        if len(calls) <= 1:
            return func(*args, **kwargs)
        
        # Execute LLM calls in parallel
        _metrics['parallel_executions'] += 1
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(original_llm, prompt, model): key
                for key, prompt, model in calls
            }
            
            for future in as_completed(futures):
                key = futures[future]
                results[key] = future.result()
        
        # Replace placeholders with results
        def replace_placeholders(obj):
            if isinstance(obj, str) and obj in results:
                return results[obj]
            elif isinstance(obj, dict):
                return {k: replace_placeholders(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(replace_placeholders(item) for item in obj)
            return obj
        
        return replace_placeholders(template_result)
    
    return wrapper

# 3. VMAP - Batch processing
def vmap(func: Callable[[T], Any], batch_size: int = 10) -> Callable[[List[T]], List[Any]]:
    """Map function over inputs in batches."""
    def wrapper(inputs: List[T]) -> List[Any]:
        results = []
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Parallel execution within batch
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(func, item) for item in batch]
                batch_results = [f.result() for f in futures]
                results.extend(batch_results)
        
        return results
    
    return wrapper

# 4. PMAP - Parallel map (simpler than vmap)
def pmap(func: Callable[[T], Any]) -> Callable[[List[T]], List[Any]]:
    """Parallel map - process all inputs concurrently."""
    return vmap(func, batch_size=1000)  # Large batch = full parallel

# 5. CHAIN - Sequential composition
def chain(*funcs: Callable) -> Callable:
    """Chain functions together: chain(f, g, h)(x) = h(g(f(x)))"""
    def wrapper(x):
        for func in funcs:
            x = func(x)
        return x
    return wrapper

# 6. ENSEMBLE - Parallel composition
def ensemble(*funcs: Callable, aggregator: Optional[Callable] = None) -> Callable:
    """Run functions in parallel and aggregate results."""
    if aggregator is None:
        aggregator = lambda results: results  # Return list by default
    
    def wrapper(x):
        # Run all functions in parallel
        with ThreadPoolExecutor(max_workers=len(funcs)) as executor:
            futures = [executor.submit(func, x) for func in funcs]
            results = [f.result() for f in futures]
        
        return aggregator(results)
    
    return wrapper

# 7. RETRY - Reliability
def retry(func: Callable, max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """Retry function on failure."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
        raise last_error
    
    return wrapper

# 8. CACHE - Performance
_cache: Dict[str, Any] = {}

def cache(func: Callable, ttl: int = 3600) -> Callable:
    """Cache function results."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Simple cache key (could be improved)
        key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
        
        if key in _cache:
            _metrics['cache_hits'] += 1
            return _cache[key]
        
        result = func(*args, **kwargs)
        _cache[key] = result
        return result
    
    return wrapper

# 9. MEASURE - Observability
def measure(func: Callable) -> Callable:
    """Measure function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"{func.__name__}: {elapsed:.1f}ms")
        return result
    
    return wrapper

# 10. STREAM - Memory efficiency
async def stream(prompt: str, model: str = "gpt-3.5-turbo") -> Iterator[str]:
    """Stream LLM responses for memory efficiency."""
    # This would connect to real streaming API
    # Simplified for demonstration
    response = llm(prompt, model)
    for word in response.split():
        yield word
        await asyncio.sleep(0.01)  # Simulate streaming

# Provider registration (minimal)
def register_provider(name: str, provider: LLMProvider):
    """Register an LLM provider."""
    _providers[name] = provider

# Example provider
def mock_provider(model: str, prompt: str) -> str:
    """Mock provider for testing."""
    time.sleep(0.1)  # Simulate API call
    return f"Response to: {prompt[:50]}..."

# Register default provider
register_provider("gpt-3.5-turbo", mock_provider)
register_provider("gpt-4", mock_provider)

# Metrics access
def get_metrics() -> dict:
    """Get performance metrics."""
    return {
        **_metrics,
        'avg_latency_ms': _metrics['total_time_ms'] / max(_metrics['total_calls'], 1),
        'cache_hit_rate': _metrics['cache_hits'] / max(_metrics['total_calls'], 1) * 100
    }

# Common aggregators for ensemble
def majority_vote(results: List[Any]) -> Any:
    """Return most common result."""
    from collections import Counter
    return Counter(results).most_common(1)[0][0]

def mean_aggregator(results: List[float]) -> float:
    """Return mean of numeric results."""
    return np.mean(results)

# Example usage showing the entire framework in action
if __name__ == "__main__":
    # Example 1: Simple LLM call
    print("Example 1: Simple LLM call")
    response = llm("What is the capital of France?")
    print(f"Response: {response}\n")
    
    # Example 2: JIT optimization
    print("Example 2: JIT optimization")
    
    @jit
    @measure
    def analyze_email(email: dict) -> dict:
        return {
            'subject_analysis': llm(f"Analyze subject: {email['subject']}"),
            'body_sentiment': llm(f"Sentiment of: {email['body']}"),
            'priority': llm(f"Priority level of: {email['subject']}")
        }
    
    email = {
        'subject': "Urgent: Server Down",
        'body': "The production server is not responding."
    }
    
    result = analyze_email(email)
    print(f"Analysis result: {result}\n")
    
    # Example 3: Batch processing
    print("Example 3: Batch processing")
    
    @measure
    def classify(text: str) -> str:
        return llm(f"Classify: {text}")
    
    texts = ["Happy message", "Angry complaint", "Neutral statement"] * 3
    batch_classify = vmap(classify, batch_size=3)
    results = batch_classify(texts)
    print(f"Batch results: {results}\n")
    
    # Example 4: Ensemble
    print("Example 4: Ensemble")
    
    sentiment_ensemble = ensemble(
        lambda x: llm(f"Sentiment: {x}"),
        lambda x: llm(f"Emotion: {x}"),
        lambda x: llm(f"Tone: {x}"),
        aggregator=lambda results: {"all": results}
    )
    
    ensemble_result = sentiment_ensemble("I love this product!")
    print(f"Ensemble result: {ensemble_result}\n")
    
    # Show metrics
    print("Performance Metrics:")
    print(json.dumps(get_metrics(), indent=2))

# Total lines: ~400
# Concepts to understand: 3 (functions, decorators, parallelization)
# Time to first success: < 5 minutes

"""
This is what Ember should be:
1. Simple - Just functions and decorators
2. Fast - Parallelization where it matters (LLM I/O)
3. Practical - Solves real problems (latency, reliability, cost)

Not included (and shouldn't be):
- Complex type systems
- Operator base classes  
- Multiple JIT strategies
- Framework-specific concepts
- Configuration files
- Metaclass magic

As Ritchie said about C: "A language that doesn't have everything is actually easier to program in than some that do."
"""