"""
Proposed minimal implementation of Ember's core functionality.

Following Jeff Dean & Sanjay Ghemawat's systems design principles:
- Measure first, optimize later
- Simple interfaces win
- Make the common case fast

Total implementation: ~500 lines (vs current ~10,000)
"""

import asyncio
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
S = TypeVar('S')

# ==============================================================================
# CORE: Just functions and composition
# ==============================================================================

def compose(*functions: Callable) -> Callable:
    """Function composition: f(g(h(x)))"""
    def composed(x):
        result = x
        for f in reversed(functions):
            result = f(result)
        return result
    return composed


def parallel(*functions: Callable) -> Callable:
    """Execute functions in parallel."""
    async def run_parallel(x):
        tasks = [f(x) if asyncio.iscoroutinefunction(f) else asyncio.to_thread(f, x) 
                 for f in functions]
        return await asyncio.gather(*tasks)
    return run_parallel


# ==============================================================================
# BATCHING: The only real optimization that matters for LLMs
# ==============================================================================

class Batcher:
    """Intelligent batching for API calls."""
    
    def __init__(self, 
                 batch_size: int = 10,
                 timeout_ms: int = 50,
                 max_concurrent: int = 100):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.max_concurrent = max_concurrent
        self.pending: deque = deque()
        self.processing = False
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def submit(self, coro) -> Any:
        """Submit a coroutine for batched execution."""
        future = asyncio.Future()
        
        async with self._lock:
            self.pending.append((coro, future))
            
            # Start processing if not already running
            if not self.processing:
                self.processing = True
                asyncio.create_task(self._process_batches())
        
        return await future
    
    async def _process_batches(self):
        """Process pending calls in batches."""
        while True:
            batch = []
            
            # Collect a batch
            async with self._lock:
                while len(batch) < self.batch_size and self.pending:
                    batch.append(self.pending.popleft())
                
                if not batch and not self.pending:
                    self.processing = False
                    return
            
            if batch:
                # Execute batch with concurrency limit
                await self._execute_batch(batch)
            
            # Small delay to allow more calls to accumulate
            await asyncio.sleep(self.timeout_ms / 1000)
    
    async def _execute_batch(self, batch):
        """Execute a batch of coroutines."""
        async def run_with_semaphore(coro, future):
            async with self._semaphore:
                try:
                    result = await coro
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
        
        tasks = [run_with_semaphore(coro, future) for coro, future in batch]
        await asyncio.gather(*tasks, return_exceptions=True)


# Global batcher for convenience
_default_batcher = Batcher()


def batch(func: Callable) -> Callable:
    """Decorator for intelligent batching."""
    @wraps(func)
    async def batched(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            coro = func(*args, **kwargs)
        else:
            coro = asyncio.to_thread(func, *args, **kwargs)
        return await _default_batcher.submit(coro)
    return batched


# ==============================================================================
# CACHING: Save money and time
# ==============================================================================

class Cache:
    """Simple TTL cache for expensive operations."""
    
    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 10000):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.access_times: deque = deque()
    
    def _make_key(self, args, kwargs) -> str:
        """Create cache key from arguments."""
        # Simple but effective for most cases
        return str((args, tuple(sorted(kwargs.items()))))
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Store value in cache."""
        # Evict oldest entries if cache is full
        while len(self.cache) >= self.max_entries:
            oldest_key = self.access_times.popleft()
            self.cache.pop(oldest_key, None)
        
        self.cache[key] = (value, time.time())
        self.access_times.append(key)


def cache(ttl: int = 3600) -> Callable:
    """Cache decorator for expensive operations."""
    cache_instance = Cache(ttl_seconds=ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def cached(*args, **kwargs):
            key = cache_instance._make_key(args, kwargs)
            
            # Check cache
            result = cache_instance.get(key)
            if result is not None:
                return result
            
            # Execute and cache
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            
            cache_instance.set(key, result)
            return result
        
        return cached
    return decorator


# ==============================================================================
# RETRY: Handle transient failures
# ==============================================================================

def retry(max_attempts: int = 3, 
          delay_seconds: float = 1.0,
          backoff_factor: float = 2.0) -> Callable:
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def with_retry(*args, **kwargs):
            last_exception = None
            delay = delay_seconds
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return await asyncio.to_thread(func, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            raise last_exception
        
        return with_retry
    return decorator


# ==============================================================================
# SIMPLE LLM INTERFACE
# ==============================================================================

class ModelCall:
    """Simple wrapper for LLM calls."""
    
    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider
        self.model = model
        self.params = kwargs
    
    @retry(max_attempts=3)
    @cache(ttl=3600)
    @batch
    async def __call__(self, prompt: str) -> str:
        """Execute the model call."""
        # This is where actual API calls would happen
        # For now, simulate with a delay
        await asyncio.sleep(0.1)  # Simulate network latency
        return f"Response from {self.model}: {prompt[:50]}..."


def model(name: str, **kwargs) -> ModelCall:
    """Create a model instance."""
    provider, model_name = name.split("/") if "/" in name else ("openai", name)
    return ModelCall(provider, model_name, **kwargs)


# ==============================================================================
# COST TRACKING
# ==============================================================================

class CostTracker:
    """Track API costs."""
    
    def __init__(self):
        self.costs = defaultdict(float)
        self.counts = defaultdict(int)
    
    def track(self, model: str, tokens: int, cost_per_1k: float):
        """Track usage and cost."""
        cost = (tokens / 1000) * cost_per_1k
        self.costs[model] += cost
        self.counts[model] += 1
        return cost
    
    def report(self) -> Dict[str, Any]:
        """Get cost report."""
        return {
            "total_cost": sum(self.costs.values()),
            "by_model": dict(self.costs),
            "call_counts": dict(self.counts)
        }


# Global cost tracker
costs = CostTracker()


# ==============================================================================
# SIMPLE PATTERNS
# ==============================================================================

async def ensemble(*models: ModelCall, prompt: str) -> List[str]:
    """Run multiple models in parallel."""
    tasks = [model(prompt) for model in models]
    return await asyncio.gather(*tasks)


async def best_of(n: int, model: ModelCall, prompt: str, 
                  judge: Optional[ModelCall] = None) -> str:
    """Generate N outputs and pick the best."""
    # Generate candidates
    candidates = await asyncio.gather(*[model(prompt) for _ in range(n)])
    
    if judge:
        # Use judge to select best
        judge_prompt = f"Select the best response:\n" + \
                      "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        choice = await judge(judge_prompt)
        # Parse choice (simplified)
        return candidates[0]  # Would parse judge's response
    else:
        # Simple heuristic: longest response
        return max(candidates, key=len)


def majority_vote(responses: List[str]) -> str:
    """Simple majority voting."""
    from collections import Counter
    votes = Counter(responses)
    return votes.most_common(1)[0][0]


# ==============================================================================
# VMAP: Simple list processing
# ==============================================================================

def vmap(func: Callable[[T], S]) -> Callable[[List[T]], List[S]]:
    """Map function over lists with automatic parallelization."""
    @wraps(func)
    async def vmapped(items: List[T]) -> List[S]:
        if asyncio.iscoroutinefunction(func):
            return await asyncio.gather(*[func(item) for item in items])
        else:
            # Run in thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(None, func, item) for item in items]
            return await asyncio.gather(*tasks)
    return vmapped


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def example_usage():
    """Show how simple the API can be."""
    
    # Create models
    gpt4 = model("gpt-4", temperature=0.7)
    claude = model("anthropic/claude-3", temperature=0.5)
    
    # Simple call
    response = await gpt4("What is the capital of France?")
    print(f"Simple: {response}")
    
    # Ensemble
    responses = await ensemble(gpt4, claude, prompt="Explain quantum computing")
    print(f"Ensemble: {responses}")
    
    # Best of N
    best = await best_of(3, gpt4, "Write a haiku about programming")
    print(f"Best of 3: {best}")
    
    # Batch processing
    texts = ["Hello", "World", "How", "Are", "You"]
    
    @vmap
    async def analyze(text: str) -> str:
        return await gpt4(f"Sentiment of '{text}':")
    
    results = await analyze(texts)
    print(f"Batch results: {results}")
    
    # Cost report
    print(f"Costs: {costs.report()}")


if __name__ == "__main__":
    # This entire implementation is under 500 lines
    # Compare to the current ~10,000 lines
    asyncio.run(example_usage())