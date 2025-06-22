"""
Example implementation showing how to backport the best innovations
to the original Ember while maintaining its simplicity.

This is what the original ember/api/models.py could look like with
the natural API pattern and smart improvements.
"""

import asyncio
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Union
import threading


class Response:
    """Simple response wrapper."""
    def __init__(self, text: str, model: str, usage: dict = None):
        self.text = text
        self.model = model
        self.usage = usage or {}


class CostTracker:
    """Track API costs across all calls."""
    
    # Cost per 1k tokens (example rates)
    COSTS = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
        "claude-3": {"prompt": 0.015, "completion": 0.075},
    }
    
    def __init__(self):
        self.costs = defaultdict(float)
        self.counts = defaultdict(int)
        self._lock = threading.Lock()
    
    def track(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Track cost for a single call."""
        with self._lock:
            if model in self.COSTS:
                cost = (
                    prompt_tokens * self.COSTS[model]["prompt"] + 
                    completion_tokens * self.COSTS[model]["completion"]
                ) / 1000
                self.costs[model] += cost
                self.counts[model] += 1
                return cost
        return 0.0
    
    def report(self) -> dict:
        """Get cost report."""
        with self._lock:
            return {
                "total_cost": sum(self.costs.values()),
                "by_model": dict(self.costs),
                "call_counts": dict(self.counts),
                "avg_cost_per_call": {
                    model: self.costs[model] / self.counts[model]
                    for model in self.costs if self.counts[model] > 0
                }
            }


class BatchCollector:
    """Collect requests for batching to respect rate limits."""
    
    def __init__(self, batch_size: int = 10, timeout_ms: int = 50):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.pending = deque()
        self._lock = threading.Lock()
        self._futures = {}
    
    def add(self, model: str, prompt: str, **kwargs) -> asyncio.Future:
        """Add request to batch, return future for result."""
        future = asyncio.Future()
        
        with self._lock:
            request_id = id(future)
            self.pending.append((request_id, model, prompt, kwargs))
            self._futures[request_id] = future
            
            if len(self.pending) >= self.batch_size:
                self._process_batch()
                
        return future
    
    def _process_batch(self):
        """Process accumulated batch."""
        batch = []
        while self.pending and len(batch) < self.batch_size:
            batch.append(self.pending.popleft())
            
        # In real implementation, this would make batched API calls
        # For now, simulate with immediate response
        for request_id, model, prompt, kwargs in batch:
            if request_id in self._futures:
                future = self._futures.pop(request_id)
                # Simulate API response
                response = Response(
                    text=f"Response to: {prompt[:50]}...",
                    model=model,
                    usage={"prompt_tokens": 10, "completion_tokens": 20}
                )
                future.set_result(response)


class Models:
    """Enhanced models API with natural calling pattern."""
    
    def __init__(self):
        self._providers = {}
        self._cost_tracker = CostTracker()
        self._batch_collector = BatchCollector()
        
    def __call__(self, model: str, prompt: str, **kwargs) -> Response:
        """
        Natural calling pattern - the one obvious way.
        
        Examples:
            response = models("gpt-4", "What is Python?")
            print(response.text)
        """
        # Simulate API call (in real implementation, use provider)
        response = Response(
            text=f"Response from {model}: {prompt[:50]}...",
            model=model,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": 50}
        )
        
        # Track costs
        if response.usage:
            self._cost_tracker.track(
                model,
                response.usage.get("prompt_tokens", 0),
                response.usage.get("completion_tokens", 0)
            )
            
        return response
    
    def instance(self, model: str, **config) -> Callable:
        """
        Create reusable model instance with configuration.
        
        Examples:
            gpt4 = models.instance("gpt-4", temperature=0.5)
            response = gpt4("Explain quantum computing")
        """
        def configured_call(prompt: str) -> Response:
            return self(model, prompt, **config)
        
        configured_call.__name__ = f"{model}_instance"
        return configured_call
    
    async def async_call(self, model: str, prompt: str, **kwargs) -> Response:
        """
        Async variant for concurrent operations.
        
        Examples:
            responses = await asyncio.gather(
                models.async_call("gpt-4", "Question 1"),
                models.async_call("gpt-4", "Question 2"),
                models.async_call("gpt-4", "Question 3"),
            )
        """
        # In real implementation, use aiohttp/httpx
        await asyncio.sleep(0.1)  # Simulate API latency
        return self(model, prompt, **kwargs)
    
    def batched(self, model: str, prompt: str, **kwargs) -> asyncio.Future:
        """
        Automatically batch calls to respect rate limits.
        
        The future will be resolved when the batch is processed.
        """
        return self._batch_collector.add(model, prompt, **kwargs)
    
    @property
    def costs(self) -> dict:
        """Get current cost tracking report."""
        return self._cost_tracker.report()


# Function combinators for operators (simple, powerful)

def parallel(*funcs: Callable) -> Callable:
    """
    Execute functions in parallel.
    
    Example:
        classify = parallel(
            lambda x: models("gpt-4", f"Classify: {x}"),
            lambda x: models("claude-3", f"Classify: {x}")
        )
        results = classify("Is this spam?")
    """
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(f, *args, **kwargs) for f in funcs]
            return [f.result() for f in futures]
    return wrapper


def chain(*funcs: Callable) -> Callable:
    """
    Chain functions: f(g(h(x))).
    
    Example:
        pipeline = chain(
            preprocess,
            lambda x: models("gpt-4", f"Analyze: {x}"),
            postprocess
        )
        result = pipeline(raw_data)
    """
    def wrapper(x):
        result = x
        for f in funcs:
            result = f(result)
        return result
    return wrapper


def retry(func: Callable, max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """
    Retry on failure with exponential backoff.
    
    Example:
        safe_call = retry(lambda: models("gpt-4", prompt), max_attempts=3)
        response = safe_call()
    """
    def wrapper(*args, **kwargs):
        last_error = None
        current_delay = delay
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(current_delay)
                    current_delay *= 2
                    
        raise last_error
    
    return wrapper


# Usage example showing the enhanced API

async def example_usage():
    """Show how the enhanced API maintains simplicity while adding power."""
    
    # 1. Simple, direct usage (most common case)
    response = models("gpt-4", "What is the capital of France?")
    print(f"Simple: {response.text}")
    
    # 2. Reusable instances
    gpt4_creative = models.instance("gpt-4", temperature=0.8)
    story = gpt4_creative("Write a short story about robots")
    print(f"Creative: {story.text[:100]}...")
    
    # 3. Parallel execution with combinators
    ensemble_classify = parallel(
        models.instance("gpt-4"),
        models.instance("claude-3"),
        models.instance("gpt-3.5-turbo")
    )
    
    results = ensemble_classify("Is 'Free money click here!' spam?")
    print(f"Ensemble results: {[r.text for r in results]}")
    
    # 4. Async for true concurrency
    questions = [
        "What is machine learning?",
        "Explain quantum computing",
        "Define artificial intelligence"
    ]
    
    responses = await asyncio.gather(*[
        models.async_call("gpt-4", q) for q in questions
    ])
    print(f"Async batch: Got {len(responses)} responses")
    
    # 5. Cost tracking (automatic)
    print(f"Costs so far: {models.costs}")
    
    # 6. Pipeline with retry
    safe_pipeline = retry(
        chain(
            lambda x: x.lower().strip(),
            lambda x: models("gpt-4", f"Translate to Spanish: {x}"),
            lambda r: r.text.upper()
        ),
        max_attempts=3
    )
    
    translation = safe_pipeline("Hello world")
    print(f"Safe translation: {translation}")


# Global instance (like original)
models = Models()

# But also export useful utilities
__all__ = [
    'models',      # The main API
    'Response',    # Response type
    'parallel',    # Combinators
    'chain',
    'retry',
]


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())