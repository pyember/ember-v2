# XCS Power User Design: Clean Abstractions

## Philosophy

Following Martin's principle: "Abstractions should not depend on details. Details should depend on abstractions."

Power users need control without seeing implementation details.

## The Advanced API

```python
# ember/xcs/advanced/__init__.py
"""Advanced XCS features for power users.

Warning: Most users should just use @jit from ember.xcs.
Only use these features if you have specific requirements.
"""

from ember.xcs.advanced.transforms import Transform, transform
from ember.xcs.advanced.constraints import Constraints
from ember.xcs.advanced.plugins import Plugin, register_plugin

__all__ = ['Transform', 'transform', 'Constraints', 'Plugin', 'register_plugin']
```

## Clean Abstractions for Power Users

### 1. Transforms (Not Strategies!)

```python
# ember/xcs/advanced/transforms.py
"""Transform how operations execute, not how they're scheduled."""

from abc import ABC, abstractmethod
from typing import Any, Callable


class Transform(ABC):
    """Base class for execution transforms.
    
    Transforms modify HOW operations run, not WHEN they run.
    Scheduling remains automatic.
    """
    
    @abstractmethod
    def apply(self, func: Callable, *args, **kwargs) -> Any:
        """Apply transformation to function execution."""
        pass


class BatchTransform(Transform):
    """Automatically batch operations for efficiency."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    def apply(self, func: Callable, *args, **kwargs) -> Any:
        # Implementation batches inputs automatically
        pass


class CacheTransform(Transform):
    """Advanced caching with TTL and size limits."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
    
    def apply(self, func: Callable, *args, **kwargs) -> Any:
        # Implementation handles cache management
        pass


def transform(*transforms: Transform):
    """Apply transforms to a jitted function.
    
    Example:
        @jit
        @transform(
            BatchTransform(batch_size=64),
            CacheTransform(ttl_seconds=7200)
        )
        def process(items):
            return [model(item) for item in items]
    """
    def decorator(func):
        func._xcs_transforms = transforms
        return func
    return decorator
```

### 2. Constraints (Not Execution Options!)

```python
# ember/xcs/advanced/constraints.py
"""Constraints on execution, not configuration of execution."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Constraints:
    """Constraints for execution behavior.
    
    These are requirements, not configurations.
    The system decides how to meet them.
    """
    
    # Resource constraints
    max_memory_gb: Optional[float] = None
    max_time_seconds: Optional[float] = None
    
    # Behavioral constraints  
    deterministic: bool = False  # Require deterministic execution
    fault_tolerant: bool = False  # Require fault tolerance
    
    # Data constraints
    private_data: bool = False  # Don't cache, don't log
    
    # No scheduler configuration!
    # No strategy selection!
    # Just constraints the system must satisfy


# Usage:
@jit
@constrain(Constraints(
    max_memory_gb=8.0,
    deterministic=True,
    private_data=True
))
def process_sensitive_data(data):
    return model(data)
```

### 3. Plugins (For the 1%)

```python
# ember/xcs/advanced/plugins.py
"""Plugin system for extending XCS behavior."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Plugin(ABC):
    """Base class for XCS plugins.
    
    Plugins can observe and modify execution without
    exposing internal implementation details.
    """
    
    @abstractmethod
    def on_execute_start(self, func_name: str, args: tuple, kwargs: dict):
        """Called before execution starts."""
        pass
    
    @abstractmethod
    def on_execute_end(self, func_name: str, result: Any, duration_ms: float):
        """Called after execution completes."""
        pass
    
    def on_parallel_split(self, func_name: str, num_chunks: int):
        """Called when execution is parallelized."""
        pass
    
    def on_cache_hit(self, func_name: str, cache_key: str):
        """Called when result is served from cache."""
        pass


class MetricsPlugin(Plugin):
    """Example: Custom metrics collection."""
    
    def __init__(self, metrics_backend):
        self.backend = metrics_backend
    
    def on_execute_end(self, func_name: str, result: Any, duration_ms: float):
        self.backend.record_timing(func_name, duration_ms)
    
    def on_parallel_split(self, func_name: str, num_chunks: int):
        self.backend.record_parallelism(func_name, num_chunks)


def register_plugin(plugin: Plugin):
    """Register a global plugin for all XCS execution.
    
    Example:
        metrics = MetricsPlugin(prometheus_backend)
        register_plugin(metrics)
    """
    # Implementation adds to global plugin registry
    pass
```

## What Power Users Can Do

### 1. Custom Caching Strategy
```python
@jit
@transform(CacheTransform(ttl_seconds=3600, max_size=1000))
def expensive_computation(x):
    return complex_model(x)
```

### 2. Resource Limits
```python
@jit
@constrain(Constraints(max_memory_gb=4.0, max_time_seconds=30.0))
def memory_intensive_task(data):
    return large_model(data)
```

### 3. Custom Metrics
```python
class MyMetrics(Plugin):
    def on_execute_end(self, func_name, result, duration_ms):
        print(f"{func_name}: {duration_ms}ms")

register_plugin(MyMetrics())
```

### 4. Combining Features
```python
@jit
@transform(
    BatchTransform(batch_size=128),
    CacheTransform(ttl_seconds=7200)
)
@constrain(Constraints(
    deterministic=True,
    fault_tolerant=True
))
def production_pipeline(items):
    return [process(item) for item in items]
```

## What Power Users Still Can't Do

1. **Choose Schedulers**: System picks optimal scheduler
2. **Configure Strategies**: System picks optimal strategy  
3. **Access IR**: IR remains internal implementation
4. **Control Parallelism Details**: System decides how to parallelize
5. **See Implementation**: No access to engine internals

## Key Design Principles

1. **Constraints, Not Configuration**: Tell the system what you need, not how to do it
2. **Transforms, Not Strategies**: Transform execution, don't control scheduling
3. **Plugins, Not Hooks**: Observe and extend, don't modify internals
4. **Composition**: Features compose cleanly without conflicts
5. **No Leaks**: Implementation details remain hidden

## Error Messages for Power Users

```python
# Bad (exposes internals):
"BatchingStrategy incompatible with StreamScheduler"

# Good (user-focused):
"Cannot batch this operation because it processes streaming data"
```

## Documentation Strategy

```python
# Main docs show only @jit
# Advanced section hidden behind "Advanced Usage" 
# Each advanced feature has warnings about complexity
# Examples show when NOT to use advanced features
```

This design gives power users control without exposing implementation details, maintaining the clean abstraction boundaries that Martin advocates.