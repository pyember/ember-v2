# XCS Minimal API Implementation

## The 90% API: Just @jit

```python
# ember/xcs/__init__.py
"""XCS: Smart execution for Ember operators.

Just use @jit. That's it.
"""

from ember.xcs._simple import jit

__all__ = ['jit']

# That's the entire public API for 90% of users
```

## The Implementation (Hidden)

```python
# ember/xcs/_simple.py
"""Simple JIT implementation with smart defaults."""

import functools
import inspect
from typing import Any, Callable, Optional

from ember.xcs._internal.ir_builder import IRBuilder
from ember.xcs._internal.engine import ExecutionEngine
from ember.xcs._internal.profiler import Profiler


# Global engine instance (hidden from users)
_engine = ExecutionEngine()
_profiler = Profiler()


def jit(func: Optional[Callable] = None, *, 
        _config: Optional['Config'] = None) -> Callable:
    """Make any function faster. No configuration needed.
    
    Examples:
        @jit
        def process(x):
            return model(x)
            
        # That's it. Automatic parallelization, caching, and optimization.
    """
    # Handle both @jit and @jit()
    if func is None:
        return functools.partial(jit, _config=_config)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Build IR from function
        ir_graph = _trace_function(func, args, kwargs)
        
        # Execute with smart defaults
        result = _engine.execute(ir_graph, args, kwargs)
        
        # Profile for future optimization
        if _should_profile():
            _profiler.record(func.__name__, ir_graph, result)
        
        return result
    
    # Attach metadata for power users (but not documented)
    wrapper._xcs_jit = True
    wrapper._xcs_original = func
    
    return wrapper


def _trace_function(func: Callable, args: tuple, kwargs: dict) -> 'IRGraph':
    """Trace function execution to build IR. Hidden from users."""
    builder = IRBuilder()
    
    # Smart tracing that understands EmberModules
    with builder.tracing():
        # This is where the magic happens
        # But users never see this complexity
        pass
    
    return builder.build()


def _should_profile() -> bool:
    """Smart profiling decision. Hidden from users."""
    # Profile 1% of executions for continuous optimization
    import random
    return random.random() < 0.01
```

## The 9% API: Simple Configuration

```python
# ember/xcs/config.py
"""Configuration for advanced users who need control."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Simple configuration options. No complexity exposed.
    
    Examples:
        @jit(config=Config(cache=False))
        def process(x):
            return model(x)
    """
    parallel: bool = True      # Enable parallel execution
    cache: bool = True         # Enable result caching  
    profile: bool = False      # Force profiling
    max_workers: Optional[int] = None  # Limit parallelism
    
    # No scheduler options
    # No strategy selection
    # No execution modes
    # Just simple on/off switches
```

## Usage Examples

### Basic Usage (90% of users)
```python
from ember.xcs import jit

@jit
def classify_texts(texts):
    embeddings = [embed(text) for text in texts]
    scores = [score(emb) for emb in embeddings]
    return scores

# Automatically parallelized!
results = classify_texts(["hello", "world"])
```

### Advanced Usage (9% of users)
```python
from ember.xcs import jit
from ember.xcs.config import Config

@jit(config=Config(cache=False, profile=True))
def process_sensitive_data(data):
    # Don't cache sensitive data
    return model(data)
```

### Expert Usage (1% of users)
```python
# Only if you really need it
from ember.xcs.advanced import ExecutionEngine, IRTransform

# Custom IR transformation
class MyTransform(IRTransform):
    def transform(self, ir_graph):
        # Expert-level control
        pass

# But this is hidden in advanced submodule
```

## What's Hidden

1. **Schedulers**: Automatic selection based on graph
2. **Strategies**: System picks the best one
3. **IR Details**: Users never see nodes/graphs
4. **Parallelism Detection**: Automatic from structure
5. **Optimization Choices**: Made by the system

## Error Messages

```python
# Bad (exposes internals):
"ParallelScheduler failed: node_5 depends on node_3"

# Good (user-friendly):
"Cannot parallelize 'process_text' because each result depends on the previous one"
```

## Performance Feedback

```python
# Optional environment variable for insights
# XCS_PROFILE=1 python my_script.py

# Output:
# classify_texts: 120ms total
#   - Parallel sections: 80ms (67% efficiency)
#   - Sequential sections: 40ms
#   - Suggestion: Consider batching inputs for better performance
```

## Migration Path

```python
# Old code (still works):
from ember.xcs import jit, JITMode
@jit(mode=JITMode.TRACE)
def old_function(x):
    return x

# New code (simpler):
from ember.xcs import jit
@jit
def new_function(x):
    return x
```

## Key Design Decisions

1. **No Modes**: System picks the best approach
2. **No Options**: Smart defaults for everything
3. **No Leaks**: Implementation details hidden
4. **Just Works**: Like Jobs would want
5. **Fast by Default**: Like Carmack would build

This is what Dean and Ghemawat would build: simple on the outside, smart on the inside.