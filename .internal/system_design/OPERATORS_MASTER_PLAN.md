# Operators Master Plan: Balancing SOLID with Radical Simplicity

## Core Philosophy

Drawing from our mentors:
- **Ritchie**: "Keep it simple, make it general"
- **Jobs**: "Simplicity is the ultimate sophistication"
- **Carmack**: "If you're not sure what to do, do the simplest thing"
- **Martin**: "The only way to go fast is to go well"

## The Fundamental Insight

**Functions are operators.** Everything else is optional enhancement.

## Design Principles

### 1. Single Responsibility (But Not Too Many Classes)
- Each module has ONE clear purpose
- But we don't create classes just to follow SRP
- A function that does one thing well IS single responsibility

### 2. Open/Closed Through Protocols
- Open for extension via protocols and decorators
- Closed for modification - core stays simple
- No inheritance hierarchies

### 3. Interface Segregation via Progressive Disclosure
- Level 1: Just functions (no interface needed)
- Level 2: Optional validation (minimal interface)
- Level 3: Full specifications (complete interface)
- Users only see what they need

### 4. Dependency Inversion Without Abstractions
- Depend on protocols, not base classes
- But protocols are optional - duck typing works
- The abstraction is the function signature

### 5. Liskov Substitution Naturally
- Any callable can be an operator
- All operators compose the same way
- No special cases or exceptions

## Module Architecture

```
src/ember/core/operators/
├── __init__.py          # Public API (minimal surface)
├── core.py              # Core concepts (functions are operators)
├── compose.py           # Composition patterns (chain, parallel, etc)
├── validate.py          # Optional validation via decorators
├── enhance.py           # Progressive enhancement (one function)
├── streaming.py         # Stream/generator support
├── async_ops.py         # Async operator support
└── _specification.py    # Advanced: EmberModel support (private module)
```

## Detailed Design

### 1. **core.py** - The Heart (What Ritchie Would Write)
```python
"""Core operator concepts.

The fundamental insight: functions are operators.
Everything else builds on this.
"""

from typing import TypeVar, Protocol, runtime_checkable

T = TypeVar('T')
S = TypeVar('S')

@runtime_checkable
class Operator(Protocol[T, S]):
    """Something that transforms T to S."""
    def __call__(self, input: T) -> S: ...

# That's it. No base classes. No magic.
```

### 2. **compose.py** - Composition (What Dean & Ghemawat Would Build)
```python
"""Operator composition with proper error handling and monitoring."""

from typing import Callable, List, Any, Optional
import time
from contextlib import contextmanager

class CompositionError(Exception):
    """Raised when composition fails, with full context."""
    pass

@contextmanager
def _monitor_operator(name: str, monitor: Optional[Callable] = None):
    """Monitor operator execution if requested."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if monitor:
            monitor(name, time.perf_counter() - start)

def chain(*ops: Callable, monitor: Optional[Callable] = None) -> Callable:
    """Sequential composition with optional monitoring.
    
    What Dean would add: performance monitoring hooks.
    What Carmack would ensure: zero overhead when not used.
    """
    if not ops:
        raise ValueError("chain requires at least one operator")
    
    def chained(x):
        result = x
        for i, op in enumerate(ops):
            name = getattr(op, '__name__', f'op{i}')
            with _monitor_operator(name, monitor):
                try:
                    result = op(result)
                except Exception as e:
                    raise CompositionError(
                        f"Failed at {name} with input {repr(result)[:50]}"
                    ) from e
        return result
    
    # What Knuth would add: self-documenting names
    names = [getattr(op, '__name__', '?') for op in ops]
    chained.__name__ = f"chain({' → '.join(names)})"
    return chained

def parallel(*ops: Callable, max_workers: int = None) -> Callable:
    """True parallel execution.
    
    What Carmack would say: Don't lie. If it's not parallel, don't call it that.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def paralleled(x):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(op, x) for op in ops]
            return [f.result() for f in futures]
    
    return paralleled
```

### 3. **validate.py** - Optional Validation (What Jobs Would Approve)
```python
"""Simple validation - one way to do it."""

from typing import Type, Optional, Callable, TypeVar
from functools import wraps

F = TypeVar('F', bound=Callable)

def validate(input: Type = None, output: Type = None) -> Callable[[F], F]:
    """Validate types at runtime - but only if you ask.
    
    What Jobs would say: One decorator, clear purpose, no confusion.
    What Martin would add: Single responsibility - just validation.
    """
    def decorator(func: F) -> F:
        if not (input or output):  # No validation requested
            return func
            
        @wraps(func)
        def wrapper(x, *args, **kwargs):
            # Validate input
            if input and not isinstance(x, input):
                raise TypeError(
                    f"{func.__name__} expected {input.__name__}, "
                    f"got {type(x).__name__}"
                )
            
            # Execute
            result = func(x, *args, **kwargs)
            
            # Validate output
            if output and not isinstance(result, output):
                raise TypeError(
                    f"{func.__name__} should return {output.__name__}, "
                    f"got {type(result).__name__}"
                )
            
            return result
        
        return wrapper
    return decorator
```

### 4. **enhance.py** - Progressive Enhancement (What Brockman Would Design)
```python
"""One function to rule them all - progressive enhancement."""

from typing import Any, Dict, Callable
import time
from dataclasses import dataclass, field

@dataclass
class Enhanced:
    """Enhanced operator with all capabilities."""
    operator: Callable
    
    # Capabilities (all optional)
    batch_size: Optional[int] = None
    cost_per_call: Optional[float] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    
    # Metrics (auto-collected)
    _calls: int = field(default=0, init=False)
    _errors: int = field(default=0, init=False)
    _total_time: float = field(default=0.0, init=False)
    
    def __call__(self, x):
        """Execute with all enhancements."""
        start = time.perf_counter()
        try:
            self._calls += 1
            return self.operator(x)
        except Exception as e:
            self._errors += 1
            raise
        finally:
            self._total_time += time.perf_counter() - start
    
    def batch_forward(self, items: List[Any]) -> List[Any]:
        """Batch processing if requested."""
        if self.batch_size:
            # Process in batches
            results = []
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]
                results.extend([self(x) for x in batch])
            return results
        return [self(x) for x in items]
    
    @property
    def metrics(self) -> Dict[str, float]:
        """Performance metrics."""
        return {
            'calls': self._calls,
            'errors': self._errors,
            'error_rate': self._errors / max(self._calls, 1),
            'total_time': self._total_time,
            'avg_time': self._total_time / max(self._calls, 1)
        }

def enhance(operator: Callable, **capabilities) -> Enhanced:
    """One function for all enhancements.
    
    What Jobs would love: One way to enhance operators.
    What Brockman would ensure: Great developer experience.
    """
    return Enhanced(operator, **capabilities)
```

### 5. **streaming.py** - Unix Philosophy (What Ritchie Would Build)
```python
"""Stream processing - compose like Unix pipes."""

from typing import Iterator, Callable, TypeVar

T = TypeVar('T')
S = TypeVar('S')

def stream_map(op: Callable[[T], S]) -> Callable[[Iterator[T]], Iterator[S]]:
    """Transform a stream of values.
    
    What Ritchie would say: This is just map for streams.
    Keep it simple, make it composable.
    """
    def mapper(stream: Iterator[T]) -> Iterator[S]:
        return (op(item) for item in stream)
    return mapper

def stream_filter(predicate: Callable[[T], bool]) -> Callable[[Iterator[T]], Iterator[T]]:
    """Filter a stream of values."""
    def filterer(stream: Iterator[T]) -> Iterator[T]:
        return (item for item in stream if predicate(item))
    return filterer

def stream_chain(*ops: Callable) -> Callable[[Iterator], Iterator]:
    """Chain streaming operators - like Unix pipes."""
    def chained(stream):
        for op in ops:
            stream = op(stream)
        return stream
    return chained

# Usage: Just like Unix!
# pipeline = stream_chain(
#     stream_map(str.lower),
#     stream_filter(lambda x: len(x) > 3),
#     stream_map(str.upper)
# )
# results = pipeline(word_stream)
```

### 6. **async_ops.py** - Modern Concurrency (What Modern Dean Would Add)
```python
"""Async operator support - because it's 2024."""

import asyncio
from typing import Callable, Awaitable, Union

AsyncOperator = Callable[[Any], Awaitable[Any]]
MixedOperator = Union[Callable, AsyncOperator]

async def achain(*ops: MixedOperator) -> AsyncOperator:
    """Chain that handles both sync and async operators."""
    async def chained(x):
        result = x
        for op in ops:
            if asyncio.iscoroutinefunction(op):
                result = await op(result)
            else:
                result = op(result)
        return result
    return chained

async def aparallel(*ops: MixedOperator) -> AsyncOperator:
    """True async parallel execution."""
    async def paralleled(x):
        tasks = []
        for op in ops:
            if asyncio.iscoroutinefunction(op):
                tasks.append(op(x))
            else:
                # Run sync in executor
                tasks.append(
                    asyncio.get_event_loop().run_in_executor(None, op, x)
                )
        return await asyncio.gather(*tasks)
    return paralleled
```

### 7. **_specification.py** - Advanced Use (What Martin Would Isolate)
```python
"""Advanced EmberModel support - hidden but available.

This is a private module. Users should prefer simple functions
and only use this for complex enterprise requirements.
"""

# Import existing Specification
from ember.core.registry.specification import Specification as _BaseSpec
from ember.core.types.ember_model import EmberModel

class Specification(_BaseSpec):
    """EmberModel-based validation for complex cases.
    
    Hidden in private module to discourage overuse.
    What Martin would say: Separate the complex from the simple.
    """
    
    def to_simple_operator(self, impl: Callable) -> Callable:
        """Convert specification-based operator to simple function."""
        def operator(inputs: dict) -> dict:
            validated = self.parse_inputs(inputs)
            result = impl(validated)
            return self.parse_output(result)
        return operator

# One function to convert when needed
def from_specification(spec: Specification, impl: Callable) -> Callable:
    """Bridge from complex to simple world."""
    return spec.to_simple_operator(impl)
```

## Public API Design

```python
# src/ember/api/operators.py

"""Operators API - Functions are operators.

Simple things should be simple, complex things should be possible.
- Alan Kay (quoted by Jobs)
"""

# Level 1: Just functions (90% of users stop here)
from ember.core.operators import Operator  # Just the protocol

# Level 2: Composition (everyone needs this)
from ember.core.operators.compose import chain, parallel

# Level 3: Optional enhancements (when needed)
from ember.core.operators.validate import validate
from ember.core.operators.enhance import enhance

# Level 4: Advanced patterns (power users)
from ember.core.operators.streaming import (
    stream_map, stream_filter, stream_chain
)
from ember.core.operators.async_ops import achain, aparallel

__all__ = [
    # Core (just one thing)
    'Operator',
    
    # Composition (the essentials)
    'chain',
    'parallel',
    
    # Enhancement (one function)
    'validate',
    'enhance',
    
    # Advanced (when needed)
    'stream_map',
    'stream_filter', 
    'stream_chain',
    'achain',
    'aparallel',
]

# Note: Specification is NOT exported. Use simple functions.
```

## Migration Strategy

### Phase 1: Core Implementation
1. Implement core.py (30 min)
2. Implement compose.py with monitoring (2 hours)
3. Implement validate.py - ONE way (1 hour)
4. Implement enhance.py - unified enhancement (2 hours)

### Phase 2: Advanced Features
1. Implement streaming.py (2 hours)
2. Implement async_ops.py (2 hours)
3. Move Specification to _specification.py (1 hour)

### Phase 3: Testing & Documentation
1. Comprehensive test suite (4 hours)
2. Performance benchmarks (2 hours)
3. Migration guide (1 hour)
4. Operator cookbook (2 hours)

## What Each Mentor Would Say

- **Dean & Ghemawat**: "Good performance monitoring, clean error handling"
- **Jobs**: "One way to enhance, no choice paralysis"
- **Brockman**: "Great API, easy to understand"
- **Ritchie**: "Simple, composable, Unix-like"
- **Knuth**: "Well documented, tested, literate"
- **Carmack**: "No lies, actual parallelism, minimal overhead"
- **Martin**: "SOLID without over-engineering"

## Review Against CLAUDE.md

✅ **Principled, root-node fixes**: Functions as operators is fundamental
✅ **Google Python Style Guide**: All modules properly documented
✅ **No Claude references**: Clean, professional code
✅ **Opinionated decisions**: ONE way to validate, ONE way to enhance
✅ **Explicit over magic**: No metaclasses, clear behavior
✅ **Common case design**: Simple functions work immediately
✅ **Professional documentation**: Technical, clear, no emojis
✅ **Comprehensive testing**: Non-negotiable, will be complete

## The Result

A system that is:
- **Simple by default**: Just use functions
- **Powerful when needed**: Full capabilities available
- **SOLID but not rigid**: Principles without dogma
- **Fast and monitored**: Performance built in
- **Thoroughly tested**: Correctness guaranteed

This is what happens when Dean, Ghemawat, Jobs, Brockman, Ritchie, Knuth, Carmack, and Martin pair program: radical simplicity with hidden power.