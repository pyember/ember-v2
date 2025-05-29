# XCS Architecture Migration Guide

This guide helps you migrate from the old XCS architecture to the new simplified version.

## Key Changes

### 1. Graph API Simplification

**Before:**
```python
from ember.xcs.graph import XCSGraph

graph = XCSGraph()
node = graph.add_node(func)
```

**After:**
```python
from ember.xcs.graph import Graph

graph = Graph()
node = graph.add(func)
```

### 2. JIT Strategy Changes

The trace strategy has been removed. Only structural and enhanced strategies remain.

**Before:**
```python
@jit(mode="trace")
def my_function():
    pass
```

**After:**
```python
@jit  # Automatically selects best strategy
def my_function():
    pass
```

### 3. Separate Trace Decorator

For execution analysis and debugging, use the new `@trace` decorator:

```python
from ember.xcs.trace import trace

@trace(print_summary=True)
def my_pipeline(data):
    # Your code here
    return result
```

### 4. Performance Expectations

- JIT now focuses on optimizing I/O-bound operations (like LLM calls)
- CPU-bound operations cannot be parallelized due to Python's GIL
- Use `@jit` for Operators with parallel patterns
- Use `@trace` for execution analysis

### 5. Removed Features

- `JITMode.TRACE` - use `JITMode.STRUCTURAL` instead
- `TraceStrategy` class
- Trace-based compilation options

### 6. API Compatibility

Most code should work without changes. The main updates needed:
- Replace `XCSGraph` with `Graph`
- Remove `mode="trace"` from JIT decorators
- Use `@trace` for analysis instead of JIT tracing

## Example Migration

### Old Code:
```python
from ember.xcs import XCSGraph, jit, JITMode

@jit(mode=JITMode.TRACE)
class MyOperator:
    def forward(self, *, inputs):
        graph = XCSGraph()
        # ... build graph
        return graph.execute(inputs)
```

### New Code:
```python
from ember.xcs import Graph, jit
from ember.xcs.trace import trace

@jit  # Structural analysis for parallelization
class MyOperator:
    def forward(self, *, inputs):
        graph = Graph()
        # ... build graph
        return graph.run(inputs)

# For debugging/analysis:
@trace(print_summary=True)
def analyze_pipeline(data):
    op = MyOperator()
    return op(inputs=data)
```

## Questions?

See the [XCS README](README.md) for more details on the new architecture.
