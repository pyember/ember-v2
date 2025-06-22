# XCS Implementation Quick Start Guide

## For Developers Starting Implementation

### Start Here (Day 1)

The most critical piece is the tracer. Without it, nothing else works.

```python
# ember/xcs/_internal/tracer.py
import sys
from typing import Any, Callable, Dict, List

class Operation:
    """Recorded operation during tracing."""
    def __init__(self, func: Callable, args: tuple, kwargs: dict, result: Any):
        self.func = func
        self.args = args
        self.kwargs = kwargs  
        self.result = result

class PythonTracer:
    """Traces Python execution using sys.settrace."""
    
    def __init__(self):
        self.operations: List[Operation] = []
        self.tracing = False
    
    def trace_function(self, func: Callable, args: tuple, kwargs: dict) -> List[Operation]:
        """Trace function execution and return operations."""
        self.operations = []
        self.tracing = True
        
        # Set up tracing
        sys.settrace(self._trace_calls)
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            return self.operations
        finally:
            sys.settrace(None)
            self.tracing = False
    
    def _trace_calls(self, frame, event, arg):
        """Trace callback for sys.settrace."""
        if event == 'call' and self.tracing:
            # Record function calls
            # This is where the magic happens
            pass
        return self._trace_calls
```

### The Core Flow

```
User Function → Tracer → IR Builder → Analyzer → Engine → Parallel Execution
```

Each component has a single job:
1. **Tracer**: Records what the function does
2. **IR Builder**: Builds a graph from recordings
3. **Analyzer**: Finds parallel opportunities
4. **Engine**: Executes with parallelism

### Critical Design Decisions to Remember

1. **NO RETRY LOGIC** - Fail fast, always
2. **NO CONFIGURATION** - Just @jit, nothing else
3. **EXACT SEMANTICS** - Parallel behavior = sequential behavior
4. **PERMANENT DECISIONS** - Tracing failure = permanent disable

### Simplest Test Case

Start with this test case and make it work:

```python
def test_simple_parallel():
    @jit
    def parallel_calls(x):
        a = expensive_op(x)
        b = expensive_op(x + 1)
        c = expensive_op(x + 2)
        return a + b + c
    
    # Should run expensive_op calls in parallel
    result = parallel_calls(10)
    assert result == expected_result
```

### Common Pitfalls to Avoid

1. **Don't add retry logic** - Users handle that
2. **Don't make it configurable** - Simplicity wins
3. **Don't optimize prematurely** - Make it work first
4. **Don't break semantics** - Errors must match sequential

### Implementation Order

1. **Day 1**: Get tracer recording function calls
2. **Day 2**: Build simple IR graph from trace
3. **Day 3**: Detect independent operations
4. **Day 4**: Execute them in parallel
5. **Day 5**: Connect in @jit decorator

Everything else is refinement.

### Testing Strategy

For each component:
1. **Unit test** in isolation
2. **Integration test** with next component
3. **End-to-end test** through @jit

### The One Key Insight

The current code has all the pieces but they're not connected. Your job is to wire them together, not redesign them.

## Remember the Philosophy

From the masters:
- **Dean**: Make it simple
- **Jobs**: No configuration
- **Carmack**: Fail fast
- **Martin**: Single responsibility

Just connect the components and ship it.