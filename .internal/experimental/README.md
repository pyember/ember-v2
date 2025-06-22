# Experimental Features

This directory contains experimental code that is:
- Not part of the public API
- Subject to change or removal without notice
- Not covered by semantic versioning guarantees
- May have performance implications

## Current Experiments

### tracing.py
Simple execution tracing for debugging and analysis. This was extracted from module_v4.py to keep the core module system simple and focused.

**Why it's experimental:**
- Adds overhead to function execution
- May not scale to production workloads  
- Design is still evolving
- Separate libraries (OpenTelemetry, etc.) may be better for production use

**Usage (not recommended for production):**
```python
from .internal_docs.experimental.tracing import trace

@trace
def my_function(x):
    return x * 2

result = my_function(5)
print(trace.summary())
```

## Design Philosophy

The main codebase follows Dean/Ghemawat/Martin/Jobs principles of radical simplicity. Features move here when they:
- Add complexity without clear benefit for most users
- Violate the "one way to do things" principle
- Have performance implications
- Are still being designed/tested

## Contributing

If you're interested in these experimental features:
1. Test them thoroughly in non-production environments
2. Provide feedback via issues
3. Understand they may change or disappear
4. Don't depend on them for production systems