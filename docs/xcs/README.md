# Ember XCS: Smart Execution Made Simple

XCS provides automatic optimization with zero configuration. Just use `@jit` and let XCS handle the rest.

## The Complete API

```python
from ember.api.xcs import jit

@jit
def process(data):
    return model(data)

# That's it! Automatic parallelization, caching, and optimization.
```

## What XCS Does For You

When you add `@jit` to a function or operator:

1. **Automatic Parallelization** - Identifies and executes independent operations in parallel
2. **Smart Caching** - Caches results intelligently based on inputs
3. **JIT Compilation** - Optimizes execution paths based on actual usage patterns
4. **Resource Management** - Handles memory and compute resources efficiently

## Examples

### Basic Usage

```python
from ember.api import ember
from ember.api.xcs import jit

@ember.op
@jit
async def analyze_documents(documents: list[str]) -> list[dict]:
    """Analyze multiple documents with automatic optimization."""
    return await ember.parallel([
        analyze_single(doc) for doc in documents
    ])
```

### Performance Monitoring

```python
from ember.api.xcs import jit, get_jit_stats

@jit
def expensive_operation(data):
    # Complex processing
    return result

# Run your operations
for item in dataset:
    expensive_operation(item)

# Check performance stats
stats = get_jit_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Average execution time: {stats['avg_time_ms']}ms")
```

### Advanced Configuration (For the 10%)

If you need more control:

```python
from ember.api.xcs import jit, Config

# Disable caching for sensitive data
@jit(config=Config(cache=False))
def process_sensitive(data):
    return secure_model(data)

# Custom cache size
@jit(config=Config(cache_size=1000))
def process_large_dataset(item):
    return transform(item)
```

## When to Use @jit

Use `@jit` when you have:
- Functions called repeatedly with similar inputs
- Operations that can benefit from parallelization
- Complex pipelines that need optimization

Don't use `@jit` for:
- One-time operations
- Functions with side effects
- I/O bound operations (already optimized)

## How It Works

XCS uses sophisticated tracing and analysis to understand your code's structure and data flow. It then automatically:

1. Identifies parallelizable operations
2. Builds an optimized execution graph
3. Caches results when beneficial
4. Adapts to your actual usage patterns

All of this happens automatically - you just add `@jit`.

## Migration from Old XCS

If you're using the old XCS API:

**Before:**
```python
from ember.xcs.graph import Graph
from ember.xcs.engine import execute_graph
from ember.xcs.engine.execution_options import execution_options

graph = Graph()
# ... build graph ...
with execution_options(parallel=True):
    result = execute_graph(graph)
```

**After:**
```python
from ember.api.xcs import jit

@jit
def my_pipeline(data):
    # Your pipeline logic
    return result
```

That's it! The new API handles all the complexity for you.

## Best Practices

1. **Add @jit to Hot Paths** - Functions called frequently benefit most
2. **Keep Functions Pure** - Avoid side effects for best optimization
3. **Monitor Performance** - Use `get_jit_stats()` to verify improvements
4. **Start Simple** - Just use `@jit`, add Config only if needed

## Further Reading

- [Performance Guide](./PERFORMANCE_GUIDE.md) - Optimization tips
- [JIT Overview](./JIT_OVERVIEW.md) - Technical details
- [Migration Guide](./MIGRATION_GUIDE.md) - Migrating from old XCS