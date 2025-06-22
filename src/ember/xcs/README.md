# XCS: Smart Execution Made Simple

XCS (Cross-cutting System) provides intelligent JIT compilation and parallelization for Ember operations, handling both JAX tensor operations and orchestration/model calls seamlessly.

## Philosophy

Following the principles of simplicity and power:
- **Zero configuration**: Just use `@jit` and it works
- **Automatic optimization**: Discovers parallelism from your code structure
- **Intelligent routing**: Knows when to use JAX transformations vs parallel orchestration
- **Progressive disclosure**: Simple API for the 90%, power features for the 10%

## Quick Start

```python
from ember.xcs import jit

@jit
def my_function(x):
    return model(x)

# That's it! Automatic optimization with zero configuration.
```

## Core Features

### Automatic Parallelism Detection
XCS analyzes your code to find parallelization opportunities automatically:
- Independent branches execute in parallel
- Vectorizable operations are batched efficiently
- Orchestration calls are parallelized when possible

### Intelligent Transformation Routing
XCS transformations subsume JAX transformations while adding orchestration intelligence:
- `vmap`: Batches tensor ops with JAX, parallelizes orchestration ops
- `pmap`: Distributes across devices for tensors, across workers for orchestration
- `scan`: Sequential processing with state accumulation
- `grad`: Automatic differentiation for tensor operations

### Smart Caching and Profiling
- Automatic result caching for expensive operations
- Built-in profiling to identify bottlenecks
- Performance statistics available via `get_jit_stats()`

## Architecture

The new simplified XCS consists of:

### Public API (`__init__.py`, `transformations.py`)
- `jit`, `vmap`, `pmap`, `scan`, `grad` - The main transformation functions
- `get_jit_stats()` - Performance monitoring

### Simple JIT Implementation (`_simple.py`)
- Core `@jit` decorator implementation
- Automatic strategy selection
- Zero-configuration optimization

### Configuration (`config.py`)
- Simple config object for advanced users
- Just on/off switches, no complex options
- Sensible defaults that work for 90% of cases

### Internal Implementation (`_internal/`)
- `analysis.py` - Operation type detection (tensor vs orchestration)
- `ir.py` - Intermediate representation for computation graphs
- `parallelism.py` - Parallelism detection algorithms
- `engine.py` - Execution engine
- `profiler.py` - Performance profiling
- `tracer.py` - Function tracing
- `ir_builder.py` - Graph construction
- `pytree_registration.py` - JAX pytree compatibility

## Advanced Usage

For the 10% who need more control:

```python
from ember.xcs import jit
from ember.xcs.config import Config

# Disable caching for sensitive data
@jit(config=Config(cache=False))
def process_private_data(data):
    return secure_model(data)

# Force profiling
@jit(config=Config(profile=True))
def analyze_performance(data):
    return complex_operation(data)

# Limit parallelism
@jit(config=Config(max_workers=2))
def resource_constrained(items):
    return [process(item) for item in items]
```

## Implementation Notes

### O(n) Algorithms
The parallelism detection uses efficient algorithms:
- Dependency analysis: O(V + E) using index structures
- Parallel group detection: O(V + E) using depth-based analysis
- Topological sort: O(V + E) using Kahn's algorithm

### JAX Integration
- Ember operators are automatically registered as JAX pytrees
- Seamless integration with JAX transformations
- No static array warnings

### Error Handling
Simple error model with one exception type:
- `XCSError` - Raised for any XCS-specific errors

## Migration from Legacy

The legacy XCS implementation with complex graph building, multiple strategies, and extensive configuration has been moved to `.internal_docs/xcs_legacy/` for reference. The new implementation provides the same functionality with 10% of the complexity.

Key differences:
- No manual graph building needed
- No strategy selection - system chooses automatically
- No scheduler configuration - intelligent defaults
- No complex exception hierarchy - just `XCSError`

## Future Enhancements

Planned improvements while maintaining simplicity:
- Automatic memory management for large operations
- GPU/TPU placement optimization
- Distributed orchestration across multiple machines
- Advanced profiling visualizations

Remember: The best API is no API. Just use `@jit` and let XCS handle the rest.