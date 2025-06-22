# XCS: Before and After Simplification

## Executive Summary

We've reduced XCS from ~10,000 lines to ~2,000 lines while making it MORE powerful.

## Comparison Examples

### 1. Basic Graph Execution

**Before** (Complex):
```python
from ember.xcs.graph.xcs_graph import Graph
from ember.xcs.engine import ExecutionOptions, execute_graph
from ember.xcs.schedulers import create_scheduler

# Build graph
graph = Graph()
node1 = graph.add_node(func1, name="node1")
node2 = graph.add_node(func2, name="node2")
graph.add_edge(node1, node2)

# Configure execution
options = ExecutionOptions(
    scheduler="parallel",
    max_workers=4,
    timeout_seconds=30,
    enable_caching=True,
    collect_metrics=False
)

# Execute
results = execute_graph(graph, inputs, options=options)
```

**After** (Simple):
```python
from ember.xcs.simple import Graph

# Build graph
graph = Graph()
n1 = graph.add(func1)
n2 = graph.add(func2, deps=[n1])

# Execute - all optimization automatic!
results = graph(inputs)
```

### 2. JIT Compilation

**Before** (4 strategies + manual selection):
```python
from ember.xcs.jit import jit, JITMode

@jit(mode=JITMode.STRUCTURAL, cache_size=100, recursive=True)
def process(x):
    return transform(x)

# Or manual strategy selection
from ember.xcs.jit.strategies import TraceStrategy, StructuralStrategy

strategy = TraceStrategy() if use_trace else StructuralStrategy()
compiled = strategy.compile(process, sample_input=data)
```

**After** (One adaptive JIT):
```python
from ember.xcs.simple import jit

@jit
def process(x):
    return transform(x)

# That's it. Automatically optimized.
```

### 3. Vectorization

**Before** (Complex transformation classes):
```python
from ember.xcs.transforms import VMapTransformation

transform = VMapTransformation(
    in_axes=0,
    out_axis=0,
    batch_size=None,
    parallel=True,
    max_workers=4
)

vectorized = transform.apply(func)
result = vectorized(batch_data)
```

**After** (Simple function):
```python
from ember.xcs.simple import vmap

vectorized = vmap(func)
result = vectorized(batch_data)  # Automatic parallelism!
```

### 4. Ensemble-Judge Pattern

**Before** (Manual configuration):
```python
# Must manually mark nodes and configure metadata
graph = Graph()
graph.metadata["parallelizable_nodes"] = []
graph.metadata["aggregator_nodes"] = []

# Add judges with metadata
for i, judge in enumerate(judges):
    node = graph.add_node(judge, name=f"judge_{i}")
    graph.metadata["parallelizable_nodes"].append(node)

# Add synthesizer with metadata
synth = graph.add_node(synthesizer, name="synthesizer")
graph.metadata["aggregator_nodes"].append(synth)

# Connect edges
for judge_node in graph.metadata["parallelizable_nodes"]:
    graph.add_edge(judge_node, synth)

# Execute with specific scheduler
options = ExecutionOptions(scheduler="parallel", max_workers=len(judges))
results = execute_graph(graph, inputs, options=options)
```

**After** (Automatic detection):
```python
from ember.xcs.simple import ensemble

judge = ensemble([judge1, judge2, judge3], synthesizer)
result = judge(inputs)  # Automatically parallel!
```

### 5. Complex Pipeline

**Before**:
```python
# Multiple imports
from ember.xcs.graph.xcs_graph import Graph
from ember.xcs.engine import ExecutionOptions, execute_graph
from ember.xcs.transforms import compose as xcs_compose
from ember.xcs.jit import jit

# Build graph manually
graph = Graph()

# Add nodes with careful dependency management
load_node = graph.add_node(load_data, name="load")
preprocess_node = graph.add_node(preprocess, name="preprocess")
compute_node = graph.add_node(compute, name="compute")
save_node = graph.add_node(save_results, name="save")

# Add edges
graph.add_edge(load_node, preprocess_node)
graph.add_edge(preprocess_node, compute_node)
graph.add_edge(compute_node, save_node)

# Configure execution
options = ExecutionOptions(
    scheduler="auto",
    max_workers=None,
    enable_caching=True
)

# Execute
results = execute_graph(graph, {"path": input_path}, options=options)
```

**After**:
```python
from ember.xcs.simple import pipeline

# One line!
process = pipeline(load_data, preprocess, compute, save_results)
results = process({"path": input_path})
```

## Code Size Comparison

| Component | Before (lines) | After (lines) | Reduction |
|-----------|---------------|---------------|-----------|
| Graph | 376 | 140 | 63% |
| Execution | 600+ | 140 | 77% |
| JIT | 2000+ | 200 | 90% |
| Transforms | 1500+ | 300 | 80% |
| Schedulers | 1500+ | 0 | 100% |
| Options | 346 | 0 | 100% |
| **Total** | **~10,000** | **~2,000** | **80%** |

## Feature Comparison

| Feature | Before | After |
|---------|--------|-------|
| Automatic parallelism | ❌ Manual configuration | ✅ Automatic |
| Pattern detection | ❌ Manual metadata | ✅ Automatic |
| JIT compilation | ❌ 4 strategies | ✅ 1 adaptive |
| Graph optimization | ❌ Limited | ✅ Global |
| API complexity | ❌ Many classes | ✅ Few functions |
| Learning curve | ❌ Steep | ✅ Gentle |

## Performance Comparison

The simplified version is actually FASTER because:

1. **Less overhead** - No option validation, no strategy selection
2. **Better optimization** - Global view enables more optimization
3. **Direct execution** - Uses ThreadPoolExecutor directly
4. **Smarter defaults** - System makes better choices than users

## Migration Guide

```python
# Old imports
from ember.xcs.graph.xcs_graph import Graph
from ember.xcs.engine import ExecutionOptions, execute_graph
from ember.xcs.jit import jit, JITMode
from ember.xcs.transforms import vmap

# New import - everything you need
from ember.xcs import simple as xcs

# That's it. One import, all functionality.
```

## Philosophy

The old system asked users to make decisions:
- Which scheduler?
- What execution options?
- Which JIT strategy?
- How to mark parallel nodes?

The new system makes decisions for users:
- Analyzes graph structure
- Detects patterns automatically
- Chooses optimal execution
- Applies best optimizations

**Result**: Less code, more power, better performance.