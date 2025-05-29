# XCS Migration Guide: Moving to Simplified Architecture

## Overview

We've created simplified versions of key files that remove complexity while preserving functionality:

1. `src/ember/xcs/__init__simplified.py` - Clean XCS exports
2. `src/ember/xcs/graph/__init__simplified.py` - Just Graph and Node
3. `src/ember/api/xcs_simplified.py` - Clean API surface

## Migration Steps

### Step 1: Replace Core Files

```bash
# Backup originals
mv src/ember/xcs/__init__.py src/ember/xcs/__init__old.py
mv src/ember/xcs/graph/__init__.py src/ember/xcs/graph/__init__old.py
mv src/ember/api/xcs.py src/ember/api/xcs_old.py

# Use simplified versions
mv src/ember/xcs/__init__simplified.py src/ember/xcs/__init__.py
mv src/ember/xcs/graph/__init__simplified.py src/ember/xcs/graph/__init__.py
mv src/ember/api/xcs_simplified.py src/ember/api/xcs.py
```

### Step 2: Update Imports

**Find and replace:**
```python
# Old
from ember.xcs.graph import Graph
from ember.xcs.graph.xcs_graph import Graph

# New
from ember.xcs.graph import Graph
from ember.xcs.graph.graph import Graph
```

### Step 3: Update JIT Usage

**Old pattern:**
```python
@jit(mode="structural")  # Doesn't actually optimize
def my_function():
    pass
```

**New pattern:**
```python
# For optimization (Operators only)
@jit
class MyOperator(Operator):
    def forward(self, inputs):
        # Structural analysis optimizes this
        pass

# For analysis
@trace(print_summary=True)
def my_function():
    # Understand execution
    pass
```

### Step 4: Update Tests

1. Remove tests for removed features:
   - Trace JIT strategy tests
   - Complex strategy selection tests
   - Graph-specific tests

2. Add tests for new features:
   - Simplified Graph tests
   - @trace decorator tests
   - Clear JIT behavior tests

## What Changes for Users

### Before
```python
from ember import xcs

# Confusing - what strategy will it use?
@xcs.jit
def my_func():
    pass

# Complex graph API
graph = xcs.Graph()
graph.add_node(...)
```

### After
```python
from ember import xcs

# Clear - only optimizes Operators
@xcs.jit
class MyEnsemble(Operator):
    pass

# For debugging
@xcs.trace
def my_func():
    pass

# Simple graph API
graph = xcs.Graph()
node_id = graph.add(func, deps=[])
results = graph.run(inputs)
```

## Benefits

1. **Clearer Mental Model**: JIT optimizes, trace analyzes
2. **Less Code**: ~70% reduction in XCS codebase
3. **Better Performance**: Focus on what actually works
4. **Simpler API**: Fewer concepts to learn

## Compatibility

For temporary backward compatibility, keep a shim:

```python
# In xcs/__init__.py
Graph = Graph  # Alias for compatibility

def jit(*args, mode=None, **kwargs):
    if mode == "trace":
        warnings.warn("trace mode deprecated, use @trace for analysis")
    return simplified_jit(*args, **kwargs)
```

## Timeline

1. **Week 1**: Replace core files, test internally
2. **Week 2**: Update all tests
3. **Week 3**: Update documentation
4. **Week 4**: Deprecation warnings
5. **Week 5-8**: Remove old code

This migration simplifies XCS while preserving all useful functionality.