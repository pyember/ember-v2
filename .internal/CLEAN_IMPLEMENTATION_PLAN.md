# Clean Implementation Plan

## The Problem

The current code is messy because we're trying to maintain compatibility with multiple calling conventions:
- Old operators expecting `inputs` keyword argument
- New functions with natural signatures
- Special cases for single vs multiple dependencies
- Introspection to detect function signatures

This violates the principle of "one obvious way to do things."

## The Solution

### Core Principles

1. **Functions are just functions** - no special calling conventions
2. **Explicit over implicit** - dependencies are clear
3. **Simple types** - no custom objects when primitives work
4. **No magic** - no introspection, no hidden behavior

### Clean API Design

```python
# Functions have natural signatures
def process(data: Dict) -> Dict:
    return {"result": data["value"] * 2}

# Dependencies determine how functions are called
graph = Graph()
n1 = graph.add(load_data)       # Called with initial inputs
n2 = graph.add(process, deps=[n1])  # Called with n1's result
n3 = graph.add(combine, deps=[n1, n2])  # Called with dict of results

# Clear execution
results = graph.execute({"input": "data"})
```

### Calling Convention Rules

1. **No dependencies**: Function receives initial inputs
   ```python
   def source(inputs: Dict) -> Any:
       return inputs["data"]
   ```

2. **Single dependency**: Function receives that node's result directly
   ```python
   def transform(data: Any) -> Any:
       return data * 2
   ```

3. **Multiple dependencies**: Function receives dict mapping node_id to result
   ```python
   def combine(deps: Dict[str, Any]) -> Any:
       return deps["node1"] + deps["node2"]
   ```

### Migration Strategy

Instead of trying to support old code, we should:

1. **Create clean implementation** (done: `clean_graph.py`)
2. **Provide migration guide** showing how to update old code
3. **Update all internal usage** to use clean API
4. **Delete compatibility layers**

### Example Migration

**Old (messy)**:
```python
class MyOperator:
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": inputs["value"] * 2}

# Complex graph building
graph = Graph()
node = graph.add_node(MyOperator(), name="process")
```

**New (clean)**:
```python
def process(value: int) -> int:
    return value * 2

# Simple graph building
graph = Graph()
node = graph.add(process)
```

## Benefits

1. **Simpler code** - No introspection, no special cases
2. **Better types** - Functions have natural type signatures
3. **Easier testing** - Functions are just functions
4. **Clearer semantics** - Dependencies determine calling convention
5. **Better performance** - No runtime introspection

## Implementation Steps

1. ✅ Create `clean_graph.py` with no compatibility
2. ✅ Create clean examples showing proper usage
3. ⏳ Update test framework to use clean API
4. ⏳ Migrate internal code to clean API
5. ⏳ Delete old graph implementation
6. ⏳ Delete compatibility layers

## The End Result

```python
from ember.xcs import Graph

# Build
graph = Graph()
n1 = graph.add(preprocess)
n2 = graph.add(analyze, deps=[n1])
n3 = graph.add(postprocess, deps=[n2])

# Execute
results = graph.execute({"data": input_data})
```

No options, no strategies, no compatibility. Just a clean DAG executor.