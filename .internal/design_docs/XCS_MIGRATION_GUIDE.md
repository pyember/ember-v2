# XCS Migration Guide: From v1 to v2

## Overview

XCS v2 is a complete reimplementation focused on simplicity and automatic optimization. Most code will become simpler after migration.

## Key Changes

### 1. No More Base Classes
**Before (v1):**
```python
from ember.xcs import Operator, Specification

class MyOperator(Operator[InputType, OutputType]):
    specification = Specification(
        name="my_operator",
        inputs=["x", "y"],
        outputs=["result"]
    )
    
    def forward(self, *, inputs: InputType) -> OutputType:
        return inputs.x + inputs.y
```

**After (v2):**
```python
from ember.xcs import jit

@jit
def my_operator(x, y):
    return x + y
```

### 2. No More Manual Graph Building
**Before (v1):**
```python
from ember.xcs.graph import Graph
from ember.xcs.engine import execute_graph, ExecutionOptions

graph = Graph()
node1 = graph.add_node(func1, name="step1")
node2 = graph.add_node(func2, name="step2")
graph.add_edge(node1, node2)

options = ExecutionOptions(scheduler="parallel", max_workers=4)
result = execute_graph(graph, inputs, options=options)
```

**After (v2):**
```python
from ember.xcs import jit

@jit
def pipeline(inputs):
    step1 = func1(inputs)
    step2 = func2(step1)
    return step2

result = pipeline(inputs)  # Parallelism is automatic
```

### 3. No More Execution Options
**Before (v1):**
```python
options = ExecutionOptions(
    scheduler="parallel",
    max_workers=4,
    timeout_seconds=30,
    enable_caching=True,
    cache_size_mb=100
)
result = execute_graph(graph, inputs, options=options)
```

**After (v2):**
```python
result = my_func(inputs)  # Everything is automatic
```

### 4. Simplified Vectorization
**Before (v1):**
```python
from ember.xcs.transforms import VMapTransformation

transform = VMapTransformation(axis=0, parallel=True)
vmapped_op = transform.apply(my_operator)
```

**After (v2):**
```python
from ember.xcs import vmap

@vmap
def my_operator(x):
    return x * 2
```

## Common Migration Patterns

### Pattern 1: Operator to Function
```python
# Old
class AddOperator(Operator):
    def forward(self, *, inputs):
        return inputs.x + inputs.y

# New  
@jit
def add(x, y):
    return x + y
```

### Pattern 2: Graph Building to Direct Code
```python
# Old
graph = Graph()
a = graph.add_node(lambda x: x * 2, name="double")
b = graph.add_node(lambda x: x + 1, name="increment")
graph.add_edge(input_node, a)
graph.add_edge(input_node, b)
c = graph.add_node(lambda x, y: x + y, name="sum")
graph.add_edge(a, c)
graph.add_edge(b, c)

# New
@jit
def compute(x):
    a = x * 2      # These run
    b = x + 1      # in parallel!
    c = a + b
    return c
```

### Pattern 3: Ensemble Operators
```python
# Old
from ember.xcs.operators import EnsembleOperator

ensemble = EnsembleOperator(
    operators=[op1, op2, op3],
    aggregation="mean",
    parallel=True
)

# New
@jit
def ensemble(x):
    # These run in parallel automatically
    r1 = op1(x)
    r2 = op2(x)
    r3 = op3(x)
    return (r1 + r2 + r3) / 3
```

### Pattern 4: Conditional Execution
```python
# Old
from ember.xcs.operators import ConditionalOperator

conditional = ConditionalOperator(
    condition=lambda x: x > 0,
    true_operator=positive_op,
    false_operator=negative_op
)

# New
@jit
def conditional(x):
    if x > 0:
        return positive_op(x)
    else:
        return negative_op(x)
```

## Advanced Migration

### Custom Schedulers → Automatic
```python
# Old: Custom scheduler for parallel execution
scheduler = ParallelScheduler(max_workers=8)
result = execute_graph(graph, inputs, scheduler=scheduler)

# New: Just write the code
@jit
def my_func(inputs):
    # Parallelism is discovered automatically
    results = [process(item) for item in inputs]
    return combine(results)
```

### Caching → Automatic
```python
# Old: Manual cache configuration
options = ExecutionOptions(
    enable_caching=True,
    cache_backend="redis",
    cache_ttl=3600
)

# New: Built-in caching for deterministic operations
@jit
def cached_compute(x):
    return expensive_operation(x)  # Cached if deterministic
```

### Distributed Execution
```python
# Old: Complex distributed setup
from ember.xcs.distributed import DistributedExecutor

executor = DistributedExecutor(
    cluster_config="cluster.yaml",
    scheduling_policy="round_robin"
)

# New: Simple export
from ember.xcs import export, get_graph

graph = get_graph(my_func)
export_data = export(graph)
# Send export_data to any distributed executor
```

## Step-by-Step Migration

1. **Remove all Operator base classes**
   - Convert to simple functions
   - Add @jit decorator

2. **Remove graph building code**
   - Write sequential Python
   - Let XCS discover parallelism

3. **Remove ExecutionOptions**
   - Delete configuration code
   - Trust automatic optimization

4. **Simplify transforms**
   - Replace complex transform classes with @vmap/@pmap
   - Remove manual batching code

5. **Test and benchmark**
   - Verify correctness
   - Measure performance (should be same or better)

## FAQ

**Q: What about my custom operators?**
A: Convert them to functions. If they have state, make them classes with `__call__`.

**Q: How do I control parallelism?**
A: You don't need to. XCS finds optimal parallelism automatically.

**Q: What about GPU execution?**
A: Currently CPU only. GPU support can be added when needed (YAGNI).

**Q: Can I still build graphs manually?**
A: Yes, use the `trace()` function for advanced use cases.

**Q: What about distributed execution?**
A: Use `export()` to get a graph representation, then execute remotely.

## Benefits After Migration

1. **Less Code**: Typically 70-90% reduction
2. **Better Performance**: Automatic optimization
3. **Fewer Bugs**: No manual graph building
4. **Easier Testing**: Just test functions
5. **Better Debugging**: Standard Python debugging tools work

## Example: Complete Migration

**Before (v1):** 150 lines
```python
from ember.xcs import Operator, Graph, execute_graph, ExecutionOptions
from ember.xcs.operators import EnsembleOperator, MapOperator

class PreprocessOperator(Operator):
    specification = Specification(name="preprocess")
    
    def forward(self, *, inputs):
        return inputs.data * 0.9

class Model1Operator(Operator):
    specification = Specification(name="model1")
    
    def forward(self, *, inputs):
        return self.model.predict(inputs)

class Model2Operator(Operator):
    specification = Specification(name="model2")
    
    def forward(self, *, inputs):
        return self.model.predict(inputs)

# Build graph
graph = Graph()
preprocess = graph.add_node(PreprocessOperator())
models = EnsembleOperator([Model1Operator(), Model2Operator()])
ensemble_node = graph.add_node(models)
graph.add_edge(preprocess, ensemble_node)

# Execute
options = ExecutionOptions(scheduler="parallel", enable_caching=True)
result = execute_graph(graph, inputs, options)
```

**After (v2):** 15 lines
```python
from ember.xcs import jit

@jit
def pipeline(data, model1, model2):
    # Preprocess
    preprocessed = data * 0.9
    
    # Models run in parallel automatically
    pred1 = model1.predict(preprocessed)
    pred2 = model2.predict(preprocessed)
    
    # Ensemble
    return (pred1 + pred2) / 2

result = pipeline(data, model1, model2)
```

## Conclusion

Migration to XCS v2 is about deleting code, not writing it. The new system is simpler, faster, and more powerful. Most migrations can be completed in hours, not days.