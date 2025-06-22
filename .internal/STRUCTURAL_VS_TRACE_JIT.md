# Structural vs Trace JIT: When to Use Which

## Overview

The XCS system automatically selects between Structural and structural JIT using a scoring system. Here's how it works and best practices for each.

## Strategy Selection Algorithm

When you use `@xcs.jit` with `mode=JITMode.AUTO` (default), the system scores each strategy:

### Structural JIT Scoring
```python
def analyze(self, func):
    score = 0
    
    # +20: It's a class (likely an Operator)
    if inspect.isclass(func):
        score += 20
        
    # +30: Has forward() method
    if hasattr(func, "forward"):
        score += 30
        
    # +40: Has nested operator fields ← KEY!
    for attr_name in dir(func):
        attr = getattr(func, attr_name)
        if hasattr(attr, "forward"):  # It's an operator
            score += 40
            break
            
    # +10: Has specification attribute
    if hasattr(func, "specification"):
        score += 10
    
    return score  # Max ~100 for ideal operator
```

### Trace JIT Scoring
```python
def analyze(self, func):
    score = 0
    
    # +30: Simple function (< 20 lines)
    if source_lines < 20:
        score += 30
        
    # +20: Simple control flow
    if if_count < 3 and for_count < 2:
        score += 20
        
    # +5: Base score (fallback)
    score += 5
    
    return score  # Max ~55 for simple functions
```

## Key Difference: Static vs Dynamic Analysis

### Structural JIT (Static Analysis)
**Requires operators to be class attributes:**

```python
@xcs.jit  # Will use STRUCTURAL (score ~100)
class EnsembleOperator(Operator):
    def __init__(self):
        # ✅ These MUST be attributes for structural analysis
        self.model1 = ModelA()
        self.model2 = ModelB()
        self.judge = Judge()
    
    def forward(self, *, inputs):
        # Structural JIT finds self.model1, etc. statically
        r1 = self.model1(inputs=inputs)
        r2 = self.model2(inputs=inputs)
        return self.judge(inputs=[r1, r2])
```

**Benefits:**
- No execution needed for analysis
- Finds all possible paths
- Works with conditional operators
- Better for complex operator hierarchies

**Limitations:**
- Operators must be attributes
- Can't handle dynamic operator creation

### Trace JIT (Dynamic Analysis)
**Works with any execution pattern:**

```python
@xcs.jit  # Will use TRACE (structural score low)
def ensemble_function(inputs):
    # ✅ These are created dynamically - trace JIT handles it
    model1 = ModelA()
    model2 = ModelB()
    
    r1 = model1(inputs)
    r2 = model2(inputs)
    
    # Can even have conditional logic
    if inputs.get("use_ensemble"):
        judge = Judge()
        return judge([r1, r2])
    else:
        return r1
```

**Benefits:**
- Handles dynamic operator creation
- Works with conditional logic
- Traces actual execution path
- More flexible

**Limitations:**
- Requires execution to build graph
- Only captures executed path
- May need retracing for different paths

## Best Practices

### 1. For Complex Operator Hierarchies
Use class attributes + structural JIT:
```python
@xcs.jit
class ComplexPipeline(Operator):
    def __init__(self):
        # Structural JIT will find all of these
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor()
        self.ensemble = EnsembleOperator()
        self.postprocessor = Postprocessor()
```

### 2. For Dynamic Pipelines
Let trace JIT handle it:
```python
@xcs.jit
def dynamic_pipeline(inputs, config):
    # Create operators based on config
    models = []
    for model_type in config["models"]:
        if model_type == "A":
            models.append(ModelA())
        elif model_type == "B":
            models.append(ModelB())
    
    # Run them
    results = [m(inputs) for m in models]
    return combine_results(results)
```

### 3. Force Specific Strategy
```python
# Force structural analysis
@xcs.jit(mode=JITMode.STRUCTURAL)
class MyOperator(Operator):
    ...

# Force trace-based
@xcs.jit(mode=JITMode.STRUCTURAL)
def my_function(inputs):
    ...
```

### 4. Hybrid Approach
For maximum flexibility, you can separate concerns:

```python
class EnsembleComponents:
    """Holds operators as attributes for structural discovery."""
    def __init__(self):
        self.model1 = ModelA()
        self.model2 = ModelB()
        self.judge = Judge()

@xcs.jit
def flexible_ensemble(inputs, use_all=True):
    """Function with dynamic logic."""
    components = EnsembleComponents()
    
    # Trace JIT handles the dynamic logic
    results = []
    results.append(components.model1(inputs))
    
    if use_all:
        results.append(components.model2(inputs))
    
    return components.judge(results)
```

## The Magic: Same Parallelism Discovery

**Regardless of which JIT strategy is used**, the resulting graph goes through the same wave analysis:

1. **Structural JIT** → Builds graph from static analysis → Wave analysis → Parallel execution
2. **Trace JIT** → Builds graph from execution trace → Wave analysis → Parallel execution

Both paths lead to automatic parallelism discovery!

## Summary

- **Structural JIT**: Best for Operator classes with nested operators as attributes
- **Trace JIT**: Best for simple functions or dynamic operator creation
- **Auto mode**: Usually picks the right one based on scoring
- **Parallelism**: Works the same regardless of strategy!

The beauty is that users don't need to think about this - the system automatically chooses the best approach and discovers all parallelization opportunities.