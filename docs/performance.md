# Ember Performance Guide

This guide shows how to optimize Ember applications for maximum performance.

## Key Principles

1. **Automatic Optimization** - Ember handles most optimization automatically
2. **Measure First** - Use profiling to identify actual bottlenecks
3. **Simple Solutions** - Often the simplest code is also the fastest

## Automatic Optimizations

Ember automatically provides:

### Parallel Execution

```python
from ember.api import ember

# Ember detects independent operations and runs them in parallel
results = await ember.parallel([
    analyze(doc) for doc in documents
])
```

### JIT Compilation

```python
from ember.api.xcs import jit

@ember.op
@jit  # Automatic optimization for repeated calls
async def process_item(item: str) -> dict:
    return await ember.llm(f"Process: {item}")
```

### Smart Caching

```python
# Results are cached automatically when beneficial
@ember.op
@jit
async def expensive_analysis(text: str) -> dict:
    # This will be cached based on input
    return await complex_processing(text)
```

## Performance Best Practices

### 1. Batch Operations

**Bad**: Sequential processing
```python
results = []
for item in items:
    result = await process(item)
    results.append(result)
```

**Good**: Parallel processing
```python
results = await ember.parallel([
    process(item) for item in items
])
```

### 2. Use Streaming for Large Datasets

**Bad**: Load everything into memory
```python
all_data = load_huge_dataset()
results = await process_all(all_data)
```

**Good**: Stream processing
```python
async for batch in ember.stream(huge_dataset, process_batch):
    # Process incrementally
    save_results(batch)
```

### 3. Choose the Right Model

**Bad**: Using expensive models for simple tasks
```python
# Overkill for simple classification
response = await ember.llm("Is this positive?", model="gpt-4")
```

**Good**: Match model to task complexity
```python
# Use faster models for simple tasks
response = await ember.llm("Is this positive?", model="gpt-3.5-turbo")
```

### 4. Optimize Prompts

**Bad**: Verbose prompts with redundant information
```python
prompt = """You are an AI assistant. Your task is to analyze text.
Please analyze the following text carefully and thoroughly.
Here is the text to analyze: ..."""
```

**Good**: Concise, focused prompts
```python
prompt = "Analyze sentiment: {text}"
```

## Measuring Performance

### Using JIT Stats

```python
from ember.api.xcs import jit, get_jit_stats

@jit
def process(data):
    return transform(data)

# Run your workload
for item in dataset:
    process(item)

# Check performance
stats = get_jit_stats()
print(f"Cache hit rate: {stats['cache_hits'] / stats['total_calls']:.2%}")
print(f"Average time: {stats['avg_time_ms']}ms")
```

### Profiling with Context Managers

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f"{name}: {time.time() - start:.2f}s")

# Profile sections
with timer("Data loading"):
    data = load_data()

with timer("Processing"):
    results = await process_all(data)
```

## Common Performance Patterns

### Ensemble Optimization

When using multiple models, run them in parallel:

```python
from ember.api import ember

@ember.op
async def ensemble_predict(text: str) -> str:
    # Run all models in parallel
    predictions = await ember.parallel([
        ember.llm(text, model="gpt-4"),
        ember.llm(text, model="claude-3"),
        ember.llm(text, model="gemini-pro")
    ])
    
    # Return majority vote
    return max(set(predictions), key=predictions.count)
```

### Caching Expensive Operations

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def preprocess_text(text: str) -> str:
    # Expensive preprocessing
    return cleaned_text

@ember.op
async def analyze_with_cache(text: str) -> dict:
    # Preprocessing is cached
    cleaned = preprocess_text(text)
    return await ember.llm(f"Analyze: {cleaned}")
```

### Pipeline Optimization

Structure pipelines to maximize parallelism:

```python
@ember.op
@jit
async def optimized_pipeline(documents: List[str]) -> List[dict]:
    # Stage 1: Parallel preprocessing
    preprocessed = await ember.parallel([
        preprocess(doc) for doc in documents
    ])
    
    # Stage 2: Batch processing
    batches = list(ember.batch(preprocessed, size=10))
    results = []
    
    for batch in batches:
        # Process batch in parallel
        batch_results = await ember.parallel([
            analyze(doc) for doc in batch
        ])
        results.extend(batch_results)
    
    return results
```

## Performance Checklist

Before optimizing:
- [ ] Profile to identify actual bottlenecks
- [ ] Check if you're using the simplest solution
- [ ] Verify you're not optimizing prematurely

Quick wins:
- [ ] Use `ember.parallel()` for independent operations
- [ ] Add `@jit` to frequently called functions
- [ ] Batch API calls when possible
- [ ] Choose appropriate model sizes
- [ ] Stream large datasets

Advanced optimization:
- [ ] Custom caching strategies
- [ ] Model quantization for edge deployment
- [ ] Distributed processing for massive scale

## Anti-Patterns to Avoid

### 1. Over-Engineering
```python
# Bad: Complex abstraction for simple task
class OptimizedProcessorFactoryBuilderStrategy:
    # 100 lines of "optimization" code
```

```python
# Good: Simple and fast
@ember.op
@jit
async def process(text):
    return await ember.llm(text)
```

### 2. Premature Optimization
Don't optimize without measuring. Ember's automatic optimizations handle most cases.

### 3. Fighting the Framework
Work with Ember's patterns, not against them. The framework is designed for optimal performance when used idiomatically.

## Summary

1. **Let Ember optimize automatically** - Use `@jit` and `ember.parallel()`
2. **Measure before optimizing** - Use profiling to find real bottlenecks
3. **Keep it simple** - Simple code is often fastest
4. **Use the right tool** - Match model size to task complexity
5. **Think in batches** - Process multiple items together when possible

Remember: The best optimization is often the code you don't write. Ember handles the complex performance optimizations so you can focus on your application logic.