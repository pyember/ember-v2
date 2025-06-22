# Ember API Reference

This directory contains comprehensive API documentation for all Ember modules.

## Core APIs

### [ember](./ember.md)
The main Ember API for common operations.

### [operators](./operators.md)
Building and composing operators for AI workflows.

### [models](./models.md)
Working with different LLM providers and configurations.

### [data](./data.md)
Data loading, processing, and streaming utilities.

### [xcs](./xcs.md)
Advanced execution strategies and optimizations.

## Import Guide

### Simple Imports (90% Use Case)

```python
from ember.api import ember

# Everything you need for basic usage
response = await ember.llm("Hello, Ember!")
```

### Standard Imports

```python
from ember.api import ember, models, data, operators
from ember.api import validate, retry, jit
```

### Advanced Imports

```python
# For complex operator patterns
from ember.api.operators import Operator, Specification

# For custom model configurations  
from ember.api.models import ModelConfig, Provider

# For data pipelines
from ember.api.data import Dataset, Transform

# For execution control
from ember.api.xcs import Graph, execute
```

## Quick Examples

### Basic LLM Call
```python
from ember.api import ember
response = await ember.llm("What is machine learning?")
```

### Structured Output
```python
from pydantic import BaseModel

class Answer(BaseModel):
    explanation: str
    confidence: float

result = await ember.llm(
    "Explain quantum computing",
    output_type=Answer
)
```

### Simple Operator
```python
@ember.op
async def summarize(text: str) -> str:
    return await ember.llm(f"Summarize: {text}")
```

### Parallel Processing
```python
results = await ember.parallel([
    summarize(doc) for doc in documents
])
```

## Module Documentation

- [ember](./ember.md) - Core functionality and utilities
- [operators](./operators.md) - Operator creation and composition
- [models](./models.md) - Model management and configuration
- [data](./data.md) - Data processing utilities
- [xcs](./xcs.md) - Execution and optimization
- [types](./types.md) - Type definitions and schemas