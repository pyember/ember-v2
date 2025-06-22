# Final Implementation Plan for Ember

## Overview

This document consolidates all our design decisions into a clear implementation plan. We've simplified dramatically based on first principles and our dream team's insights.

## Core Design Decisions

### 1. Module System: Use Equinox Directly

**Decision**: Import and use Equinox as our module system. No reinvention.

```python
# ember/core/module.py
from equinox import Module, field

# Add one special field type for learnable parameters
def param(init=None, shape=None, **kwargs):
    """Mark field as learnable (dynamic) for JAX transforms."""
    return field(static=False, **kwargs)

# That's it. Everything else is Equinox.
```

**Key insights**:
- Static by default (aligns with LLM orchestration reality)
- Only mark dynamic fields with `ember.param()`
- Arbitrary nesting with mixed static/dynamic works perfectly
- Full JAX compatibility out of the box

### 2. Data Models: Simple Facade

**Decision**: Replace the 288-line EmberModel with a simple alias.

```python
# ember/core/types/model.py
"""Ember data model base class."""

from pydantic import BaseModel

# Simple facade - can change implementation later
EmberModel = BaseModel

__all__ = ["EmberModel"]
```

**Why this works**:
- Zero overhead (it's just Pydantic)
- Full validation features available today
- Can change implementation with one line
- Clear story: "EmberModel is how you define data"

### 3. Operators: Modules with Optional Validation

**Decision**: Operators are just Modules with conventions.

```python
# ember/core/operators/base.py
from ember.core.module import Module
from typing import Optional, Type

class Operator(Module):
    """Base operator - just a Module with forward() method."""
    
    # Optional type specifications (static fields)
    input_spec: Optional[Type] = None
    output_spec: Optional[Type] = None
    
    # For future optimization (static field)
    examples: Optional[List[Tuple]] = None
    
    def forward(self, input):
        """Override with operator logic."""
        raise NotImplementedError
    
    def __call__(self, input):
        # Base class: no validation overhead
        return self.forward(input)

# Add validation only when needed
class ValidatedOperator(Operator):
    """Operator with automatic validation."""
    
    def __call__(self, input):
        if self.input_spec:
            input = self._validate_input(input)
        output = self.forward(input)
        if self.output_spec:
            output = self._validate_output(output)
        return output
```

### 4. Progressive Disclosure API

```python
# Level 1: Simple functions
@ember.op
def classify(text: str) -> str:
    return ModelBinding("gpt-4")(f"Classify: {text}")

# Level 2: Basic operators  
class Classifier(ember.Operator):
    model: ModelBinding = ModelBinding("gpt-4")
    
    def forward(self, text: str) -> str:
        return self.model(f"Classify: {text}")

# Level 3: Validated operators
class ProductionClassifier(ember.ValidatedOperator):
    input_spec = TextInput  # EmberModel subclass
    output_spec = Label     # EmberModel subclass
    
    model: ModelBinding = ModelBinding("gpt-4")
    temperature: float = ember.param(0.7)  # Learnable!
    
    def forward(self, input: TextInput) -> Label:
        # Full validation and structure
        result = self.model(f"Classify: {input.text}")
        return Label(value=result)
```

## Implementation Steps

### Phase 1: Core Infrastructure (Week 1)

1. **Set up Equinox dependency**
   ```toml
   # pyproject.toml
   dependencies = [
       "equinox>=0.11.0",
       "jax>=0.4.0", 
       "pydantic>=2.0",
       "optax>=0.1.0"  # For optimization examples
   ]
   ```

2. **Create minimal module system**
   ```python
   # ember/core/module.py
   from equinox import *  # Re-export everything
   from equinox import Module as EmberModule  # Alias for clarity
   
   def param(...):  # Our one addition
       return field(static=False, ...)
   ```

3. **Simplify EmberModel**
   ```python
   # ember/core/types/model.py
   from pydantic import BaseModel
   EmberModel = BaseModel  # Delete 288-line wrapper
   ```

### Phase 2: Operator System (Week 1)

1. **Base operator classes**
   ```python
   # ember/core/operators/base.py
   - Operator (no validation)
   - ValidatedOperator (opt-in validation)
   - Simple validation utilities
   ```

2. **Model integration**
   ```python
   # ember/core/operators/model_binding.py
   class ModelBinding(Module):
       """Clean abstraction for LLM calls."""
       model_name: str
       
       def __call__(self, prompt: str) -> str:
           # Delegate to appropriate provider
           ...
   ```

3. **Common operators**
   ```python
   # ember/core/operators/common.py
   - Router (conditional execution)
   - Ensemble (parallel execution)
   - Chain (sequential execution)
   ```

### Phase 3: Progressive API (Week 2)

1. **Function decorator**
   ```python
   # ember/api/decorators.py
   def op(fn):
       """Turn function into operator."""
       class FunctionOperator(Operator):
           def forward(self, *args, **kwargs):
               return fn(*args, **kwargs)
       return FunctionOperator()
   ```

2. **High-level patterns**
   ```python
   # ember/api/patterns.py
   def ensemble(*ops): ...
   def chain(*ops): ...
   def router(routes: Dict[str, Op]): ...
   ```

### Phase 4: Testing & Documentation (Week 2)

1. **Core tests**
   - Module system (static/dynamic fields)
   - JAX transform compatibility
   - Operator validation
   - Nested operator structures

2. **Integration tests**
   - End-to-end pipelines
   - Performance benchmarks
   - Memory usage

3. **Documentation**
   - Getting started guide
   - API reference
   - Migration guide from old system

## What We're NOT Building

1. **Complex validation framework** - Just use types and Pydantic
2. **Prompt rendering in specs** - That's the operator's job
3. **Heavy abstractions** - Keep it simple
4. **Magic** - Everything explicit

## Success Metrics

1. **Simplicity**: Core implementation < 500 lines
2. **Performance**: Zero overhead for non-validated operators
3. **Compatibility**: All JAX transforms work
4. **Progressive**: Simple things simple, complex things possible

## Migration Strategy

### From Old Ember

```python
# Old
class MyOperator(Operator):
    spec = Specification(...)
    
# New  
class MyOperator(Operator):
    input_spec = InputType
    output_spec = OutputType
```

### Key Changes

1. **Modules**: Now using Equinox directly
2. **EmberModel**: Now just an alias for Pydantic BaseModel
3. **Operators**: Simpler, with optional validation
4. **Specifications**: Removed in favor of simple types

## Example: Complete System

```python
# Define data models
class Question(EmberModel):
    text: str
    context: Optional[str] = None

class Answer(EmberModel):
    text: str
    confidence: float
    sources: List[str] = []

# Define operators
class QAOperator(Operator):  # Single class, validation via specs
    input_spec = Question
    output_spec = Answer
    
    # Static components
    retriever: Retriever
    model: ModelBinding
    
    # Dynamic components (auto-detected)
    confidence_weights: jax.Array
    relevance_threshold: jax.Array
    
    def __init__(self, key: jax.Array):
        self.retriever = Retriever("prod-index")
        self.model = ModelBinding("gpt-4")
        # JAX arrays are automatically dynamic
        self.confidence_weights = jax.random.normal(key, (10,))
        self.relevance_threshold = jnp.array(0.8)
    
    def forward(self, q: Question) -> Answer:
        docs = self.retriever.search(q.text)
        # Use learnable parameters
        confidence = jax.nn.sigmoid(self.confidence_weights.sum() + self.relevance_threshold)
        response = self.model(f"Context: {docs}\nQ: {q.text}")
        return Answer(
            text=response,
            confidence=float(confidence),
            sources=[d.id for d in docs]
        )

# Use it
qa = QAOperator(jax.random.PRNGKey(0))
answer = qa(Question(text="What is JAX?"))

# JAX transforms work!
grads = jax.grad(loss_fn)(qa)
# grads.confidence_weights and grads.relevance_threshold have gradients
# grads.retriever and grads.model are None (static)
```

## Summary

We've dramatically simplified Ember by:
1. Using Equinox directly instead of reimplementing modules
2. Making EmberModel a simple alias instead of 288-line wrapper
3. Single Operator class with optional validation via specs
4. Removing complex specification system in favor of types
5. No special field markers - JAX arrays are automatically dynamic
6. Unified API - no separate ValidatedOperator class

The result is more powerful, more correct, and much simpler.