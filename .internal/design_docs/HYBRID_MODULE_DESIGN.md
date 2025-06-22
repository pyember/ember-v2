# Hybrid Static/Dynamic Module Design

## Overview

This document outlines the design for EmberModule v5, which adds critical support for hybrid static/dynamic fields. This enables proper JAX transformations on modules that contain both learnable parameters (dynamic) and external API calls (static).

## Problem Statement

Current EmberModule (v4) treats all fields as dynamic, which causes issues when:
1. JAX tries to differentiate through model API calls (impossible)
2. JAX transforms static configuration that shouldn't change
3. Operators need both learnable weights AND external model calls

## Solution Design

### Core Concept: Field Metadata

We'll use dataclass field metadata to mark fields as static or dynamic:

```python
class HybridOperator(Module):
    # Dynamic - JAX can transform/differentiate
    weights: jax.Array
    temperature: float = 1.0
    
    # Static - excluded from JAX transformations
    model: ModelBinding = field(static=True)
    prompt: str = field(static=True, default="Classify: {}")
```

### Key Components

#### 1. Enhanced field() function
```python
def field(*, static: bool = False, converter: Optional[Callable] = None, **kwargs) -> Field:
    """Create a field with static/dynamic control."""
```

**Changes needed:**
- Add to `/Users/jq/Downloads/eb1/src/ember/core/module_v5.py` (already started)
- Import and re-export from module.py

#### 2. Modified PyTree Registration

Current v4 PyTree registration treats all fields equally. V5 will:

```python
def _register_pytree_class(cls):
    def flatten(instance):
        dynamic_values = []
        static_dict = {}
        
        for f in dataclasses.fields(instance):
            value = getattr(instance, f.name)
            
            # KEY CHANGE: Check field metadata
            if f.metadata.get('static', False):
                static_dict[f.name] = value  # Goes in aux_data
            else:
                dynamic_values.append(value)  # Goes in children
        
        aux = (cls, tuple(dynamic_keys), static_dict)
        return dynamic_values, aux
```

**Files to modify:**
- `/Users/jq/Downloads/eb1/src/ember/core/module_v5.py`: Update `_register_pytree_class` (lines 304-356)

#### 3. Helper Functions

New utility functions for working with hybrid modules:

```python
def partition(module: Module, filter_fn: Optional[Callable] = None) -> Tuple[Module, Module]:
    """Separate learnable from non-learnable components."""

def combine(module1: Module, module2: Module) -> Module:
    """Recombine after partition."""

def is_static(module: Module, field_name: str) -> bool:
    """Check if a field is static."""
```

**Already implemented** in module_v5.py (lines 80-161)

### Integration with Operators

#### 1. Update Operator Base Classes

Operators need to properly declare static fields:

```python
class ModelOperator(Operator):
    # Model is static - not differentiable
    model: ModelBinding = static_field()
    
    # Prompts/config are static
    prompt_template: str = static_field(default="{input}")
    
    # But these could be dynamic if needed
    temperature: float = 0.7  # Could be learned!
```

**Files to modify:**
- `/Users/jq/Downloads/eb1/src/ember/core/operators/base_v4.py`: Update ModelOperator base class
- Add static_field imports and usage

#### 2. Update Concrete Operators

Concrete operators should mark appropriate fields:

```python
class JudgeOperator(ModelOperator):
    model: ModelBinding = static_field()  # Static
    criteria: List[str] = static_field()   # Static
    score_scale: int = static_field()      # Static
    
    # But could have learnable scoring weights!
    scoring_weights: Optional[jax.Array] = None  # Dynamic
```

**Files to modify:**
- `/Users/jq/Downloads/eb1/src/ember/core/operators/concrete_v2.py`: Update all operators
- Mark model, prompts, config as static
- Keep weights, thresholds as dynamic

#### 3. Update LearnableOperator Pattern

LearnableOperator becomes more powerful with explicit static/dynamic:

```python
class LearnableRouterOperator(LearnableOperator):
    # Dynamic - these get updated via gradients
    routing_weights: jax.Array
    temperature: float = 1.0
    
    # Static - these don't participate in gradients
    operators: List[Operator] = static_field()
    embedding_model: Any = static_field()
```

### Migration Path

#### 1. Create module.py as v5
```python
# src/ember/core/module.py
"""EmberModule with hybrid static/dynamic support."""

# Import everything from v5
from ember.core.module_v5 import *

__all__ = module_v5.__all__
```

#### 2. Update imports
- Change operator imports from `module_v4` to `module`
- Add `static_field` to imports where needed

#### 3. Backward Compatibility
- Fields without explicit `static=True` remain dynamic (default)
- Existing code continues to work
- Only need changes where static behavior is desired

## Implementation Checklist

### Phase 1: Core Module System
- [x] Complete module_v5.py implementation
- [ ] Create module.py importing from v5
- [ ] Add comprehensive tests for static/dynamic behavior
- [ ] Test JAX transformations (grad, vmap, jit)

### Phase 2: Operator Updates
- [ ] Update base_v4.py → base.py with static fields
- [ ] Update concrete_v2.py → concrete.py with proper markings
- [ ] Update model_integration.py if needed
- [ ] Update examples to show static field usage

### Phase 3: Testing & Documentation
- [ ] Test gradient flow with hybrid modules
- [ ] Test tree operations (partition, combine)
- [ ] Benchmark vs pure dynamic approach
- [ ] Write user documentation

## Example: Complete Hybrid System

```python
from ember.core.module import Module, field, static_field
import jax
import jax.numpy as jnp

class HybridQASystem(Module):
    # Learnable routing
    router_weights: jax.Array
    quality_threshold: float = 0.8
    
    # Static components
    models: Dict[str, Callable] = static_field()
    judge_model: Callable = static_field()
    prompts: Dict[str, str] = static_field()
    
    def __init__(self, models: Dict[str, Callable], embedding_dim: int = 384):
        self.models = models
        self.router_weights = jax.random.normal(
            jax.random.PRNGKey(0), 
            (embedding_dim, len(models))
        )
        self.judge_model = models.get("judge", models[list(models.keys())[0]])
        self.prompts = {
            "route": "Classify this query: {query}",
            "judge": "Score this answer from 0-1: {answer}"
        }
    
    def forward(self, query: str) -> str:
        # Routing uses learnable weights (differentiable)
        embedding = self.embed(query)  # Pretend we have embeddings
        scores = embedding @ self.router_weights
        model_idx = jnp.argmax(scores)
        
        # Model call is static (not differentiable)
        model_name = list(self.models.keys())[model_idx]
        answer = self.models[model_name](query)
        
        # Quality check with static judge
        quality = self.judge_model(self.prompts["judge"].format(answer=answer))
        
        if float(quality) < self.quality_threshold:
            # Fallback - could involve more learnable logic
            answer = self.models["safe_default"](query)
        
        return answer

# This system can be:
# 1. Used as-is (static inference)
# 2. Trained via gradients on router_weights
# 3. Updated via tree operations on prompts/models
# 4. Optimized via JAX transforms on dynamic parts only
```

## Benefits

1. **Correct JAX Behavior**: Gradients only flow through differentiable parts
2. **Clear Semantics**: Explicit about what can be learned vs configured
3. **Performance**: Static fields don't create unnecessary JAX overhead
4. **Flexibility**: Can mix neural and symbolic components naturally

## Risks & Mitigations

1. **Risk**: Users forget to mark fields as static
   - **Mitigation**: Good defaults, clear docs, helpful error messages

2. **Risk**: Confusion about static vs dynamic
   - **Mitigation**: Examples showing both patterns, clear naming

3. **Risk**: Breaking existing code
   - **Mitigation**: Default behavior unchanged, only new features added

## Next Steps

1. Review and approve this design
2. Complete module_v5.py → module.py migration
3. Update operator base classes with static fields
4. Create comprehensive test suite
5. Update all examples to showcase hybrid patterns

## Summary

This design enables Ember to properly handle the hybrid nature of LLM orchestration:
- Some components are learnable (weights, thresholds)
- Others are static (API calls, prompts, configuration)
- JAX transformations work correctly on each type
- Users get power when needed, simplicity by default