# Equinox Integration Design for Ember

## Overview

We're adopting Equinox's battle-tested module system as the foundation for Ember's operator system, extending it with LLM orchestration features. This document outlines the integration strategy and necessary extensions.

## Why Equinox?

Equinox provides:
- **Robust PyTree integration** with proper static/dynamic field handling
- **Comprehensive error handling** with educational messages
- **Battle-tested design** used in production ML systems
- **Advanced features** like abstract variables, strict mode, and validation hooks
- **Proper handling of edge cases** (cycles, JAX transforms, initialization)

## Integration Strategy

### 1. Core Module System (equinox_module.py)

**Minimal Changes to Equinox Core:**
- Fix imports to work without equinox package dependencies
- Add Ember-specific field types and converters
- Integrate with EmberModel for validation

**What to Keep Unchanged:**
- The metaclass system (`_ActualModuleMeta`)
- PyTree registration and flattening logic
- BoundMethod implementation
- Initialization handling (Initable pattern)
- All safety checks and warnings

### 2. Import Fixes

Replace Equinox-specific imports:
```python
# Replace these imports:
from ._better_abstract import ABCMeta, dataclass
from ._caches import cache_clears
from ._doc_utils import doc_repr
from ._filters import is_array, is_array_like
from ._pretty_print import tree_pformat
from ._tree import tree_equal

# With Ember equivalents or inline implementations:
from abc import ABCMeta
import dataclasses
# Implement minimal versions of utilities we need
```

### 3. LLM-Specific Extensions

Add new field types for LLM orchestration:

```python
def model_field(model_binding_cls=None, *, static=True, **kwargs):
    """Field for model bindings (static by default)."""
    def converter(value):
        if isinstance(value, str):
            return ModelBinding(value)
        return value
    return field(static=static, converter=converter, **kwargs)

def prompt_field(*, static=True, template_vars=None, **kwargs):
    """Field for prompt templates with validation."""
    def converter(value):
        if template_vars:
            # Validate template variables exist
            for var in template_vars:
                if f"{{{var}}}" not in value:
                    raise ValueError(f"Missing template variable: {var}")
        return value
    return field(static=static, converter=converter, **kwargs)

def spec_field(spec_class, *, static=True, **kwargs):
    """Field for input/output specifications."""
    def converter(value):
        if isinstance(value, dict):
            return spec_class(**value)
        elif isinstance(value, type) and issubclass(value, spec_class):
            return value()
        return value
    return field(static=static, converter=converter, **kwargs)
```

### 4. EmberModule Extension

Create EmberModule as a Module subclass with LLM features:

```python
class EmberModule(Module):
    """Module with LLM orchestration features."""
    
    def __check_init__(self):
        """Validate operator invariants."""
        # Check for input/output specs if forward() exists
        if hasattr(self, 'forward') and not hasattr(self, '__abstractmethods__'):
            if not hasattr(self, 'input_spec'):
                warnings.warn(f"{self.__class__.__name__} has forward() but no input_spec")
    
    def validate_input(self, input):
        """Validate against input_spec if available."""
        if hasattr(self, 'input_spec'):
            return self.input_spec.validate(input)
        return input
    
    def validate_output(self, output):
        """Validate against output_spec if available."""
        if hasattr(self, 'output_spec'):
            return self.output_spec.validate(output)
        return output
    
    def __call__(self, input):
        """Default call implementation with validation."""
        if hasattr(self, 'forward'):
            validated_input = self.validate_input(input)
            output = self.forward(validated_input)
            return self.validate_output(output)
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward() or __call__()")
```

### 5. Operator Base Classes

Update operator base classes to use EmberModule:

```python
class Operator(EmberModule):
    """Base operator with specifications."""
    input_spec: InputSpec = spec_field(InputSpec)
    output_spec: OutputSpec = spec_field(OutputSpec)
    
    @abc.abstractmethod
    def forward(self, input):
        """Process input to output."""
        pass

class ModelOperator(Operator):
    """Operator that wraps a model."""
    model: ModelBinding = model_field()
    prompt_template: str = prompt_field(static=True)
    
    def get_prompt(self, **kwargs):
        """Format prompt with kwargs."""
        return self.prompt_template.format(**kwargs)

class LearnableOperator(Operator):
    """Operator with learnable parameters."""
    # Dynamic fields for gradients
    weights: Optional[Array] = None
    bias: Optional[float] = 0.0
    
    def parameters(self):
        """Get learnable parameters."""
        return {'weights': self.weights, 'bias': self.bias}
```

### 6. Compatibility Layer

Ensure backward compatibility:

```python
# In module.py (user-facing)
from ember.core.operators.equinox_module import (
    Module as EmberModule,
    field,
    static_field,
    model_field,
    prompt_field,
    spec_field,
    # ... other exports
)

# Compatibility aliases
Module = EmberModule

# Simple decorator for basic use cases
def op(fn):
    """Simple operator from function."""
    class FunctionOperator(EmberModule):
        func: Callable = field(static=True, default=fn)
        
        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)
    
    return FunctionOperator()
```

## Implementation Plan

### Phase 1: Core Integration
1. Copy equinox_module.py and fix imports
2. Implement missing utility functions inline
3. Add LLM-specific field types
4. Test basic Module functionality

### Phase 2: Operator Integration  
1. Update operator base classes to inherit from Module
2. Add EmberModule with validation features
3. Update concrete operators to use new field types
4. Ensure all tests pass

### Phase 3: Advanced Features
1. Add AbstractVar support for operator interfaces
2. Enable strict mode for library code
3. Add comprehensive validation with __check_init__
4. Document advanced patterns

## Key Design Decisions

### 1. Minimal Modifications to Equinox Core
We keep Equinox's core logic intact to benefit from its robustness. Changes are additive.

### 2. Static by Default for LLM Components
Model bindings, prompts, and configs are static by default since they don't participate in gradients.

### 3. Progressive Disclosure
- Simple: `@op` decorator for functions
- Medium: Inherit from ModelOperator
- Advanced: Full EmberModule with all features

### 4. Validation Integration
Leverage Equinox's __check_init__ hook for operator-specific validation.

## Benefits of This Approach

1. **Robustness**: Inherit years of battle-testing from Equinox
2. **Compatibility**: JAX transforms work correctly
3. **Safety**: Comprehensive error messages and validation
4. **Flexibility**: Support both simple and complex use cases
5. **Future-proof**: Can leverage Equinox updates

## Migration Path

For existing code:
1. `Module` becomes an alias for `EmberModule`
2. Existing operators continue to work
3. New features are opt-in via new field types
4. Gradual adoption of strict mode and validation

## Example: Complete Operator

```python
class SmartRouter(EmberModule, strict=True):
    """Router with learnable weights and static models."""
    
    # Static configuration
    models: Dict[str, ModelBinding] = model_field(static=True)
    prompt_template: str = prompt_field(
        static=True,
        template_vars=['query', 'context'],
        default="Route this query: {query}\nContext: {context}"
    )
    
    # Dynamic learnable parameters
    routing_weights: Array
    temperature: float = 1.0
    
    # Specifications
    input_spec: InputSpec = spec_field(RouterInputSpec)
    output_spec: OutputSpec = spec_field(RouterOutputSpec)
    
    def __init__(self, models: Dict[str, Any], embedding_dim: int = 384, key: Array):
        # Convert model strings/configs to ModelBinding
        self.models = {k: ModelBinding(v) for k, v in models.items()}
        self.routing_weights = jax.random.normal(key, (embedding_dim, len(models)))
        self.prompt_template = self.prompt_template  # Use default
        self.input_spec = RouterInputSpec()
        self.output_spec = RouterOutputSpec()
    
    def forward(self, input: RouterInput) -> RouterOutput:
        # Automatic validation via EmberModule
        embeddings = self.get_embeddings(input.query)
        scores = embeddings @ self.routing_weights / self.temperature
        
        # Route to best model
        best_idx = jnp.argmax(scores)
        model_name = list(self.models.keys())[best_idx]
        
        # Call model (static, not differentiable)
        result = self.models[model_name].generate(
            self.get_prompt(query=input.query, context=input.context)
        )
        
        return RouterOutput(
            result=result,
            selected_model=model_name,
            scores=scores
        )
```

## Next Steps

1. Review and approve this design
2. Implement Phase 1 (core integration)
3. Test thoroughly with existing operators
4. Implement Phase 2 (operator integration)
5. Document migration guide
6. Roll out to users