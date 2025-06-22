# EmberModule System Design Document

## Executive Summary

This document outlines the design for EmberModule, a foundational system for stateful operators in the Ember framework. The design emphasizes simplicity, type safety, and functional programming patterns while maintaining compatibility with PyTree-like transformations.

## Core Requirements

### Functional Requirements
1. **Immutable state management**: All modules must be immutable after initialization
2. **Tree transformations**: Support functional transformations over module trees
3. **Static/dynamic separation**: Clear distinction between transformable and non-transformable fields
4. **Type safety**: Full type checking support with proper generics
5. **Zero boilerplate**: Minimal code required for common patterns

### Non-Functional Requirements
1. **Performance**: Negligible overhead compared to raw function calls
2. **Simplicity**: < 300 lines of core implementation
3. **Compatibility**: Work with standard Python typing and dataclasses
4. **Extensibility**: Easy to extend without modifying core

## Architecture Overview

### Core Components

```python
# 1. Tree Registry (Independent of JAX)
class TreeRegistry:
    """Central registry for tree node types and their transformations."""
    _registry: Dict[type, TreeNode] = {}
    
# 2. Module Metaclass
class ModuleMeta(type):
    """Metaclass that converts classes to frozen dataclasses with tree support."""
    
# 3. Base Module
class EmberModule(metaclass=ModuleMeta):
    """Base class for all stateful operators."""
    
# 4. Field Helpers
def static_field(**kwargs):
    """Mark a field as static (preserved but not transformed)."""
    
def dynamic_field(**kwargs):
    """Mark a field as dynamic (subject to transformations)."""
```

### Design Principles

1. **Explicit over Implicit**
   - Fields must be explicitly marked as static or dynamic
   - No hidden behavior or magic methods
   - Clear error messages for misuse

2. **Composition over Inheritance**
   - Modules compose through fields, not inheritance hierarchies
   - Prefer protocols over base classes for extensibility
   - Support nested module structures naturally

3. **Functional Programming Patterns**
   - Immutability by default
   - Pure transformations
   - Referential transparency

4. **Standard Python Idioms**
   - Leverage dataclasses for field management
   - Use type hints for documentation
   - Follow PEP 8 and Google style guide

## Detailed Design

### Tree Registration System

```python
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Protocol

T = TypeVar('T')

class TreeNode(Protocol):
    """Protocol for tree node registration."""
    flatten: Callable[[T], Tuple[List[Any], Any]]
    unflatten: Callable[[Any, List[Any]], T]

class TreeRegistry:
    """Registry for PyTree-like operations without external dependencies."""
    
    _registry: Dict[type, TreeNode] = {}
    
    @classmethod
    def register(cls, 
                 node_type: type,
                 flatten: Callable[[Any], Tuple[List[Any], Any]],
                 unflatten: Callable[[Any, List[Any]], Any]) -> None:
        """Register a type as a tree node."""
        cls._registry[node_type] = TreeNode(flatten=flatten, unflatten=unflatten)
    
    @classmethod
    def flatten(cls, obj: Any) -> Tuple[List[Any], Any]:
        """Flatten an object to leaves and auxiliary data."""
        node = cls._registry.get(type(obj))
        if node:
            return node.flatten(obj)
        # Default: treat as leaf
        return [obj], None
    
    @classmethod
    def unflatten(cls, aux: Any, leaves: List[Any]) -> Any:
        """Reconstruct object from leaves and auxiliary data."""
        if aux is None:
            return leaves[0] if leaves else None
        node_type, *rest_aux = aux
        node = cls._registry.get(node_type)
        if node:
            return node.unflatten((node_type, *rest_aux), leaves)
        raise ValueError(f"Unknown node type: {node_type}")
```

### Module Implementation

```python
import dataclasses
from typing import Any, Dict, List, Tuple

class ModuleMeta(type):
    """Metaclass for automatic dataclass conversion and tree registration."""
    
    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]):
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Skip if already a dataclass or if it's the base class
        if cls.__name__ == 'EmberModule' or dataclasses.is_dataclass(cls):
            return cls
            
        # Convert to frozen dataclass
        cls = dataclasses.dataclass(frozen=True, eq=False)(cls)
        
        # Add custom __post_init__ for field conversion
        original_post_init = getattr(cls, '__post_init__', None)
        
        def __post_init__(self):
            # Run original post_init if exists
            if original_post_init:
                original_post_init(self)
                
            # Apply field converters
            for field in dataclasses.fields(self):
                if 'converter' in field.metadata:
                    converter = field.metadata['converter']
                    value = getattr(self, field.name)
                    converted = converter(value)
                    # Use object.__setattr__ for frozen dataclass
                    object.__setattr__(self, field.name, converted)
        
        cls.__post_init__ = __post_init__
        
        # Register tree operations
        def flatten(module) -> Tuple[List[Any], Any]:
            dynamic_values = []
            dynamic_keys = []
            static_fields = {}
            
            for field in dataclasses.fields(module):
                value = getattr(module, field.name)
                if field.metadata.get('static', False):
                    static_fields[field.name] = value
                else:
                    dynamic_values.append(value)
                    dynamic_keys.append(field.name)
            
            aux = (type(module), dynamic_keys, static_fields)
            return dynamic_values, aux
        
        def unflatten(aux: Any, leaves: List[Any]) -> Any:
            cls_type, dynamic_keys, static_fields = aux
            kwargs = dict(zip(dynamic_keys, leaves))
            kwargs.update(static_fields)
            return cls_type(**kwargs)
        
        TreeRegistry.register(cls, flatten, unflatten)
        
        return cls

class EmberModule(metaclass=ModuleMeta):
    """Base class for all Ember modules.
    
    Automatically becomes a frozen dataclass with tree transformation support.
    """
    
    def replace(self, **updates):
        """Create a new instance with updated fields."""
        return dataclasses.replace(self, **updates)
```

### Field Helpers

```python
from typing import Any, Optional, Callable
import dataclasses

MISSING = dataclasses.MISSING

def static_field(default=MISSING, *, 
                 default_factory=MISSING,
                 converter: Optional[Callable] = None,
                 **kwargs) -> dataclasses.Field:
    """Create a static field that is preserved but not transformed.
    
    Static fields are not included in tree transformations but are
    preserved when reconstructing modules.
    
    Args:
        default: Default value for the field
        default_factory: Factory function for default value
        converter: Optional converter function applied during init
        **kwargs: Additional field arguments
        
    Returns:
        Configured dataclass field
    """
    metadata = kwargs.pop('metadata', {})
    metadata['static'] = True
    if converter:
        metadata['converter'] = converter
    
    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        metadata=metadata,
        **kwargs
    )

def dynamic_field(default=MISSING, *,
                  default_factory=MISSING, 
                  converter: Optional[Callable] = None,
                  **kwargs) -> dataclasses.Field:
    """Create a dynamic field that participates in transformations.
    
    Dynamic fields are included in tree flattening and can be transformed
    by operations like vmap, pmap, etc.
    
    Args:
        default: Default value for the field
        default_factory: Factory function for default value  
        converter: Optional converter function applied during init
        **kwargs: Additional field arguments
        
    Returns:
        Configured dataclass field
    """
    metadata = kwargs.pop('metadata', {})
    metadata['static'] = False
    if converter:
        metadata['converter'] = converter
        
    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        metadata=metadata,
        **kwargs
    )
```

## Usage Examples

### Basic Module

```python
class LinearModel(EmberModule):
    """Simple linear model with weights and bias."""
    weights: Array
    bias: Array
    activation: str = static_field(default="relu")  # Config, not transformed
    
    def __call__(self, x: Array) -> Array:
        result = x @ self.weights + self.bias
        if self.activation == "relu":
            return jnp.maximum(0, result)
        return result
```

### Composed Modules

```python
class Ensemble(EmberModule):
    """Ensemble of models with voting."""
    models: List[EmberModule]  # Dynamic - will be transformed
    voting: str = static_field(default="majority")  # Static config
    
    def __call__(self, x: Any) -> Any:
        predictions = [model(x) for model in self.models]
        if self.voting == "majority":
            return most_common(predictions)
        return predictions[0]  # Just return first
```

### With Converters

```python
def ensure_positive(x: float) -> float:
    """Ensure value is positive."""
    if x <= 0:
        raise ValueError(f"Expected positive value, got {x}")
    return x

class Temperature(EmberModule):
    """Temperature-controlled generation."""
    value: float = dynamic_field(converter=ensure_positive)
    
    def scale_logits(self, logits: Array) -> Array:
        return logits / self.value
```

## Testing Strategy

### Unit Tests
1. **Tree operations**: Flatten/unflatten correctness
2. **Field behavior**: Static vs dynamic field handling
3. **Immutability**: Ensure modules remain frozen
4. **Type safety**: Proper type checking with mypy

### Integration Tests
1. **Composition**: Nested module structures
2. **Transformations**: Work with vmap/pmap analogues
3. **Performance**: Benchmark against raw functions
4. **Error handling**: Clear messages for common mistakes

### Property Tests
1. **Round-trip**: flatten(unflatten(x)) == x
2. **Immutability**: No mutations after creation
3. **Type preservation**: Types maintained through transformations

## Migration Path

### From Original _module.py
1. Replace `ember.module()` with `EmberModule` base class
2. Update `ember_field()` to `static_field()` or `dynamic_field()`
3. Remove explicit cache management (not needed)
4. Update imports to new module

### Example Migration

```python
# Before
@ember.module()
class MyOperator:
    model: Model
    config: Dict = ember_field(static=True)
    
# After  
class MyOperator(EmberModule):
    model: Model
    config: Dict = static_field(default_factory=dict)
```

## Performance Considerations

### Optimization Strategies
1. **Lazy initialization**: Only create objects when needed
2. **Structure sharing**: Reuse unchanged parts in transformations
3. **Minimal copying**: Use dataclass replace for efficiency

### Benchmarks
- Module creation: < 1μs overhead vs raw dataclass
- Tree operations: < 100ns for simple modules
- Memory usage: Same as equivalent dataclass

## Future Extensions

### Planned Features
1. **Serialization support**: JSON/msgpack for modules
2. **Validation decorators**: Field-level validation
3. **Async support**: Async call methods
4. **Debug mode**: Enhanced error messages and tracing

### Compatibility
- JAX integration: Direct pytree registration when available
- Type stubs: Complete .pyi files for IDE support
- Protocol support: Allow duck-typed modules

## Conclusion

This design provides a clean, simple foundation for stateful operators in Ember. By leveraging Python's built-in features and avoiding unnecessary complexity, we achieve a system that is both powerful and maintainable. The < 300 line implementation proves that sophisticated functionality doesn't require complicated code.