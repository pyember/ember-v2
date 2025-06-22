"""Core module system for Ember.

This module provides the foundation for all Ember components through the
Module class. Modules in Ember are immutable, composable, and work seamlessly
with JAX transformations.

The module system follows these principles:
- Static by default: Model bindings, tools, and regular Python objects
- Dynamic automatically: JAX arrays are detected as learnable parameters
- Zero configuration: No decorators or special syntax needed
- Full JAX compatibility: All transformations (grad, jit, vmap) work naturally

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

import equinox as eqx
from typing import get_origin, get_args, List, Dict, Tuple, Set, Union


class EmberModuleMeta(type(eqx.Module)):
    """Metaclass that implements static-by-default behavior."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Process field annotations to add static markers
        annotations = namespace.get('__annotations__', {})
        
        # For each field, determine if it should be static (default) or dynamic (JAX arrays)
        for field_name, field_type in annotations.items():
            # Skip if already has a field definition
            if field_name in namespace and hasattr(namespace.get(field_name), 'metadata'):
                continue
                
            # Get default value if it exists
            default_value = namespace.get(field_name)
            
            # Check if this is a known static type
            is_static = False
            
            # Direct types
            if isinstance(field_type, type):
                if issubclass(field_type, (str, int, float, bool, dict)):
                    is_static = True
            
            # Generic types like List[str], Dict[str, Any]
            origin = get_origin(field_type)
            if origin in (list, List, dict, Dict, tuple, Tuple, set, Set):
                # Check the contents - if it's List[str], Dict[str, Any], etc., mark as static
                # But if it's List[SomeModuleWithArrays], let equinox handle it
                args = get_args(field_type)
                
                # For dict/Dict, check what it contains
                if origin in (dict, Dict):
                    # If it has type args, check the value type
                    if args and len(args) >= 2:
                        value_type = args[1]
                        # Check if value type is or contains JAX arrays
                        if hasattr(value_type, '__module__') and 'jax' in value_type.__module__:
                            is_static = False  # Dict containing JAX arrays
                        else:
                            is_static = True  # Regular config dict
                    else:
                        is_static = True  # Untyped dict, assume config
                elif args:
                    # For other containers, check if all type args are primitives
                    # Also handle Any, Union, etc.
                    from typing import Any as typing_Any
                    is_static = all(
                        arg is typing_Any or (
                            isinstance(arg, type) and 
                            issubclass(arg, (str, int, float, bool))
                        )
                        for arg in args if arg is not type(None)
                    )
                else:
                    # No type args - let equinox decide
                    is_static = False
            
            if is_static:
                if default_value is not None:
                    namespace[field_name] = eqx.field(static=True, default=default_value)
                else:
                    namespace[field_name] = eqx.field(static=True)
            # For everything else, let equinox decide at runtime
            # This allows fine-grained static/dynamic partitioning
        
        # Create the class with equinox Module
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class Module(eqx.Module, metaclass=EmberModuleMeta):
    """Ember Module - equinox Module with static-by-default behavior.
    
    All fields are static by default except JAX arrays, which are automatically
    detected as dynamic (learnable) parameters.
    
    This implementation aligns with the Ember design principles:
    - Non-JAX fields (strings, ints, model bindings, etc.) are static by default
    - JAX arrays are automatically dynamic
    - Zero configuration needed
    
    Examples:
        ```python
        class MyOperator(Module):
            # These are automatically static (no JAX type in annotation)
            name: str
            config: dict
            activation: str
            
            # These are automatically dynamic (JAX type in annotation)
            weights: jnp.ndarray
            bias: jnp.ndarray
            
            def __init__(self, dim: int, name: str = "op"):
                self.name = name  # Static
                self.config = {"dim": dim}  # Static  
                self.activation = "relu"  # Static
                self.weights = jnp.ones(dim)  # Dynamic
                self.bias = jnp.zeros(dim)  # Dynamic
        ```
    """
    pass


__all__ = ["Module"]