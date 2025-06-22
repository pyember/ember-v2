"""Fixed base module that properly handles class attribute defaults.

Following CLAUDE.md principles - fix it at the root, make the common case work.
"""

from typing import Any, Dict, Optional, Tuple, Type, TypeVar, get_type_hints
import dataclasses
from functools import cached_property
import inspect

T = TypeVar('T', bound='Module')


def _is_type_annotation(annotation: Any) -> bool:
    """Check if an annotation is for a Type[T] or Optional[Type[T]]."""
    # This is a simplified check - could be more robust
    if hasattr(annotation, '__origin__'):
        origin = annotation.__origin__
        if origin is type or origin is Type:
            return True
        # Check for Optional[Type[...]]
        if origin is Optional or str(origin) == 'typing.Union':
            args = getattr(annotation, '__args__', ())
            if args and hasattr(args[0], '__origin__') and args[0].__origin__ is type:
                return True
    return False


class Module:
    """Base class for all Ember modules - with proper class attribute handling.
    
    This version fixes the issue where Type class attributes become None in
    dataclasses. It preserves the user's intent while maintaining immutability.
    """
    
    def __init_subclass__(cls, **kwargs):
        """Convert subclasses to frozen dataclasses with proper defaults."""
        super().__init_subclass__(**kwargs)
        
        # Only process if not already a dataclass
        if not hasattr(cls, '__dataclass_fields__'):
            # Collect class attributes that should become defaults
            class_defaults = {}
            
            # Get all class attributes (not methods)
            for name, value in cls.__dict__.items():
                if not name.startswith('_') and not callable(value):
                    # Check if this has a type annotation
                    annotations = getattr(cls, '__annotations__', {})
                    if name in annotations:
                        # Special handling for Type annotations
                        annotation = annotations[name]
                        if _is_type_annotation(annotation) and value is not None:
                            # Preserve the class reference
                            class_defaults[name] = value
            
            # Get annotations from class hierarchy
            annotations = {}
            for base in reversed(cls.__mro__[:-1]):  # Exclude object
                if hasattr(base, '__annotations__'):
                    annotations.update(base.__annotations__)
            
            # Update class annotations
            cls.__annotations__ = annotations
            
            # Create field specs with proper defaults
            for name, default_value in class_defaults.items():
                if name in annotations:
                    # Use default_factory to preserve class references
                    setattr(cls, name, dataclasses.field(
                        default_factory=lambda v=default_value: v
                    ))
            
            # Apply dataclass transformation
            cls = dataclasses.dataclass(
                cls,
                frozen=True,  # Immutable by default
                eq=True,      # Value equality
                repr=True     # Clean repr
            )
            
            # Update the class in the module namespace
            if cls.__module__ in globals():
                globals()[cls.__name__] = cls
            
            # Ensure the modified class is returned
            return cls
    
    def replace(self: T, **kwargs) -> T:
        """Create a new instance with updated fields."""
        return dataclasses.replace(self, **kwargs)
    
    @cached_property
    def _learnable_fields(self) -> Tuple[str, ...]:
        """Fields containing JAX arrays (detected at runtime)."""
        try:
            import jax.numpy as jnp
            learnable = []
            for field_name in self.__dataclass_fields__:
                value = getattr(self, field_name)
                if isinstance(value, jnp.ndarray):
                    learnable.append(field_name)
            return tuple(learnable)
        except ImportError:
            return ()
    
    @cached_property
    def _static_fields(self) -> Tuple[str, ...]:
        """Fields that are not learnable (models, tools, config)."""
        all_fields = set(self.__dataclass_fields__.keys())
        learnable = set(self._learnable_fields)
        return tuple(all_fields - learnable)
    
    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """JAX pytree protocol - only learnable fields are dynamic."""
        dynamic_values = tuple(getattr(self, name) for name in self._learnable_fields)
        static_data = {
            name: getattr(self, name) for name in self._static_fields
        }
        static_data['_field_names'] = self._learnable_fields
        return dynamic_values, static_data
    
    @classmethod
    def tree_unflatten(cls, static_data: Dict[str, Any], dynamic_values: Tuple[Any, ...]) -> 'Module':
        """JAX pytree protocol - reconstruct from static and dynamic parts."""
        field_names = static_data.pop('_field_names')
        all_data = dict(static_data)
        all_data.update(zip(field_names, dynamic_values))
        return cls(**all_data)


# Optional: Register as JAX pytree if available
try:
    import jax
    jax.tree_util.register_pytree_node(
        Module,
        Module.tree_flatten,
        Module.tree_unflatten
    )
except ImportError:
    pass  # JAX not required