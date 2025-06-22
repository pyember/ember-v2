"""Advanced operator capabilities for power users.

This module provides advanced features that 90% of users won't need:
- Tree protocols for JAX-style transformations
- Dependency tracking for optimization
- Static analysis hints
- Full operator base classes

For simple use cases, use ember.operators instead.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar

__all__ = [
    'Operator',
    'TreeProtocol',
    'DependencyAware',
    'StaticAnalysisHints',
    'operator',
    'static_field',
]

T = TypeVar('T')


class TreeProtocol(Protocol):
    """Protocol for operators that support tree transformations.
    
    Implement this to enable JAX-style transformations like vmap, pmap, etc.
    """
    
    def tree_flatten(self) -> Tuple[List[Any], Any]:
        """Flatten operator into (dynamic_values, static_aux_data).
        
        Returns:
            Tuple of (dynamic_values, static_aux_data) where:
            - dynamic_values: List of values that can be transformed
            - static_aux_data: Static data needed for reconstruction
        """
        ...
    
    @classmethod
    def tree_unflatten(cls, aux_data: Any, dynamic_values: List[Any]) -> 'TreeProtocol':
        """Reconstruct operator from flattened representation.
        
        Args:
            aux_data: Static auxiliary data from tree_flatten
            dynamic_values: Dynamic values to reconstruct with
            
        Returns:
            Reconstructed operator instance
        """
        ...


class DependencyAware(Protocol):
    """Protocol for operators that declare dependencies.
    
    Implement this to help XCS optimize execution order and parallelization.
    """
    
    def get_dependencies(self) -> List[str]:
        """Return list of dependency identifiers.
        
        Returns:
            List of strings identifying required resources/operators
        """
        ...
    
    def get_static_config(self) -> Dict[str, Any]:
        """Return static configuration that doesn't change.
        
        Returns:
            Dictionary of static configuration values
        """
        ...


class StaticAnalysisHints:
    """Hints to help static analysis and optimization."""
    
    vectorizable: bool = False
    stateless: bool = False
    cacheable: bool = False
    pure: bool = False
    associative: bool = False
    commutative: bool = False
    
    def __init__(self, **kwargs):
        """Initialize hints with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Operator:
    """Base class for advanced operators.
    
    Provides full functionality including:
    - Metadata management
    - Tree transformation support
    - Dependency tracking
    - Static analysis hints
    
    Most users should just use functions. This is for the 9% who need
    deep integration with XCS and advanced transformations.
    """
    
    _metadata: Dict[str, Any]
    _hints: StaticAnalysisHints
    
    def __init__(self, **kwargs):
        """Initialize operator with keyword arguments."""
        self._metadata = {}
        self._hints = StaticAnalysisHints()
        
        # Set attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __call__(self, *args, **kwargs):
        """Execute the operator.
        
        Subclasses should implement this method.
        
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    def with_metadata(self, **metadata) -> 'Operator':
        """Create a copy with additional metadata.
        
        Args:
            **metadata: Metadata key-value pairs
            
        Returns:
            New operator instance with updated metadata
        """
        new_op = self.__class__(**self.__dict__)
        new_op._metadata.update(metadata)
        return new_op
    
    def with_hints(self, **hints) -> 'Operator':
        """Create a copy with static analysis hints.
        
        Args:
            **hints: Analysis hints (vectorizable, stateless, etc.)
            
        Returns:
            New operator instance with updated hints
        """
        new_op = self.__class__(**self.__dict__)
        new_op._hints = StaticAnalysisHints(**hints)
        return new_op
    
    # Default tree protocol implementation
    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Default tree flattening implementation.
        
        Returns:
            Tuple of (dynamic_values, static_dict)
        """
        dynamic = []
        static = {}
        
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if hasattr(value, '__call__'):
                dynamic.append(value)
            else:
                static[key] = value
        
        return dynamic, static
    
    @classmethod
    def tree_unflatten(cls, static_data: Dict[str, Any], dynamic_values: List[Any]) -> 'Operator':
        """Default tree reconstruction.
        
        Args:
            static_data: Static configuration
            dynamic_values: Dynamic values
            
        Returns:
            Reconstructed operator
        """
        return cls(**static_data)


def static_field(default=None):
    """Mark a field as static (not transformed by XCS).
    
    Args:
        default: Default value for the field
        
    Returns:
        Field descriptor
    """
    # Simple implementation for now
    return default


class operator:
    """Decorator namespace for advanced operator features."""
    
    @staticmethod
    def advanced(cls):
        """Convert a class into an advanced operator.
        
        Adds tree protocol support and other advanced features.
        
        Args:
            cls: Class to convert
            
        Returns:
            Enhanced operator class
        """
        # Ensure it has required methods
        if not hasattr(cls, '__call__'):
            raise TypeError(f"{cls.__name__} must implement __call__ method")
        
        # Add operator functionality if not already present
        if not issubclass(cls, Operator):
            # Create a new class that inherits from both
            class AdvancedOperator(cls, Operator):
                def __init__(self, **kwargs):
                    cls.__init__(self, **kwargs)
                    Operator.__init__(self, **kwargs)
            
            AdvancedOperator.__name__ = cls.__name__
            AdvancedOperator.__module__ = cls.__module__
            return AdvancedOperator
        
        return cls
    
    @staticmethod
    def hints(**hints):
        """Add static analysis hints to an operator.
        
        Args:
            **hints: Analysis hints (vectorizable, stateless, etc.)
            
        Returns:
            Decorator that adds hints
        """
        def decorator(func_or_cls):
            if isinstance(func_or_cls, type):
                # Class decorator
                func_or_cls._hints = StaticAnalysisHints(**hints)
            else:
                # Function decorator
                func_or_cls._operator_hints = hints
            return func_or_cls
        return decorator