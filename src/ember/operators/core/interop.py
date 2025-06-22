"""Core interoperability layer for seamless tier mixing.

This module ensures that operators from all three tiers (simple, advanced, 
experimental) can work together seamlessly in compositions, transformations,
and optimizations.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar, Union
import functools
import inspect

__all__ = [
    'UniversalOperator',
    'ensure_operator',
    'adapt',
    'get_capabilities',
    'is_operator',
]

T = TypeVar('T')
R = TypeVar('R')


class UniversalOperator(Protocol[T, R]):
    """Protocol that all operators satisfy across tiers.
    
    This is the minimal interface - just being callable.
    Additional protocols are detected dynamically.
    """
    
    def __call__(self, inputs: T) -> R:
        """Execute the operator."""
        ...


class TreeProtocol(Protocol):
    """Optional protocol for tree transformations."""
    
    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Flatten to (values, aux_data)."""
        ...
    
    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], values: List[Any]) -> 'TreeProtocol':
        """Reconstruct from flattened representation."""
        ...


class DependencyProtocol(Protocol):
    """Optional protocol for dependency declaration."""
    
    def get_dependencies(self) -> List[str]:
        """Return list of dependencies."""
        ...
    
    def get_static_config(self) -> Dict[str, Any]:
        """Return static configuration."""
        ...


def is_operator(obj: Any) -> bool:
    """Check if an object can be used as an operator.
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object is callable
    """
    return callable(obj)


def get_capabilities(op: Any) -> Dict[str, bool]:
    """Detect what protocols and capabilities an operator supports.
    
    Args:
        op: Operator to analyze
        
    Returns:
        Dictionary of capability flags
    """
    return {
        # Basic
        'callable': callable(op),
        'is_function': inspect.isfunction(op),
        'is_method': inspect.ismethod(op),
        'is_class': inspect.isclass(op) or hasattr(op, '__class__'),
        
        # Protocols
        'tree_protocol': hasattr(op, 'tree_flatten') and hasattr(op, 'tree_unflatten'),
        'dependency_aware': hasattr(op, 'get_dependencies'),
        'has_static_config': hasattr(op, 'get_static_config'),
        
        # Decorators
        'validated': hasattr(op, '_validation_spec'),
        'measured': hasattr(op, '_measurement_enabled'),
        'signature': hasattr(op, '_signature_spec'),
        
        # Experimental
        'traced': hasattr(op, '_trace_enabled'),
        'jit_compiled': hasattr(op, '_jit_compiled'),
        'pattern_optimized': hasattr(op, '_pattern_optimize'),
        
        # Composition
        'is_chain': hasattr(op, '_is_chain'),
        'is_parallel': hasattr(op, '_is_parallel'),
        'is_ensemble': hasattr(op, '_is_ensemble'),
    }


class OperatorAdapter:
    """Base adapter class for operator interoperability."""
    
    def __init__(self, operator: Any):
        """Initialize with an operator to adapt.
        
        Args:
            operator: The operator to wrap
        """
        self.operator = operator
        self.capabilities = get_capabilities(operator)
        
        # Preserve metadata
        self.__name__ = getattr(operator, '__name__', f'adapted_{type(operator).__name__}')
        self.__doc__ = getattr(operator, '__doc__', None)
        
        # Copy over any special attributes
        for attr in ['_validation_spec', '_measurement_enabled', '_signature_spec']:
            if hasattr(operator, attr):
                setattr(self, attr, getattr(operator, attr))
    
    def __call__(self, *args, **kwargs):
        """Execute the wrapped operator."""
        return self.operator(*args, **kwargs)


class SimpleToAdvancedAdapter(OperatorAdapter):
    """Adapts simple functions to advanced operators with tree protocol.
    
    This allows simple functions to participate in:
    - JAX-style tree transformations
    - XCS optimizations
    - Advanced composition patterns
    """
    
    def __init__(self, func: Callable, **static_config):
        """Initialize adapter.
        
        Args:
            func: Simple function to adapt
            **static_config: Static configuration for the function
        """
        super().__init__(func)
        self.static_config = static_config
    
    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Flatten for tree transformations.
        
        Simple functions are treated as leaves with no dynamic state.
        
        Returns:
            Tuple of ([], metadata) since functions have no dynamic state
        """
        metadata = {
            'func': self.operator,
            'name': self.__name__,
            **self.static_config
        }
        return [], metadata
    
    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], values: List[Any]) -> 'SimpleToAdvancedAdapter':
        """Reconstruct from tree representation.
        
        Args:
            aux_data: Metadata including function and config
            values: Dynamic values (empty for simple functions)
            
        Returns:
            Reconstructed adapter
        """
        func = aux_data.pop('func')
        aux_data.pop('name', None)  # Remove name from config
        return cls(func, **aux_data)
    
    def get_dependencies(self) -> List[str]:
        """Return dependencies if specified in static config."""
        return self.static_config.get('dependencies', [])
    
    def get_static_config(self) -> Dict[str, Any]:
        """Return static configuration."""
        return self.static_config.copy()


class AdvancedToSimpleWrapper(OperatorAdapter):
    """Wraps advanced operators to work in simple contexts.
    
    This is mainly for type compatibility and doesn't remove capabilities.
    The advanced operator's protocols are still accessible if needed.
    """
    
    def __init__(self, operator: Any):
        """Initialize wrapper.
        
        Args:
            operator: Advanced operator to wrap
        """
        super().__init__(operator)
        
        # Expose tree protocol methods if available
        if self.capabilities['tree_protocol']:
            self.tree_flatten = operator.tree_flatten
            self.tree_unflatten = operator.tree_unflatten


def ensure_operator(obj: Any, target_tier: Optional[str] = None) -> Any:
    """Ensure object is a proper operator, adapting if necessary.
    
    Args:
        obj: Object to ensure is an operator
        target_tier: Optional target tier ('simple', 'advanced', 'experimental')
        
    Returns:
        An operator that satisfies requirements
        
    Raises:
        TypeError: If object cannot be made into an operator
    """
    if not is_operator(obj):
        raise TypeError(f"Object {obj} is not callable and cannot be used as an operator")
    
    capabilities = get_capabilities(obj)
    
    # No specific target, return as-is if callable
    if target_tier is None:
        return obj
    
    # Adapt based on target tier
    if target_tier == 'advanced':
        if not capabilities['tree_protocol']:
            # Lift simple to advanced
            return SimpleToAdvancedAdapter(obj)
        return obj
    
    elif target_tier == 'simple':
        if capabilities['is_class'] and not capabilities['is_function']:
            # Wrap class-based operator
            return AdvancedToSimpleWrapper(obj)
        return obj
    
    elif target_tier == 'experimental':
        # Experimental tier accepts anything callable
        return obj
    
    else:
        raise ValueError(f"Unknown target tier: {target_tier}")


def adapt(operator: Any, **config) -> Any:
    """Adapt an operator with additional configuration.
    
    This is a convenience function for adding static configuration
    to an operator while potentially lifting it to a higher tier.
    
    Args:
        operator: Operator to adapt
        **config: Additional configuration
        
    Returns:
        Adapted operator
        
    Example:
        >>> simple_func = lambda x: x * 2
        >>> advanced = adapt(simple_func, dependencies=['multiplier'])
        >>> # Now it has tree protocol and dependency info
    """
    if not config:
        return operator
    
    # If it's already advanced, just update config
    if hasattr(operator, 'get_static_config'):
        # Clone and update
        # This would need proper implementation based on operator type
        return operator
    
    # Lift to advanced with config
    return SimpleToAdvancedAdapter(operator, **config)


# Decorator for automatic adaptation
def interoperable(target_tier: Optional[str] = None):
    """Decorator to make operators interoperable.
    
    Args:
        target_tier: Target tier for adaptation
        
    Returns:
        Decorator function
        
    Example:
        >>> @interoperable('advanced')
        ... def process(x):
        ...     return x * 2
        >>> 
        >>> # process now has tree_flatten/unflatten methods
    """
    def decorator(op: Callable) -> Callable:
        return ensure_operator(op, target_tier)
    
    return decorator if target_tier else lambda op: op


# Global registry for cross-tier composition
_composition_adapters = {
    ('simple', 'advanced'): SimpleToAdvancedAdapter,
    ('advanced', 'simple'): AdvancedToSimpleWrapper,
}


def register_adapter(from_tier: str, to_tier: str, adapter_class: type):
    """Register a custom adapter for tier transitions.
    
    Args:
        from_tier: Source tier
        to_tier: Target tier  
        adapter_class: Adapter class to use
    """
    _composition_adapters[(from_tier, to_tier)] = adapter_class