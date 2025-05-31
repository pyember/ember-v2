"""Natural JIT implementation supporting transparent function decoration.

This module provides JIT compilation that works with natural Python functions
without forcing dictionary I/O conventions.
"""

import functools
import logging
from typing import Any, Callable, Optional, Type, TypeVar, Union, cast

from ember.xcs.adapters import SmartAdapter, UniversalAdapter
from ember.xcs.introspection import CallStyle, FunctionIntrospector, FunctionMetadata
from ember.xcs.jit.cache import get_cache
from ember.xcs.jit.core import (
    JITSettings, _create_operator_forward_proxy, _get_strategy, 
    _jit_function, _jit_operator_class
)
from ember.xcs.jit.strategies import Strategy


logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def natural_jit(
    func: Optional[F] = None,
    *,
    strategy: Optional[Union[str, Strategy]] = None,
    force_trace: bool = False,
    recursive: bool = True,
    cache: Optional[Any] = None,
    preserve_stochasticity: bool = True,
) -> Union[F, Callable[[F], F]]:
    """Natural JIT compilation that preserves function signatures.
    
    This is a drop-in replacement for the standard jit decorator that works
    transparently with natural Python functions, operators, and methods.
    
    Args:
        func: Function to compile (or None if used as decorator factory)
        strategy: Compilation strategy name or instance
        force_trace: Force execution tracing for graph building
        recursive: Recursively compile nested functions
        cache: Custom cache instance (uses global cache if None)
        preserve_stochasticity: Preserve random behavior across calls
        
    Returns:
        JIT-compiled function with preserved signature
        
    Examples:
        >>> # Natural function
        >>> @natural_jit
        ... def add(x, y):
        ...     return x + y
        >>> 
        >>> add(2, 3)  # Works naturally!
        5
        
        >>> # With options
        >>> @natural_jit(strategy='enhanced')
        ... def multiply(x, y, z=1):
        ...     return x * y * z
    """
    # Decorator factory pattern
    if func is None:
        return functools.partial(
            natural_jit,
            strategy=strategy,
            force_trace=force_trace,
            recursive=recursive,
            cache=cache,
            preserve_stochasticity=preserve_stochasticity,
        )
    
    # Analyze function
    introspector = FunctionIntrospector()
    metadata = introspector.analyze(func)
    
    # Choose compilation path based on function type
    if _is_operator_class(func):
        # Operator class - use enhanced operator compilation
        return _natural_jit_operator_class(
            func, metadata, strategy, force_trace, 
            recursive, cache, preserve_stochasticity
        )
    else:
        # Regular function - use natural compilation
        return _natural_jit_function(
            func, metadata, strategy, force_trace,
            recursive, cache, preserve_stochasticity
        )


def _natural_jit_function(
    func: F,
    metadata: FunctionMetadata,
    strategy: Optional[Union[str, Strategy]],
    force_trace: bool,
    recursive: bool,
    cache: Optional[Any],
    preserve_stochasticity: bool,
) -> F:
    """JIT compile a natural Python function."""
    
    # Create settings
    settings = JITSettings(
        strategy=strategy,
        force_trace=force_trace,
        recursive=recursive,
        custom_cache=cache,
        preserve_stochasticity=preserve_stochasticity,
    )
    
    # Get compilation strategy
    strategy_instance = _get_strategy(settings.strategy)
    
    # Create adapter
    adapter = SmartAdapter(metadata)
    
    # Check if function is already in internal format
    if metadata.call_style == CallStyle.OPERATOR:
        # Already internal format - compile directly
        compiled_internal = _jit_function(func, strategy_instance, settings)
        return compiled_internal
    
    # Convert to internal format for compilation
    internal_func = adapter.adapt_to_internal(func)
    
    # Compile the internal representation
    compiled_internal = _jit_function(internal_func, strategy_instance, settings)
    
    # Wrap back to natural signature
    @functools.wraps(func)
    def natural_wrapper(*args, **kwargs):
        """Natural signature wrapper around JIT-compiled function."""
        try:
            # Convert inputs to internal format
            inputs = adapter.adapt_inputs(args, kwargs)
            
            # Call compiled function
            result = compiled_internal(inputs=inputs)
            
            # Convert outputs back to natural format
            return adapter.adapt_outputs(result)
            
        except Exception as e:
            # Fall back to original function on compilation errors
            logger.debug(f"JIT execution failed for {func.__name__}, falling back: {e}")
            return func(*args, **kwargs)
    
    # Preserve metadata
    natural_wrapper._is_jit_compiled = True
    natural_wrapper._jit_strategy = strategy_instance.__class__.__name__
    natural_wrapper._original_func = func
    natural_wrapper._compiled_func = compiled_internal
    
    return cast(F, natural_wrapper)


def _natural_jit_operator_class(
    cls: Type,
    metadata: FunctionMetadata,
    strategy: Optional[Union[str, Strategy]],
    force_trace: bool,
    recursive: bool,
    cache: Optional[Any],
    preserve_stochasticity: bool,
) -> Type:
    """JIT compile an operator class with natural calling support."""
    
    # Create settings
    settings = JITSettings(
        strategy=strategy,
        force_trace=force_trace,
        recursive=recursive,
        custom_cache=cache,
        preserve_stochasticity=preserve_stochasticity,
    )
    
    # Get strategy instance
    strategy_instance = _get_strategy(settings.strategy)
    
    # Check if operator already supports natural calling
    if hasattr(cls, '__call__'):
        original_call = cls.__call__
        
        # Analyze the __call__ method
        call_metadata = FunctionIntrospector().analyze(original_call)
        
        # If it already accepts natural arguments, enhance it
        if call_metadata.call_style != CallStyle.OPERATOR:
            return _enhance_natural_operator(cls, strategy_instance, settings)
    
    # Use standard operator compilation but add natural calling support
    jit_cls = _jit_operator_class(cls, strategy_instance, settings)
    
    # Enhance with natural calling
    return _add_natural_calling_to_operator(jit_cls)


def _enhance_natural_operator(cls: Type, strategy: Strategy, settings: JITSettings) -> Type:
    """Enhance an operator that already supports natural calling."""
    
    class NaturalJITOperator(cls):
        """JIT-enhanced operator with natural calling support."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Create adapter for forward method
            self._forward_adapter = SmartAdapter()
            
            # Compile forward method
            internal_forward = self._forward_adapter.adapt_to_internal(self.forward)
            
            # Get cache
            cache = settings.custom_cache or get_cache()
            
            # Compile
            self._compiled_forward = strategy.compile(
                internal_forward,
                force_trace=settings.force_trace,
                recursive=settings.recursive,
                cache=cache,
                preserve_stochasticity=settings.preserve_stochasticity
            )
        
        def forward(self, *args, **kwargs):
            """JIT-compiled forward with natural signature."""
            # Adapt inputs
            inputs = self._forward_adapter.adapt_inputs(args, kwargs)
            
            # Call compiled function
            result = self._compiled_forward(inputs=inputs)
            
            # Adapt outputs
            return self._forward_adapter.adapt_outputs(result)
    
    # Preserve class name and docs
    NaturalJITOperator.__name__ = cls.__name__ + "_NaturalJIT"
    NaturalJITOperator.__qualname__ = cls.__qualname__ + "_NaturalJIT"
    NaturalJITOperator.__doc__ = cls.__doc__
    
    return NaturalJITOperator


def _add_natural_calling_to_operator(cls: Type) -> Type:
    """Add natural calling support to a JIT-compiled operator."""
    
    # Store original __call__
    original_call = cls.__call__
    
    def natural_call(self, *args, **kwargs):
        """Enhanced __call__ supporting both natural and dict styles."""
        # Check if this is a natural call
        if args or not all(k == "inputs" for k in kwargs):
            # Natural style call - adapt it
            adapter = SmartAdapter()
            inputs = adapter.adapt_inputs(args, kwargs)
            result = original_call(self, inputs=inputs)
            return adapter.adapt_outputs(result)
        else:
            # Dictionary style call - pass through
            return original_call(self, **kwargs)
    
    # Replace __call__ with natural version
    cls.__call__ = natural_call
    
    return cls


def _is_operator_class(obj: Any) -> bool:
    """Check if object is an operator class."""
    # Duck typing - check for operator-like attributes
    return (
        isinstance(obj, type) and
        hasattr(obj, 'forward') and
        (hasattr(obj, 'specification') or hasattr(obj, '__call__'))
    )