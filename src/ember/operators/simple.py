"""Simple operator improvements - minimal changes, maximum impact.

Following Larry Page: 10x improvements with minimal code changes.
This can be integrated into the existing system without breaking anything.
"""

import functools
import inspect
from typing import Any, Callable, Optional, TypeVar, Union, get_type_hints

from ember.core.registry.operator.base import Operator
from ember.core.registry.specification import Specification


F = TypeVar('F', bound=Callable[..., Any])


def operator(func_or_class: F) -> F:
    """Simple operator decorator - works with functions and classes.
    
    Examples:
        # Function operator
        @operator
        def multiply(x: float) -> float:
            return x * 2
            
        # Class operator (no base class needed!)
        @operator
        class Pipeline:
            def __call__(self, text: str) -> str:
                return process(text)
                
        # Works with existing operators too
        @operator
        class MyOperator(Operator):
            def forward(self, x):
                return x * 2
    """
    
    if inspect.isfunction(func_or_class):
        # Convert function to operator
        return _function_to_operator(func_or_class)
    
    elif inspect.isclass(func_or_class):
        # Enhance class
        return _enhance_class(func_or_class)
    
    else:
        # Already an instance - wrap it
        return _wrap_instance(func_or_class)


def _function_to_operator(func: Callable) -> Any:
    """Convert a function to an operator instance."""
    
    # Create operator class
    class FunctionOperator(SimpleOperator):
        def __init__(self):
            super().__init__()
            self._func = func
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__
            
        def forward(self, *args, **kwargs):
            return self._func(*args, **kwargs)
            
        def __repr__(self):
            return f"<operator {self._func.__name__}>"
    
    # Return instance
    return FunctionOperator()


def _enhance_class(cls: type) -> type:
    """Enhance a class to be an operator."""
    
    # If already an operator, return as-is
    if issubclass(cls, Operator):
        return cls
    
    # Create operator subclass
    class EnhancedOperator(cls, SimpleOperator):
        """Enhanced operator from user class."""
        
        def forward(self, *args, **kwargs):
            # If class has __call__, use it
            if hasattr(self, '__call__') and self.__class__.__call__ is not EnhancedOperator.__call__:
                return self.__call__(*args, **kwargs)
            else:
                raise NotImplementedError(
                    f"{cls.__name__} must implement __call__ or forward"
                )
    
    # Preserve class metadata
    EnhancedOperator.__name__ = cls.__name__
    EnhancedOperator.__module__ = cls.__module__
    EnhancedOperator.__doc__ = cls.__doc__
    
    return EnhancedOperator


def _wrap_instance(instance: Any) -> Any:
    """Wrap an instance to be operator-like."""
    
    if hasattr(instance, '__call__'):
        # Callable instance - make it an operator
        class InstanceOperator(SimpleOperator):
            def __init__(self):
                super().__init__()
                self._instance = instance
                
            def forward(self, *args, **kwargs):
                return self._instance(*args, **kwargs)
                
            def __repr__(self):
                return f"<operator {instance}>"
        
        return InstanceOperator()
    
    else:
        raise TypeError(f"Cannot convert {type(instance)} to operator")


class SimpleOperator(Operator):
    """Base class for simple operators - no specification required!
    
    This is a minimal subclass of Operator that:
    1. Makes specification optional
    2. Supports natural calling patterns
    3. Works with simple forward() or __call__()
    """
    
    # No specification required!
    specification = None
    
    def __call__(self, *args, **kwargs):
        """Smart calling that handles multiple patterns."""
        
        # If user overrode __call__ in subclass, avoid recursion
        if self.__class__.__call__ is not SimpleOperator.__call__:
            # Call forward instead
            return self._smart_forward(*args, **kwargs)
        
        # Otherwise delegate to forward
        return self._smart_forward(*args, **kwargs)
    
    def _smart_forward(self, *args, **kwargs):
        """Call forward with smart argument handling."""
        
        # Get forward method
        forward = getattr(self, 'forward', None)
        if not forward or not callable(forward):
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement forward() or override __call__()"
            )
        
        # Inspect forward signature
        sig = inspect.signature(forward)
        params = list(sig.parameters.values())
        
        # Remove self parameter
        if params and params[0].name == 'self':
            params = params[1:]
        
        # Smart calling based on signature
        if not params:
            # No parameters
            return forward()
        
        elif len(params) == 1:
            # Single parameter
            param = params[0]
            
            # Check if it expects kwargs
            if param.kind == param.VAR_KEYWORD:
                return forward(**kwargs)
            
            # Check if it's keyword-only
            elif param.kind == param.KEYWORD_ONLY:
                if param.name in kwargs:
                    return forward(**{param.name: kwargs[param.name]})
                elif args:
                    return forward(**{param.name: args[0]})
                else:
                    return forward()
            
            # Regular parameter
            else:
                if args:
                    return forward(args[0])
                elif param.name in kwargs:
                    return forward(kwargs[param.name])
                else:
                    return forward()
        
        else:
            # Multiple parameters - try natural calling
            try:
                return forward(*args, **kwargs)
            except TypeError:
                # Fall back to positional only
                if args:
                    return forward(*args)
                else:
                    return forward(**kwargs)
    
    def forward(self, *args, **kwargs):
        """Override this or __call__ to implement your operator."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward() or override __call__()"
        )


def validate_simple(func: F) -> F:
    """Simple validation decorator using type hints.
    
    Example:
        @validate_simple
        def process(text: str) -> dict:
            return {"result": text.upper()}
    """
    
    hints = get_type_hints(func)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Simple type checking based on hints
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        
        for param_name, value in bound.arguments.items():
            if param_name in hints and param_name != 'return':
                expected_type = hints[param_name]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"{func.__name__}() argument '{param_name}' must be "
                        f"{expected_type.__name__}, not {type(value).__name__}"
                    )
        
        result = func(*args, **kwargs)
        
        # Check return type
        if 'return' in hints:
            expected_return = hints['return']
            if not isinstance(result, expected_return):
                raise TypeError(
                    f"{func.__name__}() must return {expected_return.__name__}, "
                    f"not {type(result).__name__}"
                )
        
        return result
    
    return wrapper


# Convenience functions for common patterns

def make_operator(func: Callable, name: Optional[str] = None) -> Any:
    """Convert any callable to an operator.
    
    Example:
        model_op = make_operator(model.generate, name="model_generate")
    """
    op = operator(func)
    if name:
        op.__name__ = name
    return op


def chain_simple(*operators) -> Any:
    """Chain operators together simply.
    
    Example:
        pipeline = chain_simple(clean_text, tokenize, classify)
        result = pipeline("Hello world")
    """
    
    @operator
    def chained_operator(x):
        result = x
        for op in operators:
            result = op(result)
        return result
    
    chained_operator.__name__ = f"chain({', '.join(str(op) for op in operators)})"
    return chained_operator


def parallel_simple(*operators) -> Any:
    """Run operators in parallel on the same input.
    
    Example:
        ensemble = parallel_simple(model1, model2, model3)
        results = ensemble(input_data)  # Returns list of results
    """
    
    @operator  
    def parallel_operator(x):
        # In simple version, just run sequentially
        # Real implementation would use ThreadPoolExecutor
        return [op(x) for op in operators]
    
    parallel_operator.__name__ = f"parallel({', '.join(str(op) for op in operators)})"
    return parallel_operator