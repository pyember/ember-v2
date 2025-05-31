"""Universal adapter system for natural API support.

Adapts between natural Python functions and internal dictionary-based representations
while preserving signatures, types, and behavior.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from ember.xcs.introspection import (
    CallStyle, FunctionIntrospector, FunctionMetadata, ParameterKind
)


F = TypeVar('F', bound=Callable[..., Any])


class AdapterError(Exception):
    """Raised when adaptation fails."""
    pass


class UniversalAdapter:
    """Adapts any Python callable to/from internal representation."""
    
    def __init__(self, metadata: Optional[FunctionMetadata] = None):
        """Initialize adapter with optional pre-computed metadata."""
        self.metadata = metadata
        self.introspector = FunctionIntrospector()
    
    def adapt_to_internal(self, func: F) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Convert natural function to internal dictionary-based format.
        
        Args:
            func: Natural Python function
            
        Returns:
            Function accepting/returning dictionaries
        """
        # Get metadata if not provided
        if self.metadata is None:
            self.metadata = self.introspector.analyze(func)
        
        # Already in internal format
        if self.metadata.call_style == CallStyle.OPERATOR:
            return func
        
        @functools.wraps(func)
        def internal_wrapper(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Internal representation: dict in, dict out."""
            try:
                # Convert inputs based on function style
                args, kwargs = self._dict_to_args_kwargs(inputs, self.metadata)
                
                # Call original function
                result = func(*args, **kwargs)
                
                # Wrap result in dictionary
                return self._wrap_result(result)
                
            except Exception as e:
                # Provide helpful error context
                raise AdapterError(
                    f"Error adapting {func.__name__} from internal format: {e}"
                ) from e
        
        # Preserve metadata for introspection
        internal_wrapper._original_func = func
        internal_wrapper._adapter_metadata = self.metadata
        
        return internal_wrapper
    
    def adapt_from_internal(self, internal_func: Callable) -> F:
        """Convert internal function back to natural signature.
        
        Args:
            internal_func: Dictionary-based internal function
            
        Returns:
            Function with natural signature
        """
        # Get original metadata
        if hasattr(internal_func, '_adapter_metadata'):
            metadata = internal_func._adapter_metadata
        elif self.metadata:
            metadata = self.metadata
        else:
            # Analyze the internal function
            metadata = self.introspector.analyze(internal_func)
        
        # Already natural
        if metadata.call_style != CallStyle.OPERATOR:
            return internal_func
        
        # Get original function if available
        original_func = getattr(internal_func, '_original_func', None)
        if original_func:
            metadata = self.introspector.analyze(original_func)
        
        @functools.wraps(original_func or internal_func)
        def natural_wrapper(*args, **kwargs):
            """Natural signature wrapper."""
            try:
                # Convert args/kwargs to dictionary
                inputs = self._args_kwargs_to_dict(args, kwargs, metadata)
                
                # Call internal function
                result_dict = internal_func(inputs=inputs)
                
                # Unwrap result
                return self._unwrap_result(result_dict)
                
            except Exception as e:
                # Provide helpful error context
                raise AdapterError(
                    f"Error adapting {internal_func.__name__} to natural format: {e}"
                ) from e
        
        return natural_wrapper
    
    def create_bidirectional_adapter(self, func: F) -> Tuple[Callable, F]:
        """Create both internal and natural versions of a function.
        
        Args:
            func: Function to adapt
            
        Returns:
            Tuple of (internal_version, natural_version)
        """
        metadata = self.introspector.analyze(func)
        
        if metadata.call_style == CallStyle.OPERATOR:
            # Already internal, create natural version
            internal = func
            natural = self.adapt_from_internal(func)
        else:
            # Natural function, create internal version
            internal = self.adapt_to_internal(func)
            natural = func
            
        return internal, natural
    
    def _dict_to_args_kwargs(self, inputs: Dict[str, Any], 
                           metadata: FunctionMetadata) -> Tuple[List[Any], Dict[str, Any]]:
        """Convert dictionary inputs to args and kwargs."""
        args = []
        kwargs = {}
        
        # Handle positional parameters
        for param in metadata.positional_params:
            if param.name in inputs:
                args.append(inputs[param.name])
            elif param.has_default:
                args.append(param.default)
            else:
                raise TypeError(f"Missing required argument: {param.name}")
        
        # Handle keyword parameters
        for param in metadata.keyword_params:
            if param.name in inputs:
                kwargs[param.name] = inputs[param.name]
            elif not param.has_default:
                raise TypeError(f"Missing required keyword argument: {param.name}")
        
        # Handle var args/kwargs if present
        if metadata.has_var_args and '_args' in inputs:
            args.extend(inputs['_args'])
            
        if metadata.has_var_kwargs and '_kwargs' in inputs:
            kwargs.update(inputs['_kwargs'])
        
        return args, kwargs
    
    def _args_kwargs_to_dict(self, args: Tuple[Any, ...], 
                           kwargs: Dict[str, Any], 
                           metadata: FunctionMetadata) -> Dict[str, Any]:
        """Convert args and kwargs to dictionary format."""
        inputs = {}
        
        # Map positional arguments
        for i, (arg, param) in enumerate(zip(args, metadata.positional_params)):
            inputs[param.name] = arg
        
        # Handle excess positional args
        if metadata.has_var_args and len(args) > len(metadata.positional_params):
            inputs['_args'] = args[len(metadata.positional_params):]
        
        # Add keyword arguments
        for param in metadata.keyword_params:
            if param.name in kwargs:
                inputs[param.name] = kwargs[param.name]
        
        # Handle unknown kwargs
        if metadata.has_var_kwargs:
            known_names = {p.name for p in metadata.parameters}
            unknown_kwargs = {k: v for k, v in kwargs.items() if k not in known_names}
            if unknown_kwargs:
                inputs['_kwargs'] = unknown_kwargs
        else:
            # Add all kwargs for functions we don't fully understand
            inputs.update(kwargs)
        
        return inputs
    
    def _wrap_result(self, result: Any) -> Dict[str, Any]:
        """Wrap function result in dictionary."""
        if isinstance(result, dict):
            return result
        else:
            return {"result": result}
    
    def _unwrap_result(self, result_dict: Dict[str, Any]) -> Any:
        """Unwrap result from dictionary."""
        # Single 'result' key - unwrap it
        if len(result_dict) == 1 and "result" in result_dict:
            return result_dict["result"]
        # Otherwise return the full dictionary
        return result_dict


class SmartAdapter(UniversalAdapter):
    """Enhanced adapter with intelligent conversion strategies."""
    
    def adapt_inputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently convert various input patterns to dictionary format.
        
        Handles:
        - Positional args: f(1, 2, 3)
        - Keyword args: f(x=1, y=2)
        - Mixed: f(1, y=2)
        - Single dict: f({'x': 1, 'y': 2})
        """
        # Single dictionary argument - might be inputs dict
        if len(args) == 1 and not kwargs and isinstance(args[0], dict):
            return args[0]
        
        # Convert using metadata if available
        if self.metadata:
            return self._args_kwargs_to_dict(args, kwargs, self.metadata)
        
        # Fallback: combine all inputs
        inputs = {}
        for i, arg in enumerate(args):
            inputs[f'arg{i}'] = arg
        inputs.update(kwargs)
        return inputs
    
    def adapt_outputs(self, result: Union[Any, Dict[str, Any]]) -> Any:
        """Intelligently convert outputs based on context.
        
        Preserves structure when meaningful, unwraps when obvious.
        """
        if not isinstance(result, dict):
            return result
            
        # Check for common patterns
        keys = list(result.keys())
        
        # Single result key - probably wrapped
        if keys == ['result']:
            return result['result']
            
        # Multiple keys or meaningful names - preserve structure
        return result