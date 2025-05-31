"""Proof of concept for natural XCS API.

This demonstrates how XCS transformations can work with natural Python functions
without forcing dictionary I/O conventions.
"""

import functools
import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

F = TypeVar('F', bound=Callable[..., Any])


class CallStyle(Enum):
    """Function calling conventions."""
    NATURAL = "natural"          # def f(x, y, z=1)
    OPERATOR = "operator"        # def forward(self, *, inputs)
    KEYWORD_ONLY = "keyword"     # def f(*, x, y)
    COMPLEX = "complex"          # Mixed patterns


class FunctionAdapter:
    """Adapts between natural Python functions and internal representations."""
    
    def __init__(self, func: Callable):
        self.func = func
        self.signature = inspect.signature(func)
        self.params = list(self.signature.parameters.values())
        self.call_style = self._determine_call_style()
        
    def _determine_call_style(self) -> CallStyle:
        """Inspect function to determine its calling convention."""
        # Skip 'self' for methods
        params = [p for p in self.params if p.name != 'self']
        
        if not params:
            return CallStyle.NATURAL
            
        # Check for operator style: single keyword-only 'inputs' parameter
        if (len(params) == 1 and 
            params[0].kind == inspect.Parameter.KEYWORD_ONLY and
            params[0].name == 'inputs'):
            return CallStyle.OPERATOR
            
        # Check if all parameters are keyword-only
        if all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params):
            return CallStyle.KEYWORD_ONLY
            
        # Check if all parameters are positional
        if all(p.kind in (inspect.Parameter.POSITIONAL_ONLY, 
                         inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in params):
            return CallStyle.NATURAL
            
        return CallStyle.COMPLEX
    
    def to_internal(self) -> Callable:
        """Convert function to internal dictionary-based format."""
        if self.call_style == CallStyle.OPERATOR:
            return self.func  # Already in internal format
            
        @functools.wraps(self.func)
        def internal_func(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
            # Extract arguments based on calling style
            if self.call_style == CallStyle.NATURAL:
                # Map positional parameters
                args = []
                for param in self.params:
                    if param.name in inputs:
                        args.append(inputs[param.name])
                    elif param.default != inspect.Parameter.empty:
                        args.append(param.default)
                    else:
                        raise TypeError(f"Missing required argument: {param.name}")
                
                result = self.func(*args)
            else:
                # Keyword style
                result = self.func(**inputs)
            
            # Wrap result if not already a dict
            if not isinstance(result, dict):
                return {"result": result}
            return result
        
        return internal_func
    
    def wrap(self, internal_func: Callable) -> Callable:
        """Wrap internal function to match original signature."""
        if self.call_style == CallStyle.OPERATOR:
            return internal_func
            
        @functools.wraps(self.func)
        def wrapped(*args, **kwargs):
            # Convert arguments to internal format
            inputs = {}
            
            # Handle positional arguments
            for i, (arg, param) in enumerate(zip(args, self.params)):
                inputs[param.name] = arg
            
            # Handle keyword arguments
            inputs.update(kwargs)
            
            # Call internal function
            result = internal_func(inputs=inputs)
            
            # Unwrap result if it's a simple dict with 'result' key
            if isinstance(result, dict) and len(result) == 1 and "result" in result:
                return result["result"]
            return result
        
        return wrapped


class BatchDetector:
    """Detects batch structure in function inputs."""
    
    @staticmethod
    def detect_batching(args: Tuple, kwargs: Dict, signature: inspect.Signature) -> 'BatchInfo':
        """Detect if inputs contain batch dimensions."""
        # Single list argument
        if len(args) == 1 and isinstance(args[0], (list, tuple)) and not kwargs:
            return BatchInfo(style='positional_list', batch_size=len(args[0]))
            
        # Multiple list arguments
        if args and all(isinstance(arg, (list, tuple)) for arg in args):
            sizes = [len(arg) for arg in args]
            if len(set(sizes)) == 1:  # All same size
                return BatchInfo(style='positional_multi', batch_size=sizes[0])
                
        # Keyword arguments with lists
        list_kwargs = {k: v for k, v in kwargs.items() 
                      if isinstance(v, (list, tuple))}
        if list_kwargs:
            sizes = [len(v) for v in list_kwargs.values()]
            if len(set(sizes)) == 1:  # All same size
                return BatchInfo(style='keyword', batch_size=sizes[0], 
                               batch_keys=set(list_kwargs.keys()))
                
        return BatchInfo(style='none', batch_size=0)


class BatchInfo:
    """Information about detected batching."""
    def __init__(self, style: str, batch_size: int, batch_keys: Optional[set] = None):
        self.style = style
        self.batch_size = batch_size
        self.batch_keys = batch_keys or set()


def natural_vmap(func: F) -> F:
    """Natural vectorization that adapts to input patterns."""
    adapter = FunctionAdapter(func)
    
    @functools.wraps(func)
    def vmapped(*args, **kwargs):
        batch_info = BatchDetector.detect_batching(args, kwargs, adapter.signature)
        
        if batch_info.style == 'none':
            # No batching detected
            return func(*args, **kwargs)
            
        elif batch_info.style == 'positional_list':
            # vmap(square)([1, 2, 3]) -> [1, 4, 9]
            results = []
            for item in args[0]:
                results.append(func(item))
            return results
            
        elif batch_info.style == 'positional_multi':
            # vmap(add)([1, 2], [3, 4]) -> [4, 6]
            results = []
            for items in zip(*args):
                results.append(func(*items))
            return results
            
        elif batch_info.style == 'keyword':
            # vmap(add)(x=[1, 2], y=[3, 4]) -> [4, 6]
            results = []
            batch_size = batch_info.batch_size
            for i in range(batch_size):
                call_kwargs = {}
                for k, v in kwargs.items():
                    if k in batch_info.batch_keys:
                        call_kwargs[k] = v[i]
                    else:
                        call_kwargs[k] = v
                results.append(func(**call_kwargs))
            return results
            
        else:
            raise ValueError(f"Unsupported batch style: {batch_info.style}")
    
    return vmapped


def natural_jit(func: F) -> F:
    """Natural JIT that preserves function signatures."""
    adapter = FunctionAdapter(func)
    
    # Convert to internal representation
    internal_func = adapter.to_internal()
    
    # Apply JIT compilation (simplified for POC)
    # In real implementation, this would use the actual JIT compiler
    compiled_internal = internal_func  # Placeholder
    
    # Wrap back to original signature
    return adapter.wrap(compiled_internal)


# Demonstration
if __name__ == "__main__":
    # Natural function with JIT
    @natural_jit
    def add(x, y):
        return x + y
    
    print(f"add(2, 3) = {add(2, 3)}")  # Works naturally!
    
    # Natural function with vmap
    @natural_vmap
    def square(x):
        return x * x
    
    print(f"square([1, 2, 3]) = {square([1, 2, 3])}")  # [1, 4, 9]
    
    # Multiple arguments with vmap
    @natural_vmap
    def multiply(x, y):
        return x * y
    
    print(f"multiply([1, 2], [3, 4]) = {multiply([1, 2], [3, 4])}")  # [3, 8]
    print(f"multiply(x=[1, 2], y=[3, 4]) = {multiply(x=[1, 2], y=[3, 4])}")  # [3, 8]
    
    # Complex function
    @natural_jit
    @natural_vmap
    def compute(x, y, z=1):
        return (x + y) * z
    
    print(f"compute([1, 2], [3, 4], z=2) = {compute([1, 2], [3, 4], z=2)}")  # [8, 12]