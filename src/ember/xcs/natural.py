"""Natural API implementations for XCS.

Provides transparent function decoration without forcing dictionary I/O.
This is a cleaner implementation that works with the existing XCS infrastructure.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from ember.xcs.jit import jit as original_jit
from ember.xcs.transforms.vmap import vmap as original_vmap
from ember.xcs.introspection import CallStyle, FunctionIntrospector
from ember.xcs.adapters import SmartAdapter


logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def natural_jit(func: F) -> F:
    """Natural JIT that preserves function signatures.
    
    Drop-in replacement for @jit that works with natural Python functions.
    """
    # Check if this is already a natural vmapped function
    if hasattr(func, '_is_vmapped') and hasattr(func, '_original_func'):
        # This is a vmapped function - JIT the original function instead
        original = getattr(func, '_original_func')
        jitted_original = natural_jit(original)
        
        # Re-apply vmap to the jitted function
        vmapped_jitted = natural_vmap(jitted_original)
        
        # Copy over metadata
        vmapped_jitted._is_jit_compiled = True
        vmapped_jitted._is_vmapped = True
        
        return vmapped_jitted
    
    # Analyze function
    introspector = FunctionIntrospector()
    metadata = introspector.analyze(func)
    
    # If already operator style, use original JIT
    if metadata.call_style == CallStyle.OPERATOR:
        return original_jit(func)
    
    # For natural functions, create adapter
    adapter = SmartAdapter(metadata)
    
    # Convert to internal format
    internal_func = adapter.adapt_to_internal(func)
    
    # Apply original JIT
    jitted_internal = original_jit(internal_func)
    
    # Wrap back to natural signature
    @functools.wraps(func)
    def natural_wrapper(*args, **kwargs):
        # Convert inputs
        inputs = adapter.adapt_inputs(args, kwargs)
        
        # Call JIT function
        result = jitted_internal(inputs=inputs)
        
        # Convert outputs
        return adapter.adapt_outputs(result)
    
    # Preserve metadata
    natural_wrapper._is_jit_compiled = True
    natural_wrapper._original_func = func
    
    return natural_wrapper


def natural_vmap(func: F) -> F:
    """Natural vmap that detects batch patterns automatically.
    
    Drop-in replacement for vmap that works with natural Python functions.
    """
    # Analyze function
    introspector = FunctionIntrospector()
    metadata = introspector.analyze(func)
    
    @functools.wraps(func)
    def vmapped(*args, **kwargs):
        # Detect batch pattern
        batch_info = _detect_batch_pattern(args, kwargs)
        
        if batch_info is None:
            # No batching - call directly
            return func(*args, **kwargs)
        
        # Process batch based on pattern
        if batch_info['style'] == 'single_list':
            # f([1, 2, 3]) -> [1, 4, 9]
            results = []
            for item in args[0]:
                # Pass through any keyword arguments
                results.append(func(item, **kwargs))
            return results
            
        elif batch_info['style'] == 'multi_list':
            # f([1, 2], [3, 4]) -> [4, 6]
            results = []
            for items in zip(*args):
                results.append(func(*items))
            return results
            
        elif batch_info['style'] == 'keyword_batch':
            # f(x=[1, 2], y=[3, 4]) -> [4, 6]
            results = []
            batch_size = batch_info['size']
            batch_keys = batch_info['keys']
            
            for i in range(batch_size):
                call_kwargs = {}
                for k, v in kwargs.items():
                    if k in batch_keys:
                        call_kwargs[k] = v[i]
                    else:
                        call_kwargs[k] = v
                results.append(func(**call_kwargs))
            return results
            
        elif batch_info['style'] == 'mixed_batch':
            # f([1, 2], y=[3, 4]) -> [4, 8]
            results = []
            batch_size = batch_info['size']
            
            for i in range(batch_size):
                call_args = []
                for j, arg in enumerate(args):
                    if j in batch_info['arg_positions']:
                        call_args.append(arg[i])
                    else:
                        call_args.append(arg)
                
                call_kwargs = {}
                for k, v in kwargs.items():
                    if k in batch_info['keys']:
                        call_kwargs[k] = v[i]
                    else:
                        call_kwargs[k] = v
                        
                results.append(func(*call_args, **call_kwargs))
            return results
            
        elif batch_info['style'] == 'dict_batch':
            # f([{...}, {...}]) -> [...]
            results = []
            for item in args[0]:
                # Pass through any keyword arguments
                results.append(func(item, **kwargs))
            return results
        
        # Fallback to original behavior
        return func(*args, **kwargs)
    
    # Preserve metadata
    vmapped._is_vmapped = True
    vmapped._original_func = func
    
    return vmapped


def _detect_batch_pattern(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect batch pattern in function arguments."""
    # No arguments
    if not args and not kwargs:
        return None
    
    # Single list argument
    if len(args) == 1 and not kwargs and isinstance(args[0], (list, tuple)):
        # Check if it's a dict batch
        if args[0] and all(isinstance(item, dict) for item in args[0]):
            return {'style': 'dict_batch', 'size': len(args[0])}
        return {'style': 'single_list', 'size': len(args[0])}
    
    # Multiple list arguments
    if args and all(isinstance(arg, (list, tuple)) for arg in args) and not kwargs:
        sizes = [len(arg) for arg in args]
        if len(set(sizes)) == 1:  # All same size
            return {'style': 'multi_list', 'size': sizes[0]}
    
    # Keyword-only batching
    if not args and kwargs:
        batch_keys = []
        batch_size = None
        
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                batch_keys.append(k)
                if batch_size is None:
                    batch_size = len(v)
                elif batch_size != len(v):
                    return None  # Inconsistent sizes
        
        if batch_keys:
            return {'style': 'keyword_batch', 'size': batch_size, 'keys': batch_keys}
    
    # Mixed batching
    if args and kwargs:
        batch_size = None
        arg_positions = []
        batch_keys = []
        
        # Check args
        for i, arg in enumerate(args):
            if isinstance(arg, (list, tuple)):
                arg_positions.append(i)
                if batch_size is None:
                    batch_size = len(arg)
                elif batch_size != len(arg):
                    return None
        
        # Check kwargs
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                batch_keys.append(k)
                if batch_size is None:
                    batch_size = len(v)
                elif batch_size != len(v):
                    return None
        
        if arg_positions or batch_keys:
            return {
                'style': 'mixed_batch',
                'size': batch_size,
                'arg_positions': arg_positions,
                'keys': batch_keys
            }
    
    return None