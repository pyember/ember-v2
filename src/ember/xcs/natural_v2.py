"""Natural API v2 - Fixing leaky abstractions.

This implementation addresses the leaky abstractions identified in the analysis.
"""

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from ember.xcs.jit import jit as original_jit
from ember.xcs.introspection import CallStyle, FunctionIntrospector
from ember.xcs.adapters import SmartAdapter


logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

# Private key for storing metadata
_XCS_METADATA_KEY = '__xcs_metadata__'


class XCSMetadata:
    """Clean metadata storage for XCS transformations."""
    
    def __init__(self):
        self.transformations = []
        self.original_func = None
        self.compilation_stats = {}
    
    def add_transformation(self, name: str):
        """Add a transformation to the metadata."""
        if name not in self.transformations:
            self.transformations.append(name)
    
    def is_transformed(self, name: str) -> bool:
        """Check if a transformation has been applied."""
        return name in self.transformations


def _get_xcs_metadata(func: Callable) -> Optional[XCSMetadata]:
    """Get XCS metadata from a function."""
    return getattr(func, _XCS_METADATA_KEY, None)


def _set_xcs_metadata(func: Callable, metadata: XCSMetadata):
    """Set XCS metadata on a function."""
    setattr(func, _XCS_METADATA_KEY, metadata)


def _ensure_metadata(func: Callable) -> XCSMetadata:
    """Ensure function has metadata, creating if needed."""
    metadata = _get_xcs_metadata(func)
    if metadata is None:
        metadata = XCSMetadata()
        metadata.original_func = func
        _set_xcs_metadata(func, metadata)
    return metadata


def _translate_error(e: Exception, func_name: str) -> Exception:
    """Translate internal errors to user-friendly ones."""
    error_str = str(e)
    
    # Remove internal references
    replacements = {
        "internal_wrapper": func_name,
        "Error adapting": "Error calling",
        "from internal format": "",
        "inputs=": "",
        "JIT graph": "optimized function",
        "missing 1 required keyword-only argument: 'inputs'": "incorrect arguments"
    }
    
    for old, new in replacements.items():
        error_str = error_str.replace(old, new)
    
    # Clean up extra whitespace
    error_str = ' '.join(error_str.split())
    
    # Return appropriate error type
    if "missing" in error_str or "unexpected" in error_str:
        return TypeError(error_str)
    else:
        return RuntimeError(error_str)


def natural_jit(func: F) -> F:
    """JIT compilation that works naturally with Python functions.
    
    Examples:
        >>> @natural_jit
        ... def add(x, y):
        ...     return x + y
        >>> 
        >>> add(2, 3)
        5
    """
    # Check if already transformed
    metadata = _get_xcs_metadata(func)
    if metadata and metadata.is_transformed('jit'):
        return func  # Already JIT compiled
    
    # Handle composition with vmap
    if metadata and metadata.is_transformed('vmap'):
        # Get the original function and JIT it
        original = metadata.original_func or func
        jitted_original = natural_jit(original)
        
        # Re-apply vmap
        return natural_vmap(jitted_original)
    
    # Analyze function
    introspector = FunctionIntrospector()
    func_metadata = introspector.analyze(func)
    
    # Create adapter
    adapter = SmartAdapter(func_metadata)
    
    # Convert to internal format
    internal_func = adapter.adapt_to_internal(func)
    
    # Apply original JIT
    try:
        jitted_internal = original_jit(internal_func)
    except Exception as e:
        # If JIT fails, log warning and return original
        logger.debug(f"JIT compilation failed for {func.__name__}: {e}")
        return func
    
    # Create natural wrapper
    @functools.wraps(func)
    def jit_wrapper(*args, **kwargs):
        try:
            # Convert inputs
            inputs = adapter.adapt_inputs(args, kwargs)
            
            # Call JIT function
            result = jitted_internal(inputs=inputs)
            
            # Convert outputs
            return adapter.adapt_outputs(result)
            
        except Exception as e:
            # Translate error to user-friendly message
            raise _translate_error(e, func.__name__) from None
    
    # Set up metadata
    metadata = _ensure_metadata(jit_wrapper)
    metadata.add_transformation('jit')
    metadata.original_func = func
    
    # Preserve function attributes
    jit_wrapper.__name__ = func.__name__
    jit_wrapper.__doc__ = func.__doc__
    jit_wrapper.__annotations__ = getattr(func, '__annotations__', {})
    jit_wrapper.__module__ = func.__module__
    
    return jit_wrapper


def natural_vmap(func: F) -> F:
    """Vectorization that automatically detects batch patterns.
    
    Examples:
        >>> @natural_vmap
        ... def square(x):
        ...     return x * x
        >>> 
        >>> square([1, 2, 3])
        [1, 4, 9]
    """
    # Check if already transformed
    metadata = _get_xcs_metadata(func)
    if metadata and metadata.is_transformed('vmap'):
        return func  # Already vmapped
    
    @functools.wraps(func)
    def vmap_wrapper(*args, **kwargs):
        try:
            # Detect batch pattern
            batch_info = _detect_batch_pattern(args, kwargs)
            
            if batch_info is None:
                # No batching - call directly
                return func(*args, **kwargs)
            
            # Process batch based on pattern
            results = _process_batch(func, args, kwargs, batch_info)
            return results
            
        except Exception as e:
            # Check if it's a user error or our error
            if isinstance(e, (TypeError, ValueError)):
                raise  # User error - pass through
            else:
                # Our error - translate it
                raise _translate_error(e, func.__name__) from None
    
    # Set up metadata
    metadata = _ensure_metadata(vmap_wrapper)
    metadata.add_transformation('vmap')
    metadata.original_func = func
    
    # Preserve function attributes
    vmap_wrapper.__name__ = func.__name__
    vmap_wrapper.__doc__ = func.__doc__
    vmap_wrapper.__annotations__ = getattr(func, '__annotations__', {})
    vmap_wrapper.__module__ = func.__module__
    
    return vmap_wrapper


def _process_batch(func: Callable, args: Tuple, kwargs: Dict, 
                  batch_info: Dict) -> Any:
    """Process batch with clean error handling."""
    style = batch_info['style']
    
    if style == 'single_list':
        # f([1, 2, 3]) -> [1, 4, 9]
        # Pass through any non-batch keyword arguments
        return [func(item, **kwargs) for item in args[0]]
        
    elif style == 'multi_list':
        # f([1, 2], [3, 4]) -> [4, 6]
        return [func(*items, **kwargs) for items in zip(*args)]
        
    elif style == 'keyword_batch':
        # f(x=[1, 2], y=[3, 4]) -> [4, 6]
        results = []
        batch_size = batch_info['size']
        batch_keys = batch_info['keys']
        
        for i in range(batch_size):
            call_kwargs = {
                k: (v[i] if k in batch_keys else v)
                for k, v in kwargs.items()
            }
            results.append(func(**call_kwargs))
        return results
        
    elif style == 'dict_batch':
        # f([{...}, {...}]) -> [{...}, {...}]
        return [func(item, **kwargs) for item in args[0]]
        
    else:
        # Mixed or unknown pattern
        return _process_mixed_batch(func, args, kwargs, batch_info)


def _process_mixed_batch(func: Callable, args: Tuple, kwargs: Dict,
                        batch_info: Dict) -> List[Any]:
    """Handle mixed batching patterns."""
    results = []
    batch_size = batch_info['size']
    
    for i in range(batch_size):
        # Build arguments for this item
        call_args = []
        for j, arg in enumerate(args):
            if j in batch_info.get('arg_positions', []):
                call_args.append(arg[i])
            else:
                call_args.append(arg)
        
        call_kwargs = {}
        for k, v in kwargs.items():
            if k in batch_info.get('keys', []):
                call_kwargs[k] = v[i]
            else:
                call_kwargs[k] = v
                
        results.append(func(*call_args, **call_kwargs))
    
    return results


def _detect_batch_pattern(args: Tuple[Any, ...], 
                         kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect batch pattern with improved logic."""
    # Empty inputs
    if not args and not kwargs:
        return None
    
    # Single list argument - most common case
    if len(args) == 1 and not kwargs:
        arg = args[0]
        if isinstance(arg, (list, tuple)) and arg:
            # Check if it's a dict batch
            if all(isinstance(item, dict) for item in arg):
                return {'style': 'dict_batch', 'size': len(arg)}
            else:
                return {'style': 'single_list', 'size': len(arg)}
    
    # Analyze all inputs for batch structure
    batch_info = {
        'style': None,
        'size': None,
        'arg_positions': [],
        'keys': []
    }
    
    # Check positional arguments
    for i, arg in enumerate(args):
        if isinstance(arg, (list, tuple)) and arg:
            if batch_info['size'] is None:
                batch_info['size'] = len(arg)
            elif batch_info['size'] != len(arg):
                return None  # Inconsistent sizes
            batch_info['arg_positions'].append(i)
    
    # Check keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple)) and value:
            if batch_info['size'] is None:
                batch_info['size'] = len(value)
            elif batch_info['size'] != len(value):
                return None  # Inconsistent sizes
            batch_info['keys'].append(key)
    
    # Determine style
    if not batch_info['arg_positions'] and not batch_info['keys']:
        return None  # No batching
    
    if batch_info['arg_positions'] and not batch_info['keys']:
        if len(args) == len(batch_info['arg_positions']):
            batch_info['style'] = 'multi_list'
        else:
            batch_info['style'] = 'mixed_batch'
    elif batch_info['keys'] and not batch_info['arg_positions']:
        batch_info['style'] = 'keyword_batch'
    else:
        batch_info['style'] = 'mixed_batch'
    
    return batch_info


def get_transformation_info(func: Callable) -> Dict[str, Any]:
    """Get user-friendly information about transformations.
    
    This replaces get_jit_stats with cleaner information.
    """
    metadata = _get_xcs_metadata(func)
    
    if metadata is None:
        return {
            'transformed': False,
            'transformations': []
        }
    
    return {
        'transformed': True,
        'transformations': metadata.transformations,
        'has_jit': metadata.is_transformed('jit'),
        'has_vmap': metadata.is_transformed('vmap'),
    }