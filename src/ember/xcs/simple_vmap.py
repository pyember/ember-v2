"""Simple vmap implementation for natural Python functions.

This provides a clean vmap that works with regular Python functions
without requiring internal dictionary representations.
"""

import functools
from typing import Any, Callable, List, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


def vmap(func: F) -> F:
    """Transform a single-item function to work on batches.
    
    Works naturally with regular Python functions:
    - f(x) -> vmap(f)([x1, x2, x3])
    - f(x, y) -> vmap(f)([x1, x2], [y1, y2])
    - f(x, y=1) -> vmap(f)([x1, x2], y=1)
    
    Args:
        func: Function to transform
        
    Returns:
        Batch-processing version of the function
        
    Example:
        >>> def square(x):
        ...     return x * x
        >>> batch_square = vmap(square)
        >>> batch_square([1, 2, 3])
        [1, 4, 9]
    """
    @functools.wraps(func)
    def batch_wrapper(*args, **kwargs):
        # Handle different calling patterns
        if not args:
            # No positional args - error
            return func(*args, **kwargs)
        
        # Check if first arg is a list/tuple
        if isinstance(args[0], (list, tuple)):
            # Batch mode
            results = []
            
            if len(args) == 1:
                # Single batched argument: f([x1, x2, x3])
                for item in args[0]:
                    results.append(func(item, **kwargs))
            else:
                # Multiple batched arguments: f([x1, x2], [y1, y2])
                # Check all are sequences of same length
                batch_size = len(args[0])
                for i, arg in enumerate(args[1:], 1):
                    if not isinstance(arg, (list, tuple)):
                        # Mix of batch and non-batch args
                        # Treat non-batch as shared
                        continue
                    if len(arg) != batch_size:
                        raise ValueError(f"Inconsistent batch sizes: {batch_size} vs {len(arg)}")
                
                # Process batch
                for i in range(batch_size):
                    batch_args = []
                    for arg in args:
                        if isinstance(arg, (list, tuple)) and len(arg) == batch_size:
                            batch_args.append(arg[i])
                        else:
                            # Non-batch arg - pass through
                            batch_args.append(arg)
                    results.append(func(*batch_args, **kwargs))
            
            return results
        else:
            # Not a batch - call normally
            return func(*args, **kwargs)
    
    # Mark as vmapped
    batch_wrapper._is_vmapped = True
    batch_wrapper._original = func
    
    return batch_wrapper