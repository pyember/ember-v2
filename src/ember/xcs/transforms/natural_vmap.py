"""Natural vmap implementation with intelligent batch detection.

This module provides vectorization that works transparently with natural Python
functions, automatically detecting batch patterns and parallelizing when beneficial.
"""

import functools
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from ember.xcs.adapters import SmartAdapter
from ember.xcs.introspection import CallStyle, FunctionIntrospector, FunctionMetadata


logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class BatchStyle(Enum):
    """Detected batching patterns."""
    NO_BATCH = "no_batch"              # No batching detected
    SINGLE_LIST = "single_list"        # f([1, 2, 3])
    MULTI_LIST = "multi_list"          # f([1, 2], [3, 4])
    KEYWORD_BATCH = "keyword_batch"    # f(x=[1, 2], y=[3, 4])
    MIXED_BATCH = "mixed_batch"        # f([1, 2], y=[3, 4])
    DICT_BATCH = "dict_batch"          # f([{...}, {...}])


@dataclass
class BatchInfo:
    """Information about detected batch structure."""
    style: BatchStyle
    batch_size: int
    batch_positions: List[int] = None      # Positions of batched args
    batch_keywords: List[str] = None       # Names of batched kwargs
    non_batch_data: Dict[str, Any] = None # Non-batched parameters
    
    def __post_init__(self):
        if self.batch_positions is None:
            self.batch_positions = []
        if self.batch_keywords is None:
            self.batch_keywords = []
        if self.non_batch_data is None:
            self.non_batch_data = {}


class SmartBatchDetector:
    """Intelligently detects batch patterns in function inputs."""
    
    def detect(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], 
              metadata: Optional[FunctionMetadata] = None) -> BatchInfo:
        """Detect how inputs are batched.
        
        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            metadata: Optional function metadata for smarter detection
            
        Returns:
            BatchInfo describing the batch structure
        """
        # No inputs
        if not args and not kwargs:
            return BatchInfo(BatchStyle.NO_BATCH, 0)
        
        # Single list argument - most common pattern
        if len(args) == 1 and not kwargs and self._is_batch_like(args[0]):
            return BatchInfo(
                BatchStyle.SINGLE_LIST,
                len(args[0]),
                batch_positions=[0]
            )
        
        # Check for dict batch pattern
        if len(args) == 1 and not kwargs and self._is_dict_batch(args[0]):
            return BatchInfo(
                BatchStyle.DICT_BATCH,
                len(args[0]),
                batch_positions=[0]
            )
        
        # Multiple arguments - check which are batched
        batch_positions = []
        batch_size = None
        
        for i, arg in enumerate(args):
            if self._is_batch_like(arg):
                batch_positions.append(i)
                if batch_size is None:
                    batch_size = len(arg)
                elif batch_size != len(arg):
                    # Inconsistent sizes - not a valid batch
                    logger.warning(f"Inconsistent batch sizes: {batch_size} vs {len(arg)}")
                    return BatchInfo(BatchStyle.NO_BATCH, 0)
        
        # Check keyword arguments
        batch_keywords = []
        for key, value in kwargs.items():
            if self._is_batch_like(value):
                batch_keywords.append(key)
                if batch_size is None:
                    batch_size = len(value)
                elif batch_size != len(value):
                    # Inconsistent sizes
                    logger.warning(f"Inconsistent batch sizes in kwargs: {batch_size} vs {len(value)}")
                    return BatchInfo(BatchStyle.NO_BATCH, 0)
        
        # Determine batch style
        if not batch_positions and not batch_keywords:
            return BatchInfo(BatchStyle.NO_BATCH, 0)
        
        if batch_positions and not batch_keywords:
            style = BatchStyle.MULTI_LIST if len(batch_positions) > 1 else BatchStyle.SINGLE_LIST
        elif batch_keywords and not batch_positions:
            style = BatchStyle.KEYWORD_BATCH
        else:
            style = BatchStyle.MIXED_BATCH
        
        # Collect non-batched data
        non_batch_data = {}
        for i, arg in enumerate(args):
            if i not in batch_positions:
                non_batch_data[f'_arg{i}'] = arg
        for key, value in kwargs.items():
            if key not in batch_keywords:
                non_batch_data[key] = value
        
        return BatchInfo(
            style=style,
            batch_size=batch_size or 0,
            batch_positions=batch_positions,
            batch_keywords=batch_keywords,
            non_batch_data=non_batch_data
        )
    
    def _is_batch_like(self, obj: Any) -> bool:
        """Check if object looks like a batch."""
        return isinstance(obj, (list, tuple)) and len(obj) > 0
    
    def _is_dict_batch(self, obj: Any) -> bool:
        """Check if object is a list of dictionaries."""
        return (isinstance(obj, (list, tuple)) and 
                len(obj) > 0 and 
                all(isinstance(item, dict) for item in obj))
    
    def unbatch(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], 
               batch_info: BatchInfo) -> List[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
        """Yield individual items from batch.
        
        Returns:
            List of (args, kwargs) tuples for each batch element
        """
        items = []
        
        for i in range(batch_info.batch_size):
            # Build args for this batch element
            element_args = []
            for j, arg in enumerate(args):
                if j in batch_info.batch_positions:
                    element_args.append(arg[i])
                else:
                    element_args.append(arg)
            
            # Build kwargs for this batch element
            element_kwargs = {}
            for key, value in kwargs.items():
                if key in batch_info.batch_keywords:
                    element_kwargs[key] = value[i]
                else:
                    element_kwargs[key] = value
            
            items.append((tuple(element_args), element_kwargs))
        
        return items
    
    def rebatch(self, results: List[Any], batch_info: BatchInfo) -> Any:
        """Combine results back into batch structure.
        
        Args:
            results: List of individual results
            batch_info: Original batch structure info
            
        Returns:
            Results in appropriate batch format
        """
        if batch_info.style == BatchStyle.SINGLE_LIST:
            # Simple list of results
            return results
        
        elif batch_info.style == BatchStyle.DICT_BATCH:
            # Results should be dictionaries - combine by key
            if not results or not isinstance(results[0], dict):
                return results
            
            combined = {}
            keys = set()
            for result in results:
                keys.update(result.keys())
            
            for key in keys:
                combined[key] = [r.get(key) for r in results]
            
            return combined
        
        else:
            # Default to list of results
            return results


def natural_vmap(
    func: Optional[F] = None,
    *,
    in_axes: Union[int, Dict[str, int]] = 0,
    out_axis: int = 0,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> Union[F, Callable[[F], F]]:
    """Natural vectorization that adapts to input patterns.
    
    Transforms a function that operates on single elements into one that
    efficiently processes batches, with automatic detection of batch structure.
    
    Args:
        func: Function to vectorize
        in_axes: Which axes contain batch dimensions (unused in natural mode)
        out_axis: Output axis for batched results (unused in natural mode)
        parallel: Whether to use parallel execution
        max_workers: Maximum workers for parallel execution
        
    Returns:
        Vectorized function that preserves natural calling patterns
        
    Examples:
        >>> # Simple function
        >>> @natural_vmap
        ... def square(x):
        ...     return x * x
        >>> 
        >>> square([1, 2, 3, 4])
        [1, 4, 9, 16]
        
        >>> # Multiple arguments
        >>> @natural_vmap
        ... def add(x, y):
        ...     return x + y
        >>> 
        >>> add([1, 2, 3], [4, 5, 6])
        [5, 7, 9]
        
        >>> # Mixed batching
        >>> @natural_vmap
        ... def multiply(x, y=2):
        ...     return x * y
        >>> 
        >>> multiply([1, 2, 3], y=5)
        [5, 10, 15]
    """
    # Decorator factory pattern
    if func is None:
        return functools.partial(
            natural_vmap,
            in_axes=in_axes,
            out_axis=out_axis,
            parallel=parallel,
            max_workers=max_workers,
        )
    
    # Analyze function
    introspector = FunctionIntrospector()
    metadata = introspector.analyze(func)
    
    # Create batch detector and adapter
    detector = SmartBatchDetector()
    adapter = SmartAdapter(metadata)
    
    @functools.wraps(func)
    def vmapped(*args, **kwargs):
        """Vectorized function with natural signature."""
        # Detect batch structure
        batch_info = detector.detect(args, kwargs, metadata)
        
        # No batching - call directly
        if batch_info.style == BatchStyle.NO_BATCH:
            return func(*args, **kwargs)
        
        # Check for operator-style function
        if metadata.call_style == CallStyle.OPERATOR:
            # Use adapter for operator-style functions
            return _vmap_operator_style(func, args, kwargs, batch_info, 
                                      detector, adapter, parallel, max_workers)
        
        # Natural function - use direct batching
        return _vmap_natural_style(func, args, kwargs, batch_info,
                                 detector, parallel, max_workers)
    
    # Preserve metadata
    vmapped._is_vmapped = True
    vmapped._original_func = func
    vmapped._vmap_parallel = parallel
    
    return vmapped


def _vmap_natural_style(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    batch_info: BatchInfo,
    detector: SmartBatchDetector,
    parallel: bool,
    max_workers: Optional[int],
) -> Any:
    """Vectorize a natural Python function."""
    # Get individual items
    items = detector.unbatch(args, kwargs, batch_info)
    
    # Process items
    if parallel and len(items) > 1:
        results = _process_parallel(func, items, max_workers)
    else:
        results = _process_sequential(func, items)
    
    # Rebatch results
    return detector.rebatch(results, batch_info)


def _vmap_operator_style(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    batch_info: BatchInfo,
    detector: SmartBatchDetector,
    adapter: SmartAdapter,
    parallel: bool,
    max_workers: Optional[int],
) -> Any:
    """Vectorize an operator-style function."""
    # Convert inputs to internal format
    inputs = adapter.adapt_inputs(args, kwargs)
    
    # Process batch
    results = []
    
    if batch_info.style == BatchStyle.DICT_BATCH:
        # Special handling for dict batches
        for item in args[0]:
            result = func(inputs=item)
            results.append(result)
    else:
        # Standard batching
        items = detector.unbatch(args, kwargs, batch_info)
        
        for item_args, item_kwargs in items:
            item_inputs = adapter.adapt_inputs(item_args, item_kwargs)
            result = func(inputs=item_inputs)
            results.append(result)
    
    # Combine results
    return _combine_operator_results(results)


def _process_sequential(func: Callable, items: List[Tuple]) -> List[Any]:
    """Process items sequentially."""
    results = []
    for args, kwargs in items:
        try:
            result = func(*args, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            raise
    return results


def _process_parallel(func: Callable, items: List[Tuple], 
                    max_workers: Optional[int]) -> List[Any]:
    """Process items in parallel."""
    results = [None] * len(items)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for i, (args, kwargs) in enumerate(items):
            future = executor.submit(func, *args, **kwargs)
            futures[future] = i
        
        # Collect results in order
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:
                logger.error(f"Error in parallel execution: {e}")
                raise
    
    return results


def _combine_operator_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine operator-style results intelligently."""
    if not results:
        return {}
    
    # Get all keys from results
    keys = set()
    for result in results:
        if isinstance(result, dict):
            keys.update(result.keys())
    
    # Combine by key
    combined = {}
    for key in keys:
        values = []
        for result in results:
            if isinstance(result, dict) and key in result:
                values.append(result[key])
            else:
                values.append(None)
        combined[key] = values
    
    return combined