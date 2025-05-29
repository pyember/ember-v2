"""Simplified transformations using the new Graph system.

These replace the complex transformation classes with simple functions.
"""

import logging
from typing import Any, Callable, List, Optional, Tuple, Union

from ember.xcs.graph.graph import Graph
from ember.xcs.jit.simple_jit import jit

logger = logging.getLogger(__name__)


def vmap(func: Callable, in_axes: int = 0, out_axes: int = 0) -> Callable:
    """Vectorizing map - automatically parallelizes over batched inputs.
    
    Args:
        func: Function to vectorize
        in_axes: Which axis to map over (default 0)
        out_axes: Which axis to stack outputs on (default 0)
        
    Returns:
        Vectorized function
        
    Example:
        # Vectorize over batch dimension
        batch_process = vmap(process_single)
        results = batch_process(batch_data)  # Automatically parallel!
    """
    def vmapped_func(inputs: Any) -> Any:
        # Determine batch size
        if hasattr(inputs, 'shape'):
            batch_size = inputs.shape[in_axes]
        elif isinstance(inputs, (list, tuple)):
            batch_size = len(inputs)
        else:
            raise ValueError(f"Cannot determine batch size from {type(inputs)}")
        
        # Build graph for parallel execution
        graph = Graph()
        
        # Create nodes for each batch element
        nodes = []
        for i in range(batch_size):
            # Extract batch element
            if hasattr(inputs, '__getitem__'):
                if in_axes == 0:
                    batch_input = inputs[i]
                else:
                    # Handle other axes
                    batch_input = inputs.take(i, axis=in_axes)
            else:
                batch_input = inputs[i]
            
            # Add node
            node_id = graph.add(
                func=func,
                args=[batch_input],
                name=f"vmap_{func.__name__}_{i}"
            )
            nodes.append(node_id)
        
        # Execute graph - automatically parallel!
        results = graph({})
        
        # Stack results
        outputs = [results[node_id] for node_id in nodes]
        return _stack_outputs(outputs, out_axes)
    
    # Preserve function metadata
    vmapped_func.__name__ = f"vmap({func.__name__})"
    vmapped_func._is_vmap = True
    vmapped_func._original = func
    
    # Apply JIT for extra optimization
    return jit(vmapped_func)


def pmap(func: Callable, axis_name: Optional[str] = None) -> Callable:
    """Parallel map - explicitly parallel execution over inputs.
    
    Like vmap but guarantees parallel execution (no auto-vectorization).
    
    Args:
        func: Function to parallelize  
        axis_name: Optional name for the mapped axis
        
    Returns:
        Parallel-mapped function
        
    Example:
        # Parallel processing
        parallel_process = pmap(expensive_computation)
        results = parallel_process(inputs)  # Runs in parallel
    """
    def pmapped_func(inputs: List[Any]) -> List[Any]:
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("pmap requires list or tuple of inputs")
        
        # Build graph
        graph = Graph()
        
        # Add nodes
        nodes = []
        for i, inp in enumerate(inputs):
            node_id = graph.add(
                func=func,
                args=[inp],
                name=f"pmap_{func.__name__}_{i}"
            )
            nodes.append(node_id)
        
        # Execute with explicit parallelism
        results = graph({}, parallel=len(inputs))
        
        # Return results in order
        return [results[node_id] for node_id in nodes]
    
    # Metadata
    pmapped_func.__name__ = f"pmap({func.__name__})"
    pmapped_func._is_pmap = True
    pmapped_func._axis_name = axis_name
    
    return pmapped_func


def scan(func: Callable, init: Any, xs: Any) -> Tuple[Any, Any]:
    """Scan (fold) over a sequence, building up state.
    
    Unlike vmap/pmap, scan is inherently sequential but we can still
    optimize by building a graph and potentially fusing operations.
    
    Args:
        func: Function (carry, x) -> (new_carry, output)
        init: Initial carry value
        xs: Sequence to scan over
        
    Returns:
        (final_carry, outputs)
    """
    # Build graph for scan
    graph = Graph()
    
    # Determine sequence length
    if hasattr(xs, 'shape'):
        length = xs.shape[0]
    else:
        length = len(xs)
    
    # Build chain of operations
    carry_nodes = []
    output_nodes = []
    
    # First node uses init
    if length > 0:
        first_x = xs[0] if hasattr(xs, '__getitem__') else next(iter(xs))
        first_node = graph.add(
            func=lambda: func(init, first_x),
            name=f"scan_{func.__name__}_0"
        )
        carry_nodes.append(first_node)
        output_nodes.append(first_node)
    
    # Chain remaining nodes
    for i in range(1, length):
        x = xs[i] if hasattr(xs, '__getitem__') else xs[i]
        prev_carry = carry_nodes[-1]
        
        # Node depends on previous carry
        node_id = graph.add(
            func=lambda prev_result=prev_carry, xi=x: func(prev_result, xi),
            deps=[prev_carry],
            name=f"scan_{func.__name__}_{i}"
        )
        carry_nodes.append(node_id)
        output_nodes.append(node_id)
    
    # Execute graph (will be sequential due to dependencies)
    results = graph({})
    
    # Extract final carry and all outputs
    if carry_nodes:
        final_carry = results[carry_nodes[-1]][0]  # First element is carry
        outputs = [results[node][1] for node in output_nodes]  # Second element is output
        return final_carry, _stack_outputs(outputs, 0)
    else:
        return init, []


def compose(*funcs: Callable) -> Callable:
    """Compose multiple functions into a pipeline.
    
    The Graph will automatically optimize the composition.
    
    Args:
        *funcs: Functions to compose (applied right to left)
        
    Returns:
        Composed function
        
    Example:
        pipeline = compose(postprocess, compute, preprocess)
        result = pipeline(data)  # Optimized execution
    """
    def composed(x):
        # Build graph for composition
        graph = Graph()
        
        # Add nodes in reverse order (right to left composition)
        nodes = []
        for i, func in enumerate(reversed(funcs)):
            if i == 0:
                # First function takes input
                node_id = graph.add(func, args=[x])
            else:
                # Subsequent functions depend on previous
                node_id = graph.add(func, deps=[nodes[-1]])
            nodes.append(node_id)
        
        # Execute optimized graph
        results = graph({})
        
        # Return final result
        return results[nodes[-1]] if nodes else x
    
    # Apply JIT for optimization
    return jit(composed)


def parallelize(funcs: List[Callable], reducer: Optional[Callable] = None) -> Callable:
    """Create a parallel ensemble of functions with optional reduction.
    
    This is the generalized ensemble-judge pattern.
    
    Args:
        funcs: List of functions to run in parallel
        reducer: Optional function to reduce results (if None, returns list)
        
    Returns:
        Parallelized function
        
    Example:
        judges = [judge_quality, judge_accuracy, judge_style]
        ensemble = parallelize(judges, synthesize_judgments)
        result = ensemble(content)
    """
    def parallel_ensemble(x):
        graph = Graph()
        
        # Add parallel nodes
        nodes = []
        for i, func in enumerate(funcs):
            node_id = graph.add(func, args=[x], name=f"parallel_{i}")
            nodes.append(node_id)
        
        # Add reducer if provided
        if reducer:
            final_node = graph.add(
                reducer,
                deps=nodes,
                name="reducer"
            )
            results = graph({})
            return results[final_node]
        else:
            # Return all results
            results = graph({})
            return [results[node] for node in nodes]
    
    return jit(parallel_ensemble)


def batch(func: Callable, batch_size: int) -> Callable:
    """Automatically batch inputs for efficient processing.
    
    Args:
        func: Function to batch
        batch_size: Size of batches
        
    Returns:
        Batched function that processes inputs in chunks
    """
    def batched_func(inputs: List[Any]) -> List[Any]:
        graph = Graph()
        results = []
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Create nodes for batch
            if len(batch) == 1:
                # Single item, no parallelism needed
                node_id = graph.add(func, args=[batch[0]])
                results.append(node_id)
            else:
                # Multiple items, use vmap pattern
                batch_nodes = []
                for j, item in enumerate(batch):
                    node_id = graph.add(
                        func,
                        args=[item],
                        name=f"batch_{i}_{j}"
                    )
                    batch_nodes.append(node_id)
                results.extend(batch_nodes)
        
        # Execute graph
        graph_results = graph({})
        
        # Return results in order
        return [graph_results[node_id] for node_id in results]
    
    return batched_func


# Helper functions
def _stack_outputs(outputs: List[Any], axis: int) -> Any:
    """Stack outputs along specified axis."""
    if not outputs:
        return []
    
    # Check first output type
    first = outputs[0]
    
    if hasattr(first, 'shape'):
        # Numpy-like arrays
        import numpy as np
        return np.stack(outputs, axis=axis)
    elif isinstance(first, list):
        # Lists - simple concatenation for axis=0
        if axis == 0:
            return outputs
        else:
            # Transpose for other axes
            return list(zip(*outputs))
    elif isinstance(first, dict):
        # Stack dicts by key
        keys = first.keys()
        return {
            key: _stack_outputs([out[key] for out in outputs], axis)
            for key in keys
        }
    else:
        # Default: just return list
        return outputs


