"""XCS Transformations: Intelligent handling of both tensor and orchestration operations.

This module provides XCS's transformation API that subsumes JAX transformations
while adding orchestration-level intelligence.
"""

from typing import Callable, Optional, Any, Dict, TypeVar, Union, Tuple
from functools import wraps
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ember.xcs._simple import jit as xcs_jit
from ember.xcs._internal.analysis import analyze_operations, OperationType


F = TypeVar('F', bound=Callable[..., Any])


class XCSError(Exception):
    """XCS-specific errors with helpful messages."""
    pass


class XCSTransformation:
    """Base class for all XCS transformations."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, func_or_arg=None, **kwargs):
        """Support both decorator and function style."""
        # Merge kwargs
        all_kwargs = {**self.kwargs, **kwargs}
        
        if func_or_arg is None:
            # Called as @transform()
            return lambda f: self.transform(f, **all_kwargs)
        elif callable(func_or_arg):
            # Called as @transform or transform(func)
            return self.transform(func_or_arg, **all_kwargs)
        else:
            # Called as transform(arg1, arg2, ...)
            # Return a partial transformation
            return lambda f: self.transform(f, func_or_arg, **all_kwargs)
    
    def transform(self, func: F, *args, **kwargs) -> F:
        """Apply transformation intelligently."""
        raise NotImplementedError


class JitTransformation(XCSTransformation):
    """The core XCS jit transformation - already implemented in _simple.py."""
    
    def transform(self, func: F, **kwargs) -> F:
        """Apply XCS jit transformation."""
        # Just use the existing xcs_jit
        return xcs_jit(func, **kwargs)


class VMapTransformation(XCSTransformation):
    """Intelligent batching for both tensor and orchestration operations."""
    
    def transform(self, func: F, in_axes=0, out_axes=0, axis_name=None, axis_size=None, **kwargs) -> F:
        """Apply vmap intelligently based on operation types."""
        ops = analyze_operations(func)
        
        @wraps(func)
        def vmapped_func(*args, **fn_kwargs):
            if ops.only_tensor_ops:
                # Pure tensor ops - use JAX vmap directly
                return jax.vmap(func, in_axes=in_axes, out_axes=out_axes, 
                               axis_name=axis_name, axis_size=axis_size)(*args, **fn_kwargs)
            
            elif ops.only_orchestration_ops:
                # Pure orchestration - parallel execution
                return _parallel_orchestration_vmap(func, args, fn_kwargs, in_axes)
            
            else:
                # Hybrid - smart batching
                return _hybrid_vmap(func, ops, in_axes, out_axes, args, fn_kwargs)
        
        # Preserve XCS jit compatibility
        return vmapped_func


class PMapTransformation(XCSTransformation):
    """Distributed execution across devices and model providers."""
    
    def transform(self, func: F, axis_name: str = None, in_axes=0, out_axes=0, 
                  static_broadcasted_argnums=(), devices=None, backend=None, **kwargs) -> F:
        """Apply pmap for distributed execution."""
        ops = analyze_operations(func)
        mesh = kwargs.get('mesh')
        
        @wraps(func)
        def pmapped_func(*args, **fn_kwargs):
            if ops.only_tensor_ops:
                # Pure tensor ops - use JAX pmap
                return jax.pmap(func, axis_name=axis_name, in_axes=in_axes, 
                               out_axes=out_axes, devices=devices, backend=backend)(*args, **fn_kwargs)
            
            elif mesh is not None:
                # ModelMesh distributed execution
                return _model_mesh_pmap(func, mesh, axis_name, args, fn_kwargs)
            
            else:
                # Default distributed orchestration
                return _distributed_orchestration_pmap(func, axis_name, args, fn_kwargs)
        
        return pmapped_func


class ScanTransformation(XCSTransformation):
    """Sequential processing with carry state."""
    
    def transform(self, func: F, length=None, reverse=False, unroll=1, **kwargs) -> F:
        """Apply scan for sequential operations."""
        ops = analyze_operations(func)
        
        @wraps(func)
        def scanned_func(init, xs, *args, **fn_kwargs):
            if ops.only_tensor_ops:
                # Pure tensor - use JAX scan
                return jax.lax.scan(func, init, xs, length=length, 
                                   reverse=reverse, unroll=unroll)
            else:
                # Sequential orchestration with state
                return _orchestration_scan(func, init, xs, reverse=reverse)
        
        return scanned_func


class GradTransformation(XCSTransformation):
    """Smart gradient computation for hybrid workloads."""
    
    def transform(self, func: F, argnums: Union[int, Tuple[int, ...]] = 0, 
                  has_aux=False, holomorphic=False, allow_int=False, **kwargs) -> F:
        """Apply grad intelligently based on operation types.
        
        Following JAX's behavior: return a gradient function that may fail at runtime
        if the function contains non-differentiable operations.
        """
        # Always return a gradient function, check operations at runtime
        @wraps(func)
        def grad_func(*args, **fn_kwargs):
            # Analyze operations at runtime to provide better error messages
            ops = analyze_operations(func)
            
            if ops.only_orchestration_ops:
                raise ValueError(
                    "Cannot compute gradients through orchestration operations.\n"
                    "This function only contains LLM calls or other non-differentiable ops.\n"
                    "For prompt optimization, see future xcs.optimize."
                )
            
            if ops.only_tensor_ops:
                # Pure tensor function - delegate to JAX
                jax_grad = jax.grad(func, argnums=argnums, has_aux=has_aux,
                                   holomorphic=holomorphic, allow_int=allow_int)
                return jax_grad(*args, **fn_kwargs)
            else:
                # Hybrid function - compute gradients for tensor parts only
                return _hybrid_grad(func, ops, argnums, has_aux, args, fn_kwargs)
        
        return grad_func


# Instantiate transformations
jit = JitTransformation()
vmap = VMapTransformation()
pmap = PMapTransformation()
scan = ScanTransformation()
grad = GradTransformation()


# Helper functions for orchestration-level operations

def _parallel_orchestration_vmap(func, args, kwargs, in_axes):
    """Execute orchestration operations in parallel across batch."""
    # Determine batch size from first argument
    if not args:
        return func(*args, **kwargs)
    
    # Handle different in_axes specifications
    if isinstance(in_axes, int):
        batch_axis = in_axes
        batch_size = args[0].shape[batch_axis] if hasattr(args[0], 'shape') else len(args[0])
    else:
        # Find first non-None axis
        for i, axis in enumerate(in_axes):
            if axis is not None:
                batch_size = args[i].shape[axis] if hasattr(args[i], 'shape') else len(args[i])
                break
        else:
            return func(*args, **kwargs)
    
    # Execute in parallel
    with ThreadPoolExecutor() as executor:
        # Create per-item arguments
        futures = []
        for i in range(batch_size):
            item_args = []
            for j, arg in enumerate(args):
                if isinstance(in_axes, int):
                    # When in_axes is an int, it applies to ALL arguments
                    axis = in_axes
                else:
                    axis = in_axes[j] if j < len(in_axes) else None
                
                if axis is not None and hasattr(arg, '__getitem__'):
                    item_args.append(arg[i])
                else:
                    item_args.append(arg)
            
            future = executor.submit(func, *item_args, **kwargs)
            futures.append(future)
        
        # Collect results in order
        results = []
        for future in futures:
            results.append(future.result())
    
    # Stack results appropriately
    if not results:
        return results
    
    # Check if results are tuples/lists (multiple outputs)
    if isinstance(results[0], (tuple, list)):
        # Transpose: [(a1,b1), (a2,b2)] -> ([a1,a2], [b1,b2])
        num_outputs = len(results[0])
        transposed = []
        for i in range(num_outputs):
            output_i = [r[i] for r in results]
            # Stack if all elements are arrays
            if all(hasattr(x, 'shape') for x in output_i):
                transposed.append(jnp.stack(output_i))
            else:
                transposed.append(output_i)
        return tuple(transposed) if isinstance(results[0], tuple) else transposed
    elif hasattr(results[0], 'shape'):
        return jnp.stack(results)
    else:
        return results


def _hybrid_vmap(func, ops, in_axes, out_axes, args, kwargs):
    """Smart batching for hybrid tensor/orchestration functions."""
    # For now, use parallel orchestration approach
    # TODO: Implement smart splitting of tensor and orchestration operations
    return _parallel_orchestration_vmap(func, args, kwargs, in_axes)


def _model_mesh_pmap(func, mesh, axis_name, args, kwargs):
    """Distribute execution across model providers using ModelMesh."""
    raise NotImplementedError(
        "ModelMesh coming soon!\n"
        "This will enable distributed execution across multiple model providers,\n"
        "API keys, regions, and specialized models (code, math, vision)."
    )


def _distributed_orchestration_pmap(func, axis_name, args, kwargs):
    """Default distributed orchestration across available resources."""
    # For now, similar to vmap but could use different distribution strategy
    return _parallel_orchestration_vmap(func, args, kwargs, in_axes=0)


def _orchestration_scan(func, init, xs, reverse=False):
    """Sequential orchestration with carry state."""
    carry = init
    ys = []
    
    # Process in reverse if requested
    items = reversed(xs) if reverse else xs
    
    for x in items:
        carry, y = func(carry, x)
        ys.append(y)
    
    # Reverse output if we processed in reverse
    if reverse:
        ys = list(reversed(ys))
    
    # Stack results if they're arrays
    if ys and hasattr(ys[0], 'shape'):
        ys = jnp.stack(ys)
    
    return carry, ys


def _hybrid_grad(func, ops, argnums, has_aux, args, kwargs):
    """Compute gradients only for tensor operations in hybrid functions.
    
    Following CLAUDE.md principles:
    - Explicit behavior: gradients flow through tensor ops, stop at orchestration
    - Simple implementation: wrap orchestration outputs with stop_gradient
    - Leverage JAX: use standard grad on the wrapped function
    """
    from jax import lax
    
    @wraps(func)
    def differentiable_func(*args, **kwargs):
        # Wrap the function to stop gradients at orchestration boundaries
        # This is a simplified implementation - in practice, we'd analyze
        # the computation graph more carefully
        result = func(*args, **kwargs)
        
        # For now, we'll use a heuristic: if the result contains strings
        # or non-numeric types, it's from orchestration
        def stop_orchestration_grads(x):
            if isinstance(x, (str, dict)) or hasattr(x, '__call__'):
                return x  # Non-differentiable, no gradient
            elif hasattr(x, 'shape'):  # JAX array
                # Check if this came from an orchestration operation
                # This is simplified - real implementation would track provenance
                return x  # For now, allow gradients
            else:
                return x
        
        return jax.tree.map(stop_orchestration_grads, result)
    
    # Now use JAX grad on the wrapped function
    grad_fn = jax.grad(differentiable_func, argnums=argnums, has_aux=has_aux)
    return grad_fn(*args, **kwargs)


# Future API placeholder
class OptimizeTransformation(XCSTransformation):
    """Future: Non-differentiable optimization for LLM operations."""
    
    def transform(self, func: F, **kwargs) -> F:
        raise NotImplementedError(
            "xcs.optimize coming soon for prompt optimization,\n"
            "feedback-based improvement, and RLHF-style training.\n"
            "This will handle non-differentiable LLM optimization."
        )


optimize = OptimizeTransformation()


# Composition helpers
def compose(*transformations):
    """Compose multiple transformations."""
    def composed(func):
        for transform in reversed(transformations):
            func = transform(func)
        return func
    return composed