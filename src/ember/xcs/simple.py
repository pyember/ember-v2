"""Simplified XCS API - the new, clean interface.

This is the future of XCS: simple, powerful, automatic.
"""

# Core graph execution
from ember.xcs.graph.graph import Graph, Node

# JIT compilation
from ember.xcs.jit.simple_jit import (
    jit,
    jit_parallel,
    jit_pure,
    jit_ensemble,
    build_graph)

# Transformations
from ember.xcs.transforms.simple_transforms import (
    vmap,
    pmap,
    scan,
    compose,
    parallelize,
    batch)

# Simple execution function
def run(graph: Graph, inputs: dict = None) -> dict:
    """Execute a graph with automatic optimization.
    
    This is the simplest API - just run the graph.
    
    Args:
        graph: The graph to execute
        inputs: Optional input data
        
    Returns:
        Execution results
    """
    return graph(inputs or {})


# Convenience functions for common patterns
def parallel(*funcs, reducer=None):
    """Create a parallel execution pattern.
    
    Args:
        *funcs: Functions to run in parallel
        reducer: Optional function to combine results
        
    Returns:
        Function that executes all in parallel
        
    Example:
        result = parallel(func1, func2, func3)(input)
    """
    return parallelize(list(funcs), reducer)


def pipeline(*funcs):
    """Create a sequential pipeline.
    
    Args:
        *funcs: Functions to compose (left to right)
        
    Returns:
        Pipelined function
        
    Example:
        process = pipeline(load, transform, save)
        process(data)
    """
    # Note: compose is right-to-left, so reverse for pipeline
    return compose(*reversed(funcs))


def ensemble(judges: list, synthesizer):
    """Create an ensemble-judge pattern.
    
    Args:
        judges: List of judge functions
        synthesizer: Function to synthesize judgments
        
    Returns:
        Ensemble function
        
    Example:
        judge = ensemble([judge1, judge2, judge3], synthesize)
        result = judge(content)
    """
    return jit_ensemble(judges, synthesizer)


# Advanced features
class lazy:
    """Lazy evaluation wrapper for building graphs declaratively.
    
    Example:
        @lazy
        def process(x):
            y = expensive_op1(x)
            z = expensive_op2(y)
            return combine(y, z)
        
        # Builds optimized graph on first call
        result = process(data)
    """
    
    def __init__(self, func):
        self.func = func
        self._graph = None
    
    def __call__(self, *args, **kwargs):
        if self._graph is None:
            # Build graph from function
            self._graph = self._trace_to_graph(self.func, args, kwargs)
        
        # Execute graph
        inputs = {"args": args, "kwargs": kwargs}
        results = self._graph(inputs)
        
        # Return final result
        nodes = self._graph._topological_sort()
        return results[nodes[-1]] if nodes else None
    
    def _trace_to_graph(self, func, args, kwargs):
        """Convert function to graph through tracing."""
        # This would use the tracer from simple_jit
        # For now, create simple graph
        graph = Graph()
        
        # Add single node for the function
        graph.add(func, args=list(args), kwargs=kwargs)
        
        return graph


# Export the simple API
__all__ = [
    # Core
    "Graph",
    "Node", 
    "run",
    
    # JIT
    "jit",
    "jit_parallel",
    "jit_pure",
    "jit_ensemble",
    
    # Transforms
    "vmap",
    "pmap",
    "scan",
    "compose",
    "parallelize",
    "batch",
    
    # Patterns
    "parallel",
    "pipeline", 
    "ensemble",
    
    # Advanced
    "lazy",
    "build_graph"]


# Usage examples in docstring
"""
Simple XCS - Powerful Computation Made Easy

Basic Usage:
    from ember.xcs import simple as xcs
    
    # Build graph
    graph = xcs.Graph()
    n1 = graph.add(func1)
    n2 = graph.add(func2, deps=[n1])
    
    # Execute - automatic optimization!
    results = graph()

JIT Compilation:
    @xcs.jit
    def process(x):
        return complex_computation(x)
    
    # Automatically optimized
    result = process(data)

Transformations:
    # Vectorize
    batch_process = xcs.vmap(process_single)
    results = batch_process(batch_data)
    
    # Parallel map
    results = xcs.pmap(expensive_op)(inputs)

Common Patterns:
    # Parallel execution
    result = xcs.parallel(op1, op2, op3)(input)
    
    # Pipeline
    process = xcs.pipeline(load, transform, save)
    process(data)
    
    # Ensemble
    judge = xcs.ensemble([judge1, judge2], synthesize)
    result = judge(content)

The key insight: let the system optimize, you just express what you want!
"""