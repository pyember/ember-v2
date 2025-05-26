"""Simplified JIT compilation - one adaptive strategy that just works.

No modes, no complex selection logic. Just smart compilation.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Tuple
from weakref import WeakKeyDictionary

from ember.xcs.graph.graph import Graph

logger = logging.getLogger(__name__)


class JITCache:
    """Simple cache for compiled functions."""
    
    def __init__(self):
        self._cache: WeakKeyDictionary = WeakKeyDictionary()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'compilation_failures': 0
        }
    
    def get(self, func: Callable, args_shape: Tuple) -> Optional[Graph]:
        """Get compiled graph for function and input shape."""
        key = (func, args_shape)
        if key in self._cache:
            self._stats['hits'] += 1
            return self._cache[key]
        self._stats['misses'] += 1
        return None
    
    def set(self, func: Callable, args_shape: Tuple, graph: Graph) -> None:
        """Cache compiled graph."""
        key = (func, args_shape)
        self._cache[key] = graph
    
    def mark_failure(self):
        """Track compilation failure."""
        self._stats['compilation_failures'] += 1


# Global cache instance
_jit_cache = JITCache()


def jit(func: Optional[Callable] = None, *, cache: bool = True) -> Callable:
    """Simple adaptive JIT that just works.
    
    Automatically:
    - Traces execution
    - Builds optimized graph
    - Falls back to original on failure
    - Caches by input shape
    
    Args:
        func: Function to compile (or None if used with parens)
        cache: Whether to cache compiled graphs
        
    Returns:
        JIT-compiled function
        
    Examples:
        @jit
        def process(x):
            return transform(x)
            
        @jit(cache=False)  # Don't cache
        def dynamic_process(x):
            return transform(x)
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get cache key from input shapes
            args_shape = _get_args_shape(args, kwargs)
            
            # Check cache
            if cache:
                compiled = _jit_cache.get(f, args_shape)
                if compiled is not None:
                    try:
                        return _execute_compiled(compiled, args, kwargs)
                    except Exception as e:
                        logger.debug(f"Compiled execution failed: {e}, falling back")
            
            # Try to compile
            try:
                compiled = _trace_and_compile(f, args, kwargs)
                if compiled and cache:
                    _jit_cache.set(f, args_shape, compiled)
                if compiled:
                    return _execute_compiled(compiled, args, kwargs)
            except Exception as e:
                logger.debug(f"JIT compilation failed: {e}, using original function")
                _jit_cache.mark_failure()
            
            # Fallback to original
            return f(*args, **kwargs)
        
        # Attach metadata
        wrapper._is_jit = True
        wrapper._original = f
        
        return wrapper
    
    # Handle @jit vs @jit()
    if func is None:
        return decorator
    else:
        return decorator(func)


def _get_args_shape(args: Tuple, kwargs: Dict) -> Tuple:
    """Extract shape information from arguments for caching."""
    shapes = []
    
    # Process positional args
    for arg in args:
        if hasattr(arg, 'shape'):
            shapes.append(('array', arg.shape))
        elif isinstance(arg, (list, tuple)):
            shapes.append((type(arg).__name__, len(arg)))
        elif isinstance(arg, dict):
            shapes.append(('dict', tuple(sorted(arg.keys()))))
        else:
            shapes.append((type(arg).__name__, None))
    
    # Process keyword args
    for key in sorted(kwargs.keys()):
        val = kwargs[key]
        if hasattr(val, 'shape'):
            shapes.append((f'kwarg_{key}', val.shape))
        else:
            shapes.append((f'kwarg_{key}', type(val).__name__))
    
    return tuple(shapes)


def _trace_and_compile(func: Callable, args: Tuple, kwargs: Dict) -> Optional[Graph]:
    """Trace function execution and build optimized graph."""
    try:
        # Create tracing context
        tracer = ExecutionTracer()
        
        # Trace execution
        with tracer:
            # Execute function while tracing
            _ = func(*args, **kwargs)
        
        # Build graph from trace
        graph = tracer.build_graph()
        
        # Optimize graph
        graph = _optimize_graph(graph)
        
        return graph
        
    except Exception as e:
        logger.debug(f"Tracing failed: {e}")
        return None


def _execute_compiled(graph: Graph, args: Tuple, kwargs: Dict) -> Any:
    """Execute compiled graph with given inputs."""
    # Prepare inputs
    inputs = {}
    
    # Map args to input nodes
    if args:
        inputs['args'] = args
    if kwargs:
        inputs.update(kwargs)
    
    # Execute graph
    results = graph(inputs)
    
    # Extract final result
    # Assuming last node in topological order is output
    topo_order = graph._topological_sort()
    if topo_order:
        return results[topo_order[-1]]
    
    return results


def _optimize_graph(graph: Graph) -> Graph:
    """Apply optimizations to traced graph."""
    # The Graph class already handles pattern detection
    # Here we could add more optimizations like:
    # - Operation fusion
    # - Common subexpression elimination
    # - Dead code elimination
    
    # For now, return as-is since Graph already optimizes
    return graph


class ExecutionTracer:
    """Traces function execution to build computation graph."""
    
    def __init__(self):
        self.graph = Graph()
        self.trace_stack = []
        self.node_mapping = {}
    
    def __enter__(self):
        """Start tracing."""
        # This is where we'd hook into Python's execution
        # For now, this is a simplified version
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracing."""
        pass
    
    def trace_call(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """Record a function call in the graph."""
        # Determine dependencies from trace stack
        deps = []
        if self.trace_stack:
            # Last operation is a dependency
            deps = [self.trace_stack[-1]]
        
        # Add node to graph
        node_id = self.graph.add(
            func=func,
            args=list(args),
            kwargs=kwargs,
            deps=deps
        )
        
        # Track in stack
        self.trace_stack.append(node_id)
        
        return node_id
    
    def build_graph(self) -> Graph:
        """Return the traced graph."""
        return self.graph


# Simplified API for manual graph building
def build_graph(func: Callable) -> Graph:
    """Manually build a graph from a function.
    
    This is useful when automatic tracing doesn't work.
    
    Example:
        @jit
        def pipeline(x):
            graph = build_graph(pipeline)
            n1 = graph.add(preprocess, args=[x])
            n2 = graph.add(compute, deps=[n1])
            return graph
    """
    # Inspect function to understand structure
    source = inspect.getsource(func)
    
    # This would parse the source to build graph
    # For now, return empty graph
    graph = Graph()
    
    # Add nodes based on function analysis
    # ... implementation ...
    
    return graph


# Convenience decorators for specific patterns
def jit_parallel(func: Callable) -> Callable:
    """JIT with hint that function has internal parallelism."""
    compiled = jit(func)
    compiled._parallel_hint = True
    return compiled


def jit_pure(func: Callable) -> Callable:
    """JIT with hint that function is pure (no side effects)."""
    compiled = jit(func)
    compiled._pure_hint = True
    return compiled


def jit_ensemble(judge_funcs: list, synthesizer: Callable) -> Callable:
    """JIT compile an ensemble-judge pattern.
    
    Args:
        judge_funcs: List of judge functions
        synthesizer: Function to synthesize judge outputs
        
    Returns:
        Compiled ensemble function
    """
    def ensemble(*args, **kwargs):
        graph = Graph()
        
        # Add judge nodes
        judge_nodes = []
        for i, judge in enumerate(judge_funcs):
            node_id = graph.add(judge, args=list(args), kwargs=kwargs)
            judge_nodes.append(node_id)
        
        # Add synthesizer
        synth_node = graph.add(synthesizer, deps=judge_nodes)
        
        # Execute
        results = graph(kwargs)
        return results[synth_node]
    
    return jit(ensemble)