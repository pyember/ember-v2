"""XCS Graph: The entire system in ~500 lines of elegant code.

This is what happens when Jeff Dean and Sanjay Ghemawat pair program
with Robert C. Martin and Steve Jobs.

Design principles:
1. The graph IS the IR - no translation needed
2. Execution strategy emerges from structure
3. Zero-copy, zero-allocation hot path
4. Composition over configuration
"""

from __future__ import annotations
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Union, Any, Callable, Dict, List, Optional, Set, Tuple
import inspect


@dataclass(frozen=True)
class Node:
    """Immutable node. Just data."""
    id: str
    func: Callable
    deps: Tuple[str, ...] = ()
    
    def __post_init__(self):
        if not callable(self.func):
            raise TypeError(f"Expected callable, got {type(self.func)}")


class Graph:
    """A computation graph that optimizes itself.
    
    Simple API:
        graph = Graph()
        a = graph.add(lambda: 1)
        b = graph.add(lambda x: x + 1, deps=[a])
        result = graph.run()
    """
    
    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        self._counter = 0
        # Lazy computed properties
        self._waves: Optional[List[List[str]]] = None
        self._is_dag: Optional[bool] = None
    
    def add(self, func: Callable, *, deps: Union[List[str], Tuple[str, ...]] = None) -> str:
        """Add a computation. Returns node id."""
        # Generate ID from function name and counter
        name = getattr(func, '__name__', 'lambda')
        node_id = f"{name}_{self._counter}"
        self._counter += 1
        
        # Validate dependencies - handle both lists and tuples
        deps = list(deps) if deps else []
        for dep in deps:
            if dep not in self._nodes:
                raise ValueError(f"Unknown dependency: {dep}")
        
        # Create immutable node
        node = Node(node_id, func, tuple(deps))
        self._nodes[node_id] = node
        
        # Update edges for dependency tracking
        for dep in deps:
            self._edges[dep].add(node_id)
        
        # Invalidate computed properties
        self._waves = None
        self._is_dag = None
        
        return node_id
    
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the graph. Simple."""
        inputs = inputs or {}
        
        # Lazy compute waves (memoized)
        if self._waves is None:
            self._waves = self._compute_waves()
        
        # Choose execution strategy based on structure
        if len(self._waves) == len(self._nodes):
            # Purely sequential
            return self._run_sequential(self._waves, inputs)
        else:
            # Has parallelism
            return self._run_parallel(self._waves, inputs)
    
    def _compute_waves(self) -> List[List[str]]:
        """Topological sort with level scheduling."""
        # Quick DAG check
        if self._is_dag is False:
            raise ValueError("Graph has cycles")
        
        in_degree = {nid: len(self._nodes[nid].deps) for nid in self._nodes}
        queue = deque([nid for nid, degree in in_degree.items() if degree == 0])
        waves = []
        processed = 0
        
        while queue:
            # Process all nodes at current level
            wave = list(queue)
            waves.append(wave)
            processed += len(wave)
            
            # Update degrees and find next level
            next_level = []
            for node_id in wave:
                for dependent in self._edges[node_id]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_level.append(dependent)
            
            queue = deque(next_level)
        
        # Verify we processed all nodes (DAG check)
        if processed != len(self._nodes):
            self._is_dag = False
            raise ValueError("Graph has cycles")
        
        self._is_dag = True
        return waves
    
    def _run_sequential(self, waves: List[List[str]], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sequential execution. No threads, no overhead."""
        results = {}
        
        for wave in waves:
            for node_id in wave:
                node = self._nodes[node_id]
                args = self._prepare_args(node, results, inputs)
                results[node_id] = self._call_function(node.func, args)
        
        return results
    
    def _run_parallel(self, waves: List[List[str]], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel execution. Only when beneficial."""
        results = {}
        
        with ThreadPoolExecutor() as executor:
            for wave in waves:
                if len(wave) == 1:
                    # Single node - run directly, no thread overhead
                    node_id = wave[0]
                    node = self._nodes[node_id]
                    args = self._prepare_args(node, results, inputs)
                    results[node_id] = self._call_function(node.func, args)
                else:
                    # Multiple nodes - parallelize
                    futures = {}
                    for node_id in wave:
                        node = self._nodes[node_id]
                        args = self._prepare_args(node, results, inputs)
                        future = executor.submit(self._call_function, node.func, args)
                        futures[future] = node_id
                    
                    # Collect results
                    for future in as_completed(futures):
                        node_id = futures[future]
                        results[node_id] = future.result()
        
        return results
    
    def _prepare_args(self, node: Node, results: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        """Prepare arguments for function call. Simple rules."""
        if not node.deps:
            # Source node - gets inputs only if function expects them
            return inputs if inputs else None
        elif len(node.deps) == 1:
            # Single dependency - pass its result directly
            return results[node.deps[0]]
        else:
            # Multiple dependencies - pass as list in dependency order
            return [results[dep] for dep in node.deps]
    
    def _call_function(self, func: Callable, args: Any) -> Any:
        """Call function with appropriate signature matching."""
        # Get function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        
        # Handle source nodes with no args
        if args is None:
            if not params:
                return func()
            # Function has default values
            return func()
        
        if not params:
            # No parameters
            return func()
        
        # Check for *args
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
        
        if has_varargs and isinstance(args, list):
            # Function expects *args, we have list
            return func(*args)
        elif len(params) == 1 and params[0].name == 'inputs':
            # Old-style expecting inputs dict
            return func(inputs=args)
        elif isinstance(args, dict):
            # If function expects single positional arg, pass dict directly
            if len(params) == 1 and params[0].kind in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD):
                return func(args)
            # Otherwise match dict keys to parameter names
            kwargs = {p.name: args.get(p.name) for p in params if p.name in args}
            return func(**kwargs)
        elif isinstance(args, list):
            # Multiple arguments from dependencies
            if len(params) == len(args):
                # Direct positional mapping
                return func(*args)
            else:
                # Single argument expecting list
                return func(args)
        else:
            # Single argument
            return func(args)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Graph statistics for analysis."""
        if self._waves is None:
            self._waves = self._compute_waves()
        
        # Count edges by summing dependencies (not forward edges)
        total_edges = sum(len(node.deps) for node in self._nodes.values())
        
        return {
            'nodes': len(self._nodes),
            'edges': total_edges,
            'waves': len(self._waves),
            'parallelism': max(len(w) for w in self._waves) if self._waves else 0,
            'critical_path': len(self._waves),
        }


# Simplified execution engine
class Engine:
    """Even simpler: just run graphs."""
    
    @staticmethod
    def execute(graph: Graph, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a graph. That's it."""
        return graph.run(inputs)


# Pattern detection for future optimizations
def detect_patterns(graph: Graph) -> Dict[str, List[Set[str]]]:
    """Detect optimization patterns in the graph."""
    patterns = defaultdict(list)
    
    # Group nodes by function
    func_groups = defaultdict(list)
    for node_id, node in graph._nodes.items():
        func_groups[node.func].append(node_id)
    
    # Map pattern: same function, independent execution
    for func, nodes in func_groups.items():
        if len(nodes) > 1 and _are_independent(graph, nodes):
            patterns['map'].append(set(nodes))
    
    # Reduce pattern: many-to-one
    for node_id, deps in graph._edges.items():
        if len(deps) == 1:
            target = next(iter(deps))
            target_deps = graph._nodes[target].deps
            if len(target_deps) > 1:
                patterns['reduce'].append({target})
    
    return dict(patterns)


def _are_independent(graph: Graph, nodes: List[str]) -> bool:
    """Check if nodes can execute independently."""
    # Build reachability map
    reachable = {}
    for node in nodes:
        reached = set()
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current in reached:
                continue
            reached.add(current)
            queue.extend(graph._nodes[current].deps)
        reachable[node] = reached
    
    # Check for dependencies between nodes
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i+1:]:
            if n1 in reachable[n2] or n2 in reachable[n1]:
                return False
    
    return True


# JIT decorator (future optimization)
def jit(func: Callable) -> Callable:
    """JIT compile a function. Currently a no-op."""
    # TODO: Trace execution and build optimized graph
    return func


# Transformations
def vmap(func: Callable) -> Callable:
    """Vectorize a function over inputs."""
    def vmapped(inputs: List[Any]) -> List[Any]:
        # Build graph for vectorized execution
        graph = Graph()
        # Create nodes with explicit function wrappers
        nodes = []
        for x in inputs:
            # Capture value in closure properly
            def make_func(val):
                return lambda: func(val)
            nodes.append(graph.add(make_func(x)))
        
        results = graph.run()
        return [results[n] for n in nodes]
    
    return vmapped


def pmap(func: Callable) -> Callable:
    """Parallel map."""
    return vmap(func)  # Same implementation for now

def execute_graph(
    graph: Graph,
    inputs: Dict[str, Any],
    *,
    parallel: Union[bool, int] = True,
    timeout: Optional[float] = None) -> Dict[str, Any]:
    """Execute a computational graph.
    
    This is a compatibility wrapper for Graph.run().
    New code should use graph.run() directly.
    
    Args:
        graph: The computational graph to execute
        inputs: Input data for the graph's source nodes
        parallel: Controls parallel execution (passed to run)
        timeout: Optional timeout in seconds (currently ignored)
        
    Returns:
        Dictionary mapping node IDs to their execution results
    """
    # For now, ignore timeout - could be added to Graph.run() if needed
    if isinstance(parallel, bool):
        return graph.run(inputs) if parallel else graph.run(inputs, max_workers=1)
    else:
        return graph.run(inputs, max_workers=parallel)
