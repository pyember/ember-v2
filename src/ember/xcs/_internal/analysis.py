"""Operation analysis for XCS transformations.

This module analyzes functions to determine whether they contain
tensor operations, orchestration operations, or both.
"""

from typing import Any, Callable, Set, NamedTuple
from functools import lru_cache
import inspect
import ast
import jax
import jax.numpy as jnp


class OperationType(NamedTuple):
    """Information about operations in a function."""
    has_tensor_ops: bool
    has_orchestration_ops: bool
    tensor_ops: Set[str]
    orchestration_ops: Set[str]
    
    @property
    def only_tensor_ops(self) -> bool:
        """True if function only contains tensor operations."""
        return self.has_tensor_ops and not self.has_orchestration_ops
    
    @property
    def only_orchestration_ops(self) -> bool:
        """True if function only contains orchestration operations."""
        return self.has_orchestration_ops and not self.has_tensor_ops
    
    @property
    def is_hybrid(self) -> bool:
        """True if function contains both types of operations."""
        return self.has_tensor_ops and self.has_orchestration_ops


# Known tensor operation indicators
TENSOR_INDICATORS = {
    'jax', 'jnp', 'np', 'numpy',
    'array', 'tensor', 'matrix',
    'dot', 'matmul', 'einsum',
    'grad', 'vmap', 'pmap', 'scan',
    'nn', 'lax', 'random',
    'encoder', 'decoder', 'embed',
}

# Known orchestration operation indicators
ORCHESTRATION_INDICATORS = {
    'llm', 'model', 'api', 'call',
    'gpt', 'claude', 'anthropic', 'openai',
    'prompt', 'completion', 'chat',
    'generate', 'complete', 'respond',
    'ModelBinding', 'Operator',
    'route', 'router', 'expert',
}


class OperationAnalyzer(ast.NodeVisitor):
    """AST visitor to identify operation types."""
    
    def __init__(self):
        self.tensor_ops = set()
        self.orchestration_ops = set()
        self.in_tensor_context = False
        self.in_orchestration_context = False
    
    def visit_Call(self, node):
        """Analyze function calls."""
        # Get the function name
        func_name = self._get_call_name(node)
        
        if func_name:
            # Check if it's a tensor operation
            if any(indicator in func_name.lower() for indicator in TENSOR_INDICATORS):
                self.tensor_ops.add(func_name)
                self.in_tensor_context = True
            
            # Check if it's an orchestration operation  
            if any(indicator in func_name.lower() for indicator in ORCHESTRATION_INDICATORS):
                self.orchestration_ops.add(func_name)
                self.in_orchestration_context = True
        
        # Continue visiting
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Analyze attribute access."""
        attr_chain = self._get_attribute_chain(node)
        
        if attr_chain:
            # Check tensor indicators
            if any(indicator in attr_chain.lower() for indicator in TENSOR_INDICATORS):
                self.tensor_ops.add(attr_chain)
            
            # Check orchestration indicators
            if any(indicator in attr_chain.lower() for indicator in ORCHESTRATION_INDICATORS):
                self.orchestration_ops.add(attr_chain)
        
        self.generic_visit(node)
    
    def _get_call_name(self, node) -> str:
        """Extract function name from Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_chain(node.func)
        return ""
    
    def _get_attribute_chain(self, node) -> str:
        """Extract full attribute chain (e.g., 'jax.nn.relu')."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts))


def analyze_operations(func: Callable) -> OperationType:
    """Analyze a function to determine its operation types.
    
    This uses AST analysis as a heuristic. For more accurate analysis,
    we could trace the function, but that requires example inputs.
    """
    try:
        # Get source code
        source = inspect.getsource(func)
        
        # Handle indentation issues for nested functions
        # Remove common leading whitespace
        import textwrap
        source = textwrap.dedent(source)
        
        # Parse AST
        tree = ast.parse(source)
        
        # Analyze operations
        analyzer = OperationAnalyzer()
        analyzer.visit(tree)
        
        # Also check function annotations and docstring
        has_tensor = bool(analyzer.tensor_ops)
        has_orchestration = bool(analyzer.orchestration_ops)
        
        # Additional heuristics from function signature
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            if param.annotation:
                ann_str = str(param.annotation)
                if any(t in ann_str for t in ['Array', 'ndarray', 'Tensor']):
                    has_tensor = True
                if any(t in ann_str for t in ['str', 'Model', 'Operator']):
                    has_orchestration = True
        
        return OperationType(
            has_tensor_ops=has_tensor,
            has_orchestration_ops=has_orchestration,
            tensor_ops=analyzer.tensor_ops,
            orchestration_ops=analyzer.orchestration_ops
        )
        
    except (OSError, TypeError, IndentationError, SyntaxError):
        # Can't get source - make conservative assumptions
        # Assume it might have both types
        return OperationType(
            has_tensor_ops=True,
            has_orchestration_ops=True,
            tensor_ops=set(),
            orchestration_ops=set()
        )


def is_jax_array(obj: Any) -> bool:
    """Check if an object is a JAX array."""
    return hasattr(obj, 'shape') and hasattr(obj, 'dtype') and hasattr(obj, '__array__')


def has_jax_arrays(args: tuple, kwargs: dict) -> bool:
    """Check if arguments contain JAX arrays."""
    # Check args
    for arg in args:
        if is_jax_array(arg):
            return True
        if isinstance(arg, (list, tuple)):
            if any(is_jax_array(x) for x in arg):
                return True
    
    # Check kwargs
    for value in kwargs.values():
        if is_jax_array(value):
            return True
        if isinstance(value, (list, tuple)):
            if any(is_jax_array(x) for x in value):
                return True
    
    return False