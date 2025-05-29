"""XCS Graph - Simplified.

The graph IS the intermediate representation.
Simple, powerful, automatic parallelism detection.
"""

# The only graph implementation you need
from ember.xcs.graph.graph import Graph, Node

# Kept for compatibility
from ember.xcs.graph.dependency_analyzer import DependencyAnalyzer

# Backward compatibility aliases
XCSGraph = Graph
XCSNode = Node

# Legacy imports for compatibility
try:
    from ember.xcs.graph.graph_builder import GraphBuilder
except ImportError:
    GraphBuilder = None

__all__ = [
    "Graph",
    "Node",
    "DependencyAnalyzer",
    # Compatibility
    "XCSGraph",
    "XCSNode",
    "GraphBuilder"]