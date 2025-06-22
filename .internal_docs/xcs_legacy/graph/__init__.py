"""
Graph representation and analysis for XCS.

Provides data structures and utilities for representing, analyzing, and
manipulating computational graphs.
"""

# Dependency analysis
from ember.xcs.graph.dependency_analyzer import DependencyAnalyzer

# Graph building
from ember.xcs.graph.graph_builder import EnhancedTraceGraphBuilder, GraphBuilder

# Core graph representation
from ember.xcs.graph.xcs_graph import XCSGraph, XCSNode

__all__ = [
    # Core graph representation
    "XCSGraph",
    "XCSNode",
    # Dependency analysis
    "DependencyAnalyzer",
    # Graph building
    "GraphBuilder",
    "EnhancedTraceGraphBuilder",
]
