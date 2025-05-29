"""Test fixtures for the simplified Graph API."""

from ember.xcs.graph import Graph


def create_simple_graph():
    """Create a simple test graph."""
    graph = Graph()
    
    # Add some test nodes
    node1 = graph.add(lambda: {"result": 1})
    node2 = graph.add(lambda: {"result": 2}, deps=[node1])
    node3 = graph.add(lambda: {"result": 3}, deps=[node1, node2])
    
    return graph


def create_parallel_graph():
    """Create a graph with parallel operations."""
    graph = Graph()
    
    # Parallel operations (no dependencies)
    op1 = graph.add(lambda: {"data": "op1"})
    op2 = graph.add(lambda: {"data": "op2"})
    op3 = graph.add(lambda: {"data": "op3"})
    
    # Merge operation
    merge = graph.add(
        lambda results: {"merged": [results[op1], results[op2], results[op3]]},
        deps=[op1, op2, op3]
    )
    
    return graph
