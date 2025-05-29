"""Simplified test helpers for the new XCS API."""

from ember.xcs.graph.graph import Graph, Node


def simple_operator():
    """Simple test operator that returns a fixed value."""
    return {"output": "test_result"}


def identity_operator(x):
    """Identity operator for testing."""
    return x


def math_operator(x):
    """Simple math operator for testing."""
    return x * 2


def combine_operator(*args):
    """Combine multiple inputs."""
    return {"combined": sum(args) if args else 0}


def create_test_graph() -> Graph:
    """Create a simple test graph."""
    graph = Graph()
    
    # Add some test nodes
    n1 = graph.add(simple_operator, name="node1")
    n2 = graph.add(lambda: 42, name="node2")
    n3 = graph.add(lambda x: x + 1, deps=(n2,), name="node3")
    
    return graph


def create_parallel_test_graph() -> Graph:
    """Create a graph with parallel nodes for testing."""
    graph = Graph()
    
    # Add parallel nodes
    n1 = graph.add(lambda: 1, name="parallel1")
    n2 = graph.add(lambda: 2, name="parallel2") 
    n3 = graph.add(lambda: 3, name="parallel3")
    
    # Add combining node
    combine = graph.add(
        lambda *args: sum(args),
        deps=(n1, n2, n3),
        name="combine"
    )
    
    return graph
