"""Tests for the simplified execution engine."""

from typing import Any, Dict

from ember.xcs.graph import Graph
from ember.xcs.graph.graph import Graph


class TestOperator:
    """Test operator for engine tests."""

    def __init__(self, name: str, multiplier: int = 1) -> None:
        self.name = name
        self.multiplier = multiplier
        self.calls = 0

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        return {"value": inputs.get("value", 1) * self.multiplier}


def create_test_graph() -> Graph:
    """Create a simple test graph for engine tests."""
    graph = Graph()

    # Add nodes
    double_op = TestOperator("double", 2)
    triple_op = TestOperator("triple", 3)

    node1 = graph.add(double_op, name="double")
    node2 = graph.add(triple_op, deps=[node1], name="triple")

    return graph


def test_execute_graph():
    """Test basic graph execution."""
    graph = create_test_graph()

    # Execute the graph
    results = graph.run({"value": 5})

    # Find nodes by examining values in the results
    double_result = None
    triple_result = None

    for node_id, node_result in results.items():
        if node_result.get("value") == 10:
            double_result = node_result
        elif node_result.get("value") == 30:
            triple_result = node_result

    # Verify results
    assert double_result is not None, "Double operator result not found"
    assert triple_result is not None, "Triple operator result not found"
    assert double_result["value"] == 10  # 5 * 2
    assert triple_result["value"] == 30  # 10 * 3


def test_execution_options():
    """Test simplified execution - no more ExecutionOptions."""
    graph = create_test_graph()
    
    # Execute with simplified API - just parallel parameter
    results = graph.run({"value": 5}, parallel=False)
    
    # Verify results
    assert any(r.get("value") == 10 for r in results.values())
    assert any(r.get("value") == 30 for r in results.values())


def test_parallel_execution():
    """Test parallel vs sequential execution."""
    graph = create_test_graph()

    # Execute sequentially
    results_seq = graph.run({"value": 5}, parallel=False)
    
    # Execute in parallel (default)
    results_par = graph.run({"value": 5}, parallel=True)
    
    # Results should be the same
    seq_values = sorted([r.get("value") for r in results_seq.values()])
    par_values = sorted([r.get("value") for r in results_par.values()])
    
    assert seq_values == par_values
    assert 10 in seq_values  # 5 * 2
    assert 30 in seq_values  # 10 * 3


def test_simplified_api():
    """Test that the simplified API is actually simpler."""
    # Old way would require:
    # - ExecutionOptions object
    # - Scheduler selection
    # - Complex graph building
    
    # New way:
    graph = Graph()
    n1 = graph.add(lambda inputs: {"x": inputs["x"] * 2})
    n2 = graph.add(lambda inputs: {"x": inputs["x"] + 10}, deps=[n1])
    
    results = graph.run({"x": 5})
    
    # Should compute (5 * 2) + 10 = 20
    final_result = results[n2]
    assert final_result["x"] == 20