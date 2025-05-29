"""Test how operators integrate with the simplified graph system."""

import time
from typing import Dict, Any
from ember.core.registry.operator import Operator
from ember.core.registry.specification import Specification
from ember.xcs import jit
from ember.xcs.graph import Graph
import pytest


# Simple specifications for testing
class SimpleSpec(Specification):
    class InputModel:
        value: float
    
    class OutputModel:
        result: float


class EnsembleSpec(Specification):
    class InputModel:
        data: Dict[str, Any]
    
    class OutputModel:
        prediction: float
        confidence: float


# Test operators
class TransformOperator(Operator):
    """Simple transformation operator."""
    specification = SimpleSpec
    
    def __init__(self, multiplier: float):
        super().__init__()
        self.multiplier = multiplier
    
    def forward(self, *, inputs):
        return {"result": inputs["value"] * self.multiplier}


class ModelOperator(Operator):
    """Simulates a model with computation time."""
    specification = EnsembleSpec
    
    def __init__(self, model_id: str, weight: float = 1.0):
        super().__init__()
        self.model_id = model_id
        self.weight = weight
    
    def forward(self, *, inputs):
        # Simulate computation
        time.sleep(0.01)
        value = inputs["data"].get("value", 1.0)
        return {
            "prediction": value * self.weight,
            "confidence": 0.9
        }


class JudgeOperator(Operator):
    """Combines predictions from multiple models."""
    specification = EnsembleSpec
    
    def forward(self, *, inputs):
        predictions = inputs["data"]["predictions"]
        avg_prediction = sum(p["prediction"] for p in predictions) / len(predictions)
        avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)
        
        return {
            "prediction": avg_prediction,
            "confidence": avg_confidence
        }


@jit
class EnsembleWithJudge(Operator):
    """Ensemble operator that should parallelize automatically."""
    specification = EnsembleSpec
    
    def __init__(self):
        super().__init__()
        # These are EmberModule attributes
        self.model1 = ModelOperator("model1", 1.0)
        self.model2 = ModelOperator("model2", 1.1)
        self.model3 = ModelOperator("model3", 0.9)
        self.judge = JudgeOperator()
    
    def forward(self, *, inputs):
        # These should run in parallel automatically!
        r1 = self.model1(inputs=inputs)
        r2 = self.model2(inputs=inputs)
        r3 = self.model3(inputs=inputs)
        
        # Judge waits for all
        return self.judge(inputs={
            "data": {"predictions": [r1, r2, r3]}
        })


def test_operator_pytree_protocol():
    """Test that operators properly implement pytree protocol."""
    
    # Create an operator
    op = TransformOperator(multiplier=2.0)
    
    # Test flattening
    values, treedef = op.__pytree_flatten__()
    
    # Should separate dynamic and static fields
    assert isinstance(values, list)
    assert isinstance(treedef, tuple)
    
    # Test unflattening
    reconstructed = TransformOperator.__pytree_unflatten__(treedef, values)
    
    # Should preserve structure
    assert isinstance(reconstructed, TransformOperator)
    assert reconstructed.multiplier == 2.0


def test_operator_graph_integration():
    """Test that operators work with the graph system."""
    
    # Create a graph manually (simulating what JIT does)
    graph = Graph()
    
    # Add operator nodes
    transform1 = TransformOperator(2.0)
    transform2 = TransformOperator(3.0)
    
    # Source data
    data_node = graph.add(lambda: {"value": 10.0})
    
    # Transform nodes - these can run in parallel!
    t1_node = graph.add(
        lambda x: transform1(inputs=x),
        deps=[data_node]
    )
    t2_node = graph.add(
        lambda x: transform2(inputs=x),
        deps=[data_node]
    )
    
    # Combine results
    combine_node = graph.add(
        lambda r1, r2: {"sum": r1["result"] + r2["result"]},
        deps=[t1_node, t2_node]
    )
    
    # Check wave analysis
    stats = graph.stats
    assert stats['waves'] == 3  # data -> transforms -> combine
    assert stats['parallelism'] == 2  # Two transforms in parallel
    
    # Execute
    results = graph.run()
    assert results[combine_node] == {"sum": 20.0 + 30.0}


def test_ensemble_automatic_parallelization():
    """Test that ensemble operators automatically parallelize."""
    
    ensemble = EnsembleWithJudge()
    
    # Time sequential execution (simulated)
    start = time.perf_counter()
    # If it were sequential: 3 models * 10ms = 30ms minimum
    
    # Execute (should be parallel)
    result = ensemble(inputs={"data": {"value": 10.0}})
    
    elapsed = time.perf_counter() - start
    
    # Should be much faster than sequential
    assert elapsed < 0.02  # Should take ~10ms, not 30ms
    
    # Check result
    assert "prediction" in result
    assert "confidence" in result
    assert result["confidence"] == 0.9


def test_mixed_function_operator_graph():
    """Test mixing simple functions with operators in a graph."""
    
    @jit
    def hybrid_pipeline(data):
        # Simple function
        normalized = lambda x: {k: v/100 for k, v in x.items()}
        norm_data = normalized(data)
        
        # Operator
        transform = TransformOperator(5.0)
        transformed = transform(inputs={"value": norm_data["value"]})
        
        # Another simple function  
        final = lambda x: x["result"] ** 2
        return final(transformed)
    
    # Execute
    result = hybrid_pipeline({"value": 100.0})
    
    # Should compute: (100/100 * 5.0) ** 2 = 25.0
    assert result == 25.0


def test_operator_metadata_preservation():
    """Test that operator metadata is preserved through transformations."""
    
    op = ModelOperator("test_model", 2.5)
    
    # Check that static fields are preserved
    values, (keys, static_fields) = op.__pytree_flatten__()
    
    # model_id and weight should be preserved
    assert "model_id" in str(static_fields) or any("test_model" in str(v) for v in values)
    
    # Reconstruct
    reconstructed = ModelOperator.__pytree_unflatten__((keys, static_fields), values)
    assert reconstructed.model_id == "test_model"
    assert reconstructed.weight == 2.5


@pytest.mark.skipif(True, reason="Demonstrates concept only")
def test_vmap_over_operators():
    """Test that vmap works with operators (conceptual)."""
    
    # This would work if vmap is implemented for operators
    from ember.xcs import vmap
    
    # Create a transform operator
    transform = TransformOperator(2.0)
    
    # Vectorize it
    vtransform = vmap(transform)
    
    # Apply to batch
    batch_inputs = [{"value": i} for i in range(5)]
    results = vtransform(batch_inputs)
    
    # Should apply transform to each input
    expected = [{"result": i * 2.0} for i in range(5)]
    assert results == expected


if __name__ == "__main__":
    print("Testing Operator-Graph Integration...\n")
    
    test_operator_pytree_protocol()
    print("✓ Operators implement pytree protocol")
    
    test_operator_graph_integration()
    print("✓ Operators work with graph system")
    
    test_ensemble_automatic_parallelization()
    print("✓ Ensemble operators parallelize automatically")
    
    test_mixed_function_operator_graph()
    print("✓ Functions and operators mix seamlessly")
    
    test_operator_metadata_preservation()
    print("✓ Operator metadata preserved through transformations")
    
    print("\nAll tests passed!")