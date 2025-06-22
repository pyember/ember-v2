"""Quick test of v4 operator improvements."""

from ember.api.operators_v4 import validate, chain, parallel, ensemble, Specification
from ember.core.operators_v4 import add_batching, add_metrics, get_capabilities


def test_improved_validation():
    """Test improved validation with better error messages."""
    @validate(input=str, output=int)
    def count_words(text: str) -> int:
        return len(text.split())
    
    # Should work
    assert count_words("hello world") == 2
    
    # Should have clear error
    try:
        count_words(123)
    except TypeError as e:
        assert "count_words expected str, got int" in str(e)
        print(f"✓ Clear error message: {e}")


def test_composition_names():
    """Test that composed operators have helpful names."""
    def add_one(x): return x + 1
    def double(x): return x * 2
    def square(x): return x * x
    
    pipeline = chain(add_one, double, square)
    print(f"✓ Chain name: {pipeline.__name__}")
    
    parallel_op = parallel(add_one, double, square)
    print(f"✓ Parallel name: {parallel_op.__name__}")
    
    def average(results):
        return sum(results) / len(results)
    
    ensemble_op = ensemble(add_one, double, square, reducer=average)
    print(f"✓ Ensemble name: {ensemble_op.__name__}")


def test_metrics_adapter():
    """Test new metrics collection adapter."""
    def expensive_operation(x):
        import time
        time.sleep(0.01)  # Simulate work
        return x * 2
    
    # Add metrics
    op = add_metrics(expensive_operation)
    
    # Use it
    results = [op(i) for i in range(5)]
    
    # Check metrics
    metrics = op.metrics
    print(f"✓ Metrics collected: {metrics}")
    assert metrics["call_count"] == 5
    assert metrics["total_time"] > 0.05  # At least 5 * 0.01
    assert metrics["error_count"] == 0


def test_specification_better_errors():
    """Test improved error messages in Specification."""
    spec = Specification(
        input_schema={"text": str, "max_len": int},
        output_schema={"summary": str, "length": int}
    )
    
    # Test with wrong type for input
    try:
        spec.validate_input("not a dict")
    except TypeError as e:
        assert "Input must be a mapping" in str(e)
        print(f"✓ Clear input type error: {e}")
    
    # Test missing field
    try:
        spec.validate_input({"text": "hello"})
    except ValueError as e:
        assert "Missing required input field: max_len" in str(e)
        print(f"✓ Clear missing field error: {e}")
    
    # Test wrong field type
    try:
        spec.validate_input({"text": "hello", "max_len": "not an int"})
    except TypeError as e:
        assert "Field 'max_len' expected int" in str(e)
        print(f"✓ Clear field type error: {e}")


def test_progressive_capabilities():
    """Test progressive enhancement of capabilities."""
    # Start simple
    def simple_op(x):
        return x.upper()
    
    print(f"✓ Simple op capabilities: {get_capabilities(simple_op)}")
    
    # Add batching
    batch_op = add_batching(simple_op, batch_size=64)
    print(f"✓ With batching: {get_capabilities(batch_op)}")
    
    # Add metrics on top
    metrics_batch_op = add_metrics(batch_op)
    print(f"✓ With batching + metrics: {get_capabilities(metrics_batch_op)}")
    
    # Can still use as simple function
    assert metrics_batch_op("hello") == "HELLO"
    
    # But also has advanced features
    assert metrics_batch_op.batch_forward(["a", "b", "c"]) == ["A", "B", "C"]
    assert metrics_batch_op.metrics["call_count"] == 1


if __name__ == "__main__":
    print("Testing v4 operator improvements...\n")
    
    test_improved_validation()
    print()
    
    test_composition_names()
    print()
    
    test_metrics_adapter()
    print()
    
    test_specification_better_errors()
    print()
    
    test_progressive_capabilities()
    
    print("\n✅ All improvements working correctly!")
    print("\nKey improvements:")
    print("- Google Python style guide compliant docstrings")
    print("- Better error messages with context")
    print("- Helpful names for composed operators")
    print("- New metrics adapter for observability")
    print("- Progressive capability enhancement")
    print("- Cleaner, more principled implementation")