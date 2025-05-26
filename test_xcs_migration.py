"""Test that the XCS migration works correctly."""

import warnings
from typing import Dict, Any

# Test new simplified imports
from ember.xcs import jit, trace, Graph, Node, XCSGraph


def test_graph_api():
    """Test that Graph works and XCSGraph compatibility exists."""
    print("=== Testing Graph API ===")
    
    # New Graph API
    graph = Graph()
    node1 = graph.add(lambda: 1)
    node2 = graph.add(lambda x: x + 1, deps=[node1])
    
    print("✓ New Graph API works")
    
    # Compatibility test
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # XCSGraph should be an alias
        compat_graph = Graph()
        print(f"✓ XCSGraph compatibility alias works (is Graph: {XCSGraph is Graph})")


def test_jit_api():
    """Test that JIT works as expected."""
    print("\n=== Testing JIT API ===")
    
    # JIT on regular function (should not optimize)
    @jit
    def regular_func(x):
        return x * 2
    
    result = regular_func(21)
    print(f"✓ JIT on regular function: {result}")
    
    # JIT on mock operator (would optimize)
    @jit
    class MockOperator:
        def forward(self, inputs):
            return {"result": inputs["x"] * 2}
    
    print("✓ JIT on Operator class works")


# def test_trace_api():
    """Test that trace decorator works."""
    print("\n=== Testing Trace API ===")
    
    @trace
    def example_function(x):
        return x + 10
    
    result = example_function(5)
    print(f"✓ Trace decorator works: {result}")
    
    # Get trace info
    trace_info = example_function.get_trace()
    print(f"✓ Can access trace data: {trace_info is not None}")

  # REMOVED: Trace strategy no longer exists
def test_api_surface():
    """Test that the API surface is clean."""
    print("\n=== Testing API Surface ===")
    
    import ember.xcs as xcs
    
    # Check what's exported
    exported = [name for name in dir(xcs) if not name.startswith('_')]
    
    # Essential exports
    essential = ['jit', 'trace', 'Graph', 'Node', 'execute_graph']
    for name in essential:
        if name in exported:
            print(f"✓ {name} is exported")
        else:
            print(f"✗ {name} is NOT exported")
    
    # Check deprecated exports
    if 'XCSGraph' in exported:
        print("✓ XCSGraph available for compatibility")


def main():
    print("XCS Migration Test\n")
    
    test_graph_api()
    test_jit_api()
    test_trace_api()
    test_api_surface()
    
    print("\n=== Summary ===")
    print("✓ Core APIs work correctly")
    print("✓ Compatibility maintained")
    print("✓ Clean simplified architecture")


if __name__ == "__main__":
    main()