#!/usr/bin/env python3
"""Simple debug script to understand the current JIT behavior."""

import sys
sys.path.insert(0, 'src')

from ember.api.operators import Operator, EnsembleOperator, Specification, EmberModel, Field
from ember.xcs import jit
from typing import List, Dict, Any


class SimpleInput(EmberModel):
    """Simple input model."""
    messages: List[Dict[str, Any]] = Field(..., description="Messages")


class SimpleOutput(EmberModel):
    """Simple output model."""
    content: str = Field(..., description="Response content")


class DummyModel(Operator[SimpleInput, SimpleOutput]):
    """Simple model for testing."""
    
    specification = Specification()
    
    def __init__(self, *, name: str):
        self.model_name = name
        # Add __name__ attribute that JIT seems to expect
        self.__name__ = name
    
    def forward(self, *, inputs: SimpleInput) -> SimpleOutput:
        print(f"  {self.model_name} called with {len(inputs.messages)} messages")
        return SimpleOutput(content=f"Response from {self.model_name}")


def test_simple_jit():
    """Test basic JIT functionality."""
    print("=== Simple JIT Test ===\n")
    
    # Test 1: Single operator
    print("1. Single Operator JIT:")
    model = DummyModel(name="single")
    
    # Apply JIT
    jit_model = jit(model)
    print(f"   Original type: {type(model)}")
    print(f"   JIT type: {type(jit_model)}")
    print(f"   Same object? {model is jit_model}")
    
    # Execute
    test_input = SimpleInput(messages=[{"role": "user", "content": "test"}])
    result = jit_model(inputs=test_input)
    print(f"   Result: {result.content}")
    
    # Test 2: Ensemble
    print("\n2. Ensemble JIT:")
    models = [DummyModel(name=f"model_{i}") for i in range(3)]
    ensemble = EnsembleOperator(models=models)
    
    # Add __name__ to ensemble too
    ensemble.__name__ = "ensemble"
    
    # Apply JIT
    jit_ensemble = jit(ensemble)
    print(f"   Original type: {type(ensemble)}")
    print(f"   JIT type: {type(jit_ensemble)}")
    
    # Execute
    try:
        result = jit_ensemble(inputs=test_input)
        print(f"   Success! Got {len(result.responses)} responses")
        for i, resp in enumerate(result.responses):
            print(f"     Response {i}: {resp}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()


def test_internal_jit():
    """Test using the internal JIT with modes."""
    print("\n=== Internal JIT Test ===\n")
    
    try:
        # Import the original JIT that accepts modes
        from ember.xcs.jit.core import jit as core_jit
        
        print("1. Core JIT available!")
        
        # Test with structural mode
        model = DummyModel(name="test")
        model.__name__ = "test"
        
        # Try different modes
        for mode in ["auto", "structural", "enhanced"]:
            print(f"\n2. Testing {mode} mode:")
            try:
                jit_model = core_jit(model, mode=mode)
                print(f"   Success! Type: {type(jit_model)}")
                
                # Execute
                test_input = SimpleInput(messages=[{"role": "user", "content": "test"}])
                result = jit_model(inputs=test_input)
                print(f"   Execution success: {result.content}")
                
            except Exception as e:
                print(f"   Error: {e}")
                if "add_edge" in str(e):
                    print("   ^ This is the Graph.add_edge issue!")
                    
    except ImportError as e:
        print(f"Could not import core JIT: {e}")


def test_graph_fix():
    """Test if we can fix the Graph issue."""
    print("\n=== Graph API Fix Test ===\n")
    
    try:
        from ember.xcs.graph.graph import Graph
        
        # Check current API
        print("1. Current Graph API:")
        g = Graph()
        print(f"   Has 'add': {hasattr(g, 'add')}")
        print(f"   Has 'add_edge': {hasattr(g, 'add_edge')}")
        
        # Try to patch it
        print("\n2. Patching Graph.add_edge:")
        def add_edge(self, from_node, to_node, **kwargs):
            """Compatibility wrapper for add_edge."""
            # The new API uses 'add' with deps
            # This is a guess at how to adapt it
            print(f"   add_edge called: {from_node} -> {to_node}")
            return self.add(to_node, deps=[from_node])
        
        # Monkey patch
        Graph.add_edge = add_edge
        print("   Patched!")
        
        # Test it
        g2 = Graph()
        print(f"   Has 'add_edge' now: {hasattr(g2, 'add_edge')}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all tests."""
    test_simple_jit()
    test_internal_jit()
    test_graph_fix()
    print("\n=== Debug Complete ===")


if __name__ == "__main__":
    main()