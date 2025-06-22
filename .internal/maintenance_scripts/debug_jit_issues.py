#!/usr/bin/env python3
"""Debug script to investigate JIT strategy issues with ensemble patterns."""

import sys
import traceback
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, 'src')

from ember.api.operators import Operator, EnsembleOperator, Specification, EmberModel, Field
from ember.xcs import jit


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
    
    def forward(self, *, inputs: SimpleInput) -> SimpleOutput:
        print(f"  {self.model_name} called with {len(inputs.messages)} messages")
        return SimpleOutput(content=f"Response from {self.model_name}")


def debug_jit_modes():
    """Debug available JIT modes."""
    print("\n=== JIT Modes Debug ===")
    
    print("1. JIT function signature:")
    try:
        import inspect
        sig = inspect.signature(jit)
        print(f"   Signature: {sig}")
        print(f"   Parameters:")
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                print(f"     - {param_name}: {param.annotation} = {param.default}")
            else:
                print(f"     - {param_name} = {param.default}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Available modes from source:")
    try:
        from ember.xcs.jit.modes import JITMode
        print("   JITMode enum values:")
        for mode in JITMode:
            print(f"     - {mode.name}: {mode.value}")
    except Exception as e:
        print(f"   Error importing JITMode: {e}")


def debug_jit_execution():
    """Debug the full JIT execution with different strategies."""
    print("\n=== JIT Execution Debug ===")
    
    # Create ensemble
    models = [DummyModel(name=f"model_{i}") for i in range(3)]
    ensemble = EnsembleOperator(models=models)
    
    # Test input
    test_input = SimpleInput(messages=[{"role": "user", "content": "Hello"}])
    
    # Test default mode first
    print("\nDEFAULT Mode Execution:")
    print("-" * 40)
    try:
        # Create JIT wrapper with default args
        jit_ensemble = jit(ensemble)
        
        # Check what we got
        print(f"JIT wrapper type: {type(jit_ensemble)}")
        print(f"JIT wrapper class: {jit_ensemble.__class__.__name__}")
        
        # Execute
        print("\nExecuting...")
        result = jit_ensemble(inputs=test_input)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Check JIT stats if available
        if hasattr(jit, 'get_stats'):
            stats = jit.get_stats()
            print(f"\nJIT Stats: {stats}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Test each mode explicitly
    for mode in ["auto", "enhanced", "structural"]:
        print(f"\n{mode.upper()} Mode Execution:")
        print("-" * 40)
        
        try:
            # Create JIT wrapper
            jit_ensemble = jit(ensemble, mode=mode)
            
            # Execute
            print("Executing...")
            result = jit_ensemble(inputs=test_input)
            print(f"Result: {result}")
            
            # Get JIT stats if available
            try:
                from ember.xcs import get_jit_stats
                stats = get_jit_stats()
                print(f"JIT stats: {stats}")
            except:
                pass
            
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            if "add_edge" in str(e):
                print("  ^ This is the Graph.add_edge error!")
            traceback.print_exc()


def debug_simple_cases():
    """Test progressively complex cases."""
    print("\n=== Progressive Complexity Test ===")
    
    # 1. Single operator
    print("\n1. Single Operator:")
    model = DummyModel(name="single")
    test_input = SimpleInput(messages=[{"role": "user", "content": "test"}])
    
    try:
        jit_model = jit(model)
        result = jit_model(inputs=test_input)
        print(f"   Success: {result.content}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Two operators in ensemble
    print("\n2. Two-Model Ensemble:")
    models = [DummyModel(name=f"model_{i}") for i in range(2)]
    ensemble = EnsembleOperator(models=models)
    
    try:
        jit_ensemble = jit(ensemble)
        result = jit_ensemble(inputs=test_input)
        print(f"   Success: Got {len(result.responses)} responses")
    except Exception as e:
        print(f"   Error: {e}")
        
    # 3. Explicit structural mode
    print("\n3. Structural Mode on Ensemble:")
    try:
        jit_ensemble = jit(ensemble, mode="structural")
        result = jit_ensemble(inputs=test_input)
        print(f"   Success with structural mode")
    except Exception as e:
        print(f"   Error: {e}")
        if "add_edge" in str(e):
            print("   ^ Graph.add_edge error detected!")


def debug_graph_api():
    """Check the Graph API directly."""
    print("\n=== Graph API Debug ===")
    
    try:
        from ember.xcs.graph.graph import Graph
        
        print("1. Graph class inspection:")
        print(f"   Available methods: {[m for m in dir(Graph) if not m.startswith('_')]}")
        
        # Try to create and use a graph
        print("\n2. Creating Graph instance:")
        graph = Graph()
        print(f"   Instance created: {graph}")
        
        # Check for add methods
        print("\n3. Checking add methods:")
        for method in ['add_edge', 'add_node', 'add', 'insert', 'connect']:
            if hasattr(graph, method):
                print(f"   {method}: ✓ Found")
                # Get method signature
                import inspect
                sig = inspect.signature(getattr(graph, method))
                print(f"      Signature: {sig}")
            else:
                print(f"   {method}: ✗ Not found")
                
    except Exception as e:
        print(f"Error importing Graph: {e}")
        traceback.print_exc()


def debug_strategy_internals():
    """Debug strategy implementation details."""
    print("\n=== Strategy Internals Debug ===")
    
    # Check structural strategy
    print("1. Structural Strategy:")
    try:
        from ember.xcs.jit.strategies.structural import StructuralStrategy
        strategy = StructuralStrategy()
        print(f"   Strategy created: {strategy}")
        print(f"   Methods: {[m for m in dir(strategy) if not m.startswith('_')]}")
        
        # Check if it uses Graph
        import inspect
        source = inspect.getsource(StructuralStrategy)
        if "Graph" in source:
            print("   Uses Graph class: YES")
            if "add_edge" in source:
                print("   Calls add_edge: YES")
            else:
                print("   Calls add_edge: NO")
        
    except Exception as e:
        print(f"   Error: {e}")


def main():
    """Run all debug tests."""
    print("=" * 60)
    print("JIT Issues Debug Script")
    print("=" * 60)
    
    debug_jit_modes()
    debug_jit_execution()
    debug_simple_cases()
    debug_graph_api()
    debug_strategy_internals()
    
    print("\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()