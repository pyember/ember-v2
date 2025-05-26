"""
Example: Hello World - Verify Your Ember Installation
Difficulty: Basic
Time: ~1 minute
Prerequisites: None

Learning Objectives:
- Verify Ember is installed correctly
- Import basic Ember components
- Run your first Ember code

Key Concepts:
- Ember imports
- Basic operator creation (simplest form)
"""

import sys
from pathlib import Path

# Add the shared utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def main():
    """Example demonstrating the simplified XCS architecture."""
    """Verify Ember installation and print basic information."""
    print_section_header("Ember Hello World")
    
    # Test basic imports
    try:
        import ember
        print("✓ Ember package imported successfully")
        
        # Import core APIs
        from ember.api import models, operators, non, xcs, data
        print("✓ Core APIs imported successfully")
        
        # Check version
        if hasattr(ember, '__version__'):
            print_example_output("Ember version", ember.__version__)
        
        # Create the simplest possible operator
        from ember.api.operators import Operator, Specification
        
        # Minimal specification - using dict for simplicity
        class HelloSpec(Specification):
            pass  # Uses default dict input/output
        
        # Simple operator with minimal boilerplate
        class HelloOperator(Operator):
            specification = HelloSpec()
            
            def forward(self, *, inputs):
                # inputs is a dict, return a dict
                name = inputs.get("name", "World")
                return {"greeting": f"Hello, {name}! Welcome to Ember."}
        
        # Test the operator - clean kwargs style
        op = HelloOperator()
        result = op(name="World")
        
        print("\n✓ Basic operator creation successful")
        print_example_output("Test result", result["greeting"])
        
        # Real-world scenario: when you have data as a dict
        # (e.g., from JSON API, database, or another operator)
        user_data = {"name": "Ember User", "unused_field": "ignored"}
        result2 = op(inputs=user_data)
        print_example_output("Dict input", result2["greeting"])
        
        print("\n🎉 Congratulations! Ember is installed and working correctly.")
        print("\nNext steps:")
        print("  - Run first_model_call.py to make your first LLM call")
        print("  - Explore the examples in numbered order")
        
    except ImportError as e:
        print(f"\n❌ Error importing Ember: {e}")
        print("\nPlease ensure Ember is installed:")
        print("  uv pip install -e .")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())