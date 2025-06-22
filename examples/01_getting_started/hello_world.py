"""Hello World - Verify Ember installation and explore the simple API.

Difficulty: Basic
Time: ~1 minute

Learning Objectives:
- Verify Ember is installed correctly
- Import basic Ember components
- See how simple Ember's new API is
- Run your first Ember code
"""

import sys
from pathlib import Path

# Add the shared utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def main():
    """Verify Ember installation and print basic information."""
    print_section_header("Ember Hello World")
    
    # Test basic imports
    try:
        import ember
        print("✓ Ember package imported successfully")
        
        # Import core APIs - notice how simple this is!
        from ember.api import models, operators, data
        from ember.api.xcs import jit
        print("✓ Core APIs imported successfully")
        
        # Check version
        if hasattr(ember, '__version__'):
            print_example_output("Ember version", ember.__version__)
        
        # The new Ember way: just write functions!
        def greet(name: str = "World") -> str:
            """The simplest possible Ember function."""
            return f"Hello, {name}! Welcome to Ember."
        
        # Test it - it's just a function
        result = greet()
        print("\n✓ Basic function creation successful")
        print_example_output("Function result", result)
        
        # Make it fast with zero configuration
        fast_greet = jit(greet)
        result2 = fast_greet("Ember User")
        print_example_output("JIT-optimized result", result2)
        
        # Want an operator? Just use the decorator
        @operators.op
        def hello_operator(name: str = "World") -> dict:
            """A simple operator that returns structured data."""
            return {
                "greeting": f"Hello, {name}!",
                "timestamp": "2024-01-20",
                "version": "ember-2.0"
            }
        
        # Use it like a function
        result3 = hello_operator("Developer")
        print_example_output("Operator result", result3)
        
        print("\n🎉 Congratulations! Ember is installed and working correctly.")
        print("\nNotice how simple the new API is:")
        print("  - Functions are first-class citizens")
        print("  - No complex base classes or specifications")
        print("  - Optimization is just a decorator away")
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