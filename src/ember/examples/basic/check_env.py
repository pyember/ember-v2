"""Environment Variable and Ember Setup Check

This script checks if API keys are properly set in the environment variables
and verifies that the Ember framework is correctly installed.

To run:
    uv run python src/ember/examples/basic/check_env.py
"""

import os
import sys


def check_ember_installation():
    """Check if Ember is properly installed and can be imported."""
    print("\n=== Ember Installation Check ===\n")
    
    try:
        # Check basic import
        import ember
        print("✓ Ember module imported successfully")
        
        # Check API imports
        from ember.api import models, operators, non, data, xcs
        print("✓ API modules imported successfully")
        
        # Check if models can be listed
        try:
            available_models = models.list()
            print(f"✓ Found {len(available_models)} available models")
        except Exception as e:
            print(f"✗ Could not list models: {str(e)}")
            
        return True
    except ImportError as e:
        print(f"✗ Failed to import Ember: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False


def main():
    """Print environment variables related to API keys and check Ember setup."""
    print("\n=== Environment Variables Check ===\n")

    # Check for OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print(f"OPENAI_API_KEY: {'*' * (len(openai_key) - 4) + openai_key[-4:]}")
    else:
        print("OPENAI_API_KEY: Not set")

    # Check for Anthropic API key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        print(
            f"ANTHROPIC_API_KEY: {'*' * (len(anthropic_key) - 4) + anthropic_key[-4:]}"
        )
    else:
        print("ANTHROPIC_API_KEY: Not set")

    print("\nAll environment variables:")
    for key, value in os.environ.items():
        # Only print sensitive values with asterisks
        if "KEY" in key or "SECRET" in key or "TOKEN" in key or "PASSWORD" in key:
            masked_value = "****" if value else "Not set"
            print(f"  {key}: {masked_value}")
        elif (
            "PATH" not in key
            and "SHELL" not in key
            and "EDITOR" not in key
            and "TERM" not in key
        ):
            # Skip common environment variables like PATH, SHELL, etc.
            print(f"  {key}: {value}")
    
    # Check Ember installation
    check_ember_installation()


if __name__ == "__main__":
    main()
