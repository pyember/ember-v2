"""Environment Variable and Ember Setup Check

This script checks if API keys are properly set in the environment variables
and verifies that the Ember framework is correctly installed.

To run:
    uv run python src/ember/examples/basic/check_env.py
    uv run python src/ember/examples/basic/check_env.py --verbose
"""

import os
import sys
from ember.core.utils.output import (
    print_header, print_success, print_error, print_warning, print_info
)
from ember.core.utils.verbosity import create_argument_parser, setup_verbosity_from_args, vprint
from ember.core.utils.logging import suppress_logs


def check_ember_installation(verbose: bool = False):
    """Check if Ember is properly installed and can be imported."""
    print_header("Ember Installation Check", width=50)
    
    try:
        # Check basic import
        import ember
        print_success("Ember module imported successfully")
        
        # Check API imports
        from ember.api import operators, non, data, xcs
        from ember.api.models import models
        print_success("API modules imported successfully")
        
        # Check if models can be listed (suppress discovery logs)
        try:
            with suppress_logs(["ember.core.registry.model", "ember.core.registry.model.initialization"]):
                available_models = models.list()
            print_success(f"Found {len(available_models)} available models")
            
            if verbose and available_models:
                print_info("Available models:")
                for model in sorted(available_models)[:10]:
                    print(f"  • {model}")
                if len(available_models) > 10:
                    print(f"  ... and {len(available_models) - 10} more")
                    
        except Exception as e:
            print_warning(f"Could not list models: {str(e)}")
            
        return True
    except ImportError as e:
        print_error(f"Failed to import Ember", str(e))
        return False
    except Exception as e:
        print_error(f"Unexpected error", str(e))
        return False


def main():
    """Print environment variables related to API keys and check Ember setup."""
    # Set up argument parser
    parser = create_argument_parser("Check Ember environment and installation")
    args = parser.parse_args()
    setup_verbosity_from_args(args)
    
    print_header("Environment Variables Check", width=50)

    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }
    
    has_any_key = False
    for key_name, key_value in api_keys.items():
        if key_value:
            # Mask the key for security
            masked = '*' * (len(key_value) - 4) + key_value[-4:]
            print_success(f"{key_name}: {masked}")
            has_any_key = True
        else:
            print_warning(f"{key_name}: Not set")
    
    if not has_any_key:
        print_warning("\nNo API keys found. Model discovery will be limited.")
        print_info("Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY for full functionality.")
    
    # Show all environment variables in verbose mode
    if args.verbose:
        print("\nAll environment variables:")
        env_vars = {}
        for key, value in sorted(os.environ.items()):
            # Skip common/long variables
            if any(skip in key for skip in ["PATH", "SHELL", "MANPATH", "INFOPATH"]):
                continue
            
            # Mask sensitive values
            if any(sensitive in key.upper() for sensitive in ["KEY", "SECRET", "TOKEN", "PASSWORD"]):
                if value:
                    env_vars[key] = "****"
                else:
                    env_vars[key] = "Not set"
            else:
                # Truncate long values
                if len(value) > 50:
                    env_vars[key] = value[:47] + "..."
                else:
                    env_vars[key] = value
        
        # Print in columns
        for key, value in env_vars.items():
            vprint(f"  {key:30} : {value}")
    
    # Check Ember installation
    check_ember_installation(verbose=args.verbose)
    
    print_success("\nEnvironment check complete!")


if __name__ == "__main__":
    main()
