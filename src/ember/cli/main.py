"""Ember CLI main entry point."""

import sys
import argparse
import subprocess
import shutil
from pathlib import Path

from ember._internal.context import EmberContext
from ember.cli.commands.configure import cmd_configure


def cmd_setup(args):
    """Run interactive setup wizard."""
    ctx = args.context
    
    # Check if npx is available
    if not shutil.which('npx'):
        print("Error: npm/npx is required for the setup wizard.")
        print("Install Node.js from https://nodejs.org/")
        return 1
        
    # Pass context info to setup wizard via environment
    import os
    env = os.environ.copy()
    env['EMBER_CONFIG_PATH'] = str(EmberContext.get_config_path())
    
    # Launch the setup wizard
    try:
        result = subprocess.run(['npx', '-y', '@ember-ai/setup'], env=env)
        
        # Reload context after setup completes
        if result.returncode == 0:
            # The setup wizard should have saved credentials
            # Reload to pick up any changes
            ctx.reload()
            
        return result.returncode
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        return 1
    except Exception as e:
        print(f"Error launching setup: {e}")
        return 1


def cmd_version(args):
    """Show version."""
    try:
        import ember
        print(f"Ember {ember.__version__}")
    except AttributeError:
        print("Ember (version unknown)")
    return 0


def cmd_models(args):
    """List models or providers."""
    # Still use the models API for discovery since it has the catalog
    from ember.api import models
    
    if args.providers:
        # Show providers
        providers = models.providers()
        print("Available providers:")
        for provider in providers:
            print(f"  - {provider}")
    else:
        # Show models
        if args.provider:
            # Filter by provider
            info = models.discover(args.provider)
        else:
            info = models.discover()
            
        print("Available models:")
        for model_id, details in sorted(info.items()):
            print(f"  {model_id:<20} {details['description']}")
    
    return 0


def cmd_test(args):
    """Test model connection."""
    ctx = args.context
    
    # Get default model from context if not specified
    model = args.model or ctx.get_config("models.default", "gpt-3.5-turbo")
    
    try:
        print(f"Testing connection with {model}...")
        # Use context's model registry
        response = ctx.model_registry.invoke_model(model, "Say hello!")
        print(f"✓ Success! Response: {response.data}")
        return 0
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 1


def main():
    """CLI entry point."""
    # Initialize context early for all commands
    ctx = EmberContext.current()
    
    parser = argparse.ArgumentParser(
        prog='ember',
        description='Ember AI - Build AI systems with elegance'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Run interactive setup wizard')
    setup_parser.set_defaults(func=cmd_setup)
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version')
    version_parser.set_defaults(func=cmd_version)
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')
    models_parser.add_argument('--provider', help='Filter by provider')
    models_parser.add_argument('--providers', action='store_true', help='List providers instead')
    models_parser.set_defaults(func=cmd_models)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test API connection')
    test_parser.add_argument('--model', help='Model to test with')
    test_parser.set_defaults(func=cmd_test)
    
    # Configure command
    config_parser = subparsers.add_parser('configure', help='Manage configuration')
    config_subparsers = config_parser.add_subparsers(dest='action', help='Action to perform')
    
    # configure get
    get_parser = config_subparsers.add_parser('get', help='Get configuration value')
    get_parser.add_argument('key', help='Configuration key (dot notation)')
    get_parser.add_argument('--default', help='Default value if not found')
    
    # configure set
    set_parser = config_subparsers.add_parser('set', help='Set configuration value')
    set_parser.add_argument('key', help='Configuration key (dot notation)')
    set_parser.add_argument('value', help='Value to set (JSON or string)')
    
    # configure list
    list_parser = config_subparsers.add_parser('list', help='List all configuration')
    list_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml', 
                           help='Output format')
    
    # configure show
    show_parser = config_subparsers.add_parser('show', help='Show configuration section')
    show_parser.add_argument('section', nargs='?', help='Section to show')
    show_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                           help='Output format')
    
    # configure migrate
    migrate_parser = config_subparsers.add_parser('migrate', help='Migrate old configuration files')
    
    config_parser.set_defaults(func=cmd_configure)
    
    # Set default to help
    parser.set_defaults(func=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command specified
    if args.func is None:
        parser.print_help()
        return 2  # Standard exit code for incorrect usage
        
    # Pass context to commands that need it
    args.context = ctx
    
    # Run command and return its exit code
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return 130  # Standard exit code for SIGINT
    except SystemExit as e:
        # Pass through SystemExit with its code
        return e.code if e.code is not None else 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())