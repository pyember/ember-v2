"""Ember CLI main entry point.

This module provides the command-line interface for Ember AI, supporting
configuration management, model discovery, API testing, and interactive setup.

The CLI follows a subcommand pattern similar to git, with commands like:
    ember setup
    ember configure get <key>
    ember models --provider openai
    ember test --model gpt-4

Exit codes follow standard Unix conventions:
    0: Success
    1: General error
    2: Incorrect usage
    130: Interrupted (SIGINT)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ember._internal.context import EmberContext
from ember.cli.commands.configure import cmd_configure
from ember.cli.commands.context import cmd_context_validate, cmd_context_view
from ember.cli.commands.registry import (
    cmd_registry_info,
    cmd_registry_list_models,
    cmd_registry_list_providers,
)


def cmd_setup(args):
    """Run interactive setup wizard for provider configuration.

    Launches an interactive terminal UI to configure API keys for AI providers.
    Attempts to run a local TypeScript/React setup wizard if available,
    falling back to an npm package if not found.

    Args:
        args: Parsed command line arguments containing:
            - context: EmberContext instance

    Returns:
        int: Exit code (0 for success, 1 for error)

    Raises:
        Never - all exceptions are caught and converted to exit codes
    """
    ctx = args.context

    # Check if npx is available
    if not shutil.which("npx"):
        print("Error: npm/npx is required for the setup wizard.")
        print("Install Node.js from https://nodejs.org/")
        return 1

    # Pass context info to setup wizard via environment
    import os

    env = os.environ.copy()
    env["EMBER_CONFIG_PATH"] = str(EmberContext.get_config_path())

    # Launch the setup wizard
    try:
        # Run the local setup wizard
        setup_wizard_path = Path(__file__).parent / "setup-wizard"
        if setup_wizard_path.exists():
            # Build and run the local setup wizard
            build_result = subprocess.run(
                ["npm", "run", "build"], cwd=setup_wizard_path, capture_output=True
            )
            if build_result.returncode == 0:
                result = subprocess.run(["npm", "run", "start"], cwd=setup_wizard_path, env=env)
            else:
                print("Error building setup wizard")
                result = build_result
        else:
            # Fallback to npx (for future npm package)
            result = subprocess.run(["npx", "-y", "@ember-ai/setup"], env=env)

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
    """Display Ember version information.

    Args:
        args: Parsed command line arguments (unused)

    Returns:
        int: Always returns 0 (success)
    """
    try:
        import ember

        print(f"Ember {ember.__version__}")
    except AttributeError:
        print("Ember (version unknown)")
    return 0


def cmd_models(args):
    """List available AI models and providers.

    Displays either a list of available providers or models based on arguments.
    Models can be filtered by provider. Uses the models API catalog for discovery.

    Args:
        args: Parsed command line arguments containing:
            - providers (bool): If True, list providers instead of models
            - provider (str, optional): Filter models by specific provider

    Returns:
        int: Always returns 0 (success)

    Examples:
        ember models                    # List all models
        ember models --providers        # List all providers
        ember models --provider openai  # List OpenAI models only
    """
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
    """Test AI model API connection.

    Sends a simple test message to verify API connectivity and credentials.
    Uses the default model from configuration if not specified.

    Args:
        args: Parsed command line arguments containing:
            - context: EmberContext instance
            - model (str, optional): Specific model to test with

    Returns:
        int: Exit code (0 for success, 1 for failure)

    Note:
        The test sends "Say hello!" as a prompt and displays the response.
        This validates both API credentials and model availability.
    """
    ctx = args.context

    # Get default model from context if not specified
    model = args.model or ctx.get_config("models.default", "gpt-3.5-turbo")

    try:
        print(f"Testing connection with {model}...")
        # Get model instance to check if it exists and has API key
        model_instance = ctx.model_registry.get_model(model)
        
        # Now invoke it
        response = ctx.model_registry.invoke_model(model, "Say hello!")
        print(f"✓ Success! Response: {response.data}")
        return 0
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 1


def main():
    """Main CLI entry point.

    Initializes the Ember context, parses command line arguments, and
    dispatches to appropriate command handlers. Handles all exceptions
    and converts them to appropriate exit codes.

    Returns:
        int: Exit code for the shell
            - 0: Success
            - 1: General error
            - 2: Incorrect usage (shows help)
            - 130: Interrupted by user (Ctrl+C)

    Architecture Notes:
        - Context is initialized early and shared with all commands
        - Commands are organized as subcommands with dedicated parsers
        - All exceptions are caught and converted to exit codes
        - SystemExit is handled specially to preserve its exit code
    """
    # Initialize context early for all commands
    ctx = EmberContext.current()

    parser = argparse.ArgumentParser(
        prog="ember", description="Ember - Build AI systems with elegance"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Run interactive setup wizard")
    setup_parser.set_defaults(func=cmd_setup)

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)

    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument("--provider", help="Filter by provider")
    models_parser.add_argument("--providers", action="store_true", help="List providers instead")
    models_parser.set_defaults(func=cmd_models)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test API connection")
    test_parser.add_argument("--model", help="Model to test with")
    test_parser.set_defaults(func=cmd_test)

    # Configure command
    config_parser = subparsers.add_parser("configure", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="action", help="Action to perform")

    # configure get
    get_parser = config_subparsers.add_parser("get", help="Get configuration value")
    get_parser.add_argument("key", help="Configuration key (dot notation)")
    get_parser.add_argument("--default", help="Default value if not found")

    # configure set
    set_parser = config_subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Configuration key (dot notation)")
    set_parser.add_argument("value", help="Value to set (JSON or string)")

    # configure list
    list_parser = config_subparsers.add_parser("list", help="List all configuration")
    list_parser.add_argument(
        "--format", choices=["yaml", "json"], default="yaml", help="Output format"
    )

    # configure show
    show_parser = config_subparsers.add_parser("show", help="Show configuration section")
    show_parser.add_argument("section", nargs="?", help="Section to show")
    show_parser.add_argument(
        "--format", choices=["yaml", "json"], default="yaml", help="Output format"
    )

    # configure migrate
    config_subparsers.add_parser("migrate", help="Migrate old configuration files")

    # configure import
    import_parser = config_subparsers.add_parser(
        "import", help="Import configuration from external tools"
    )
    import_parser.add_argument(
        "--config-path", type=Path, help="Path to external config file to import"
    )
    import_parser.add_argument(
        "--output-path",
        type=Path,
        help="Output path for Ember config (default: ~/.ember/config.yaml)",
    )
    import_parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        default=True,
        help="Skip backup of existing config before importing",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without making changes",
    )

    config_parser.set_defaults(func=cmd_configure)

    # Context command
    context_parser = subparsers.add_parser("context", help="Inspect current context")
    context_subparsers = context_parser.add_subparsers(dest="action", help="Context actions")

    # context view
    view_parser = context_subparsers.add_parser("view", help="View current configuration")
    view_parser.add_argument(
        "--format", choices=["yaml", "json"], default="yaml", help="Output format"
    )
    view_parser.add_argument("--filter", help="Filter to specific path (dot notation)")
    view_parser.set_defaults(func=cmd_context_view)

    # context validate
    validate_parser = context_subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.set_defaults(func=cmd_context_validate)

    # Registry command
    registry_parser = subparsers.add_parser("registry", help="Inspect model and data registries")
    registry_subparsers = registry_parser.add_subparsers(dest="action", help="Registry actions")

    # registry list-models
    list_models_parser = registry_subparsers.add_parser("list-models", help="List available models")
    list_models_parser.add_argument("--provider", help="Filter by provider")
    list_models_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )
    list_models_parser.set_defaults(func=cmd_registry_list_models)

    # registry list-providers
    list_providers_parser = registry_subparsers.add_parser(
        "list-providers", help="List configured providers"
    )
    list_providers_parser.set_defaults(func=cmd_registry_list_providers)

    # registry info
    info_parser = registry_subparsers.add_parser("info", help="Show model details")
    info_parser.add_argument("model_id", help="Model identifier")
    info_parser.set_defaults(func=cmd_registry_info)

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


if __name__ == "__main__":
    sys.exit(main())
