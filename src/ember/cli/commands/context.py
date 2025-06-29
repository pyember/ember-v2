"""Context introspection commands for Ember CLI.

This module provides commands for viewing and debugging the current Ember
context configuration, essential for understanding what settings are active.
"""

import json

import yaml

from ember.context import context


def cmd_context_view(args) -> int:
    """Display the fully resolved configuration context.

    Shows the merged view of settings from config.yaml, environment variables,
    and internal defaults. This is invaluable for debugging configuration issues.

    Args:
        args: Parsed command line arguments containing:
            - format (str): Output format ('yaml' or 'json')
            - filter (str, optional): Dot-notation path to show specific section

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    ctx = context.get()

    # Get all configuration
    config = ctx.get_all_config()

    # Apply filter if specified
    if hasattr(args, "filter") and args.filter:
        parts = args.filter.split(".")
        filtered = config
        try:
            for part in parts:
                if isinstance(filtered, dict) and part in filtered:
                    filtered = filtered[part]
                else:
                    print(f"Path '{args.filter}' not found in configuration")
                    return 1
            config = {args.filter: filtered}
        except Exception as e:
            print(f"Error filtering configuration: {e}")
            return 1

    # Format output
    format_type = getattr(args, "format", "yaml")

    try:
        if format_type == "json":
            output = json.dumps(config, indent=2, sort_keys=True)
        else:
            output = yaml.dump(config, default_flow_style=False, sort_keys=True)

        print(output)
        return 0

    except Exception as e:
        print(f"Error formatting configuration: {e}")
        return 1


def cmd_context_validate(args) -> int:
    """Validate the current context configuration.

    Checks for common issues like missing API keys, invalid settings,
    and configuration conflicts.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code (0 if valid, 1 if issues found)
    """
    ctx = context.get()
    issues = []

    # Check for API keys
    providers = ["openai", "anthropic", "google"]
    for provider in providers:
        api_key = ctx.get_credential(provider, f"{provider.upper()}_API_KEY")
        if not api_key:
            # Only report if provider is configured
            if ctx.get_config(f"providers.{provider}"):
                issues.append(f"Missing API key for {provider}")

    # Check default model exists
    default_model = ctx.get_config("models.default")
    if default_model:
        try:
            available_models = ctx.list_models()
            if default_model not in available_models:
                issues.append(f"Default model '{default_model}' is not available")
        except Exception:
            issues.append("Unable to verify model availability")

    # Report results
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        return 1
    else:
        print("✅ Configuration is valid")
        return 0
