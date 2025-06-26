"""Configuration management command.

This module implements the 'ember configure' command and its subcommands
for managing Ember's configuration. Supports CRUD operations on configuration
values, format conversion, migration, and importing from external tools.

Configuration values use dot notation for nested access:
    ember configure get models.default
    ember configure set models.timeout 30
    
The configuration is stored in YAML format at ~/.ember/config.yaml by default.
"""

import json
import sys
import yaml
from pathlib import Path


def cmd_configure(args):
    """Manage Ember configuration with various subcommands.
    
    This is the main dispatcher for configuration operations. If no action
    is specified, it launches the interactive setup wizard for a better UX.
    
    Args:
        args: Parsed command line arguments containing:
            - context: EmberContext instance
            - action (str, optional): Subcommand to execute
                - get: Retrieve a configuration value
                - set: Set a configuration value  
                - list: Display all configuration
                - show: Display a configuration section
                - migrate: Migrate old configuration files
                - import: Import from external tools
            - Additional arguments specific to each action
    
    Returns:
        int: Exit code (0 for success, 1 for error)
        
    Subcommand Details:
        get: Retrieves value by dot-notation key with optional default
        set: Sets value (auto-detects JSON), warns for sensitive keys
        list: Shows full config in YAML/JSON format
        show: Shows specific section in YAML/JSON format
        migrate: Runs configuration migration from old formats
        import: Imports configuration from external tools
        
    Security Notes:
        - Warns when setting keys containing sensitive patterns
        - Validates all inputs before processing
        - Atomic saves to prevent corruption
    """
    ctx = args.context
    
    # If no action specified, run the setup wizard
    if not hasattr(args, 'action') or args.action is None:
        # Import here to avoid circular dependencies
        from ember.cli.main import cmd_setup
        return cmd_setup(args)
    
    if args.action == "get":
        # Validate key
        if not args.key or not isinstance(args.key, str):
            print("Error: Configuration key is required", file=sys.stderr)
            return 1
            
        value = ctx.get_config(args.key, args.default)
        if value is None and args.default is None:
            print(f"Key '{args.key}' not found", file=sys.stderr)
            return 1
        print(value)
        return 0
        
    elif args.action == "set":
        # Validate inputs
        if not args.key or not isinstance(args.key, str):
            print("Error: Configuration key is required", file=sys.stderr)
            return 1
            
        if args.value is None:
            print("Error: Value is required for set operation", file=sys.stderr)
            return 1
        
        # Parse value as JSON if possible
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            # Use as string if not valid JSON
            value = args.value
        
        # Validate sensitive keys
        sensitive_patterns = ["api_key", "secret", "password", "token"]
        key_lower = args.key.lower()
        if any(pattern in key_lower for pattern in sensitive_patterns):
            print(f"Warning: Setting potentially sensitive configuration: {args.key}", file=sys.stderr)
            
        try:
            ctx.set_config(args.key, value)
            
            # Save to persistent storage
            ctx.save()
            print(f"Set {args.key} = {value}")
        except (ValueError, TypeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
    elif args.action == "list":
        # Show all configuration
        config = ctx.get_all_config()
        if args.format == "json":
            print(json.dumps(config, indent=2))
        else:
            # YAML format (default)
            print(yaml.dump(config, default_flow_style=False))
            
    elif args.action == "show":
        # Show specific section
        if args.section:
            config = ctx.get_config(args.section, {})
        else:
            config = ctx.get_all_config()
            
        if args.format == "json":
            print(json.dumps(config, indent=2))
        else:
            print(yaml.dump(config, default_flow_style=False))
            
    elif args.action == "migrate":
        # Run migration manually
        from ember._internal.migrations import main as migrate_main
        migrate_main()
        
    elif args.action == "import":
        # Delegate to import command
        from ember.cli.commands.config_import import cmd_import
        return cmd_import(args)
            
    return 0