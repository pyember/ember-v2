"""Configuration management command."""

import json
import sys
import yaml
from pathlib import Path


def cmd_configure(args):
    """Manage Ember configuration.
    
    Provides get/set/list operations on configuration values.
    """
    ctx = args.context
    
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
            
    return 0