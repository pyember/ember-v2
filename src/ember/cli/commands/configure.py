"""Configuration management command."""

import json
import yaml
from pathlib import Path


def cmd_configure(args):
    """Manage Ember configuration.
    
    Provides get/set/list operations on configuration values.
    """
    ctx = args.context
    
    if args.action == "get":
        value = ctx.get_config(args.key, args.default)
        if value is None:
            print(f"Key '{args.key}' not found")
            return 1
        print(value)
        
    elif args.action == "set":
        # Parse value as JSON if possible
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            # Use as string if not valid JSON
            value = args.value
            
        ctx.set_config(args.key, value)
        
        # Save to persistent storage
        ctx.save()
        print(f"Set {args.key} = {value}")
        
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