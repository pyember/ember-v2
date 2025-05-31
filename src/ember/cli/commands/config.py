"""Config command implementation."""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

from ember.core.utils.output import print_header, print_table, print_info, print_error, print_success
from ember.core.config.manager import ConfigManager


def register(subparsers) -> argparse.ArgumentParser:
    """Register config command."""
    parser = subparsers.add_parser(
        "config",
        help="Manage Ember configuration",
        description="View and modify Ember configuration settings"
    )
    
    subcommands = parser.add_subparsers(dest="subcommand", help="Config subcommands")
    
    # Show command
    show_parser = subcommands.add_parser("show", help="Show current configuration")
    show_parser.add_argument(
        "key",
        nargs="?",
        help="Specific configuration key to show"
    )
    show_parser.add_argument(
        "--format",
        choices=["yaml", "json", "table"],
        default="yaml",
        help="Output format"
    )
    
    # Set command
    set_parser = subcommands.add_parser("set", help="Set a configuration value")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", help="Configuration value")
    
    # Get command
    get_parser = subcommands.add_parser("get", help="Get a configuration value")
    get_parser.add_argument("key", help="Configuration key")
    
    # Path command
    path_parser = subcommands.add_parser("path", help="Show configuration file paths")
    
    # Validate command
    validate_parser = subcommands.add_parser("validate", help="Validate configuration")
    
    # Init command
    init_parser = subcommands.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration"
    )
    
    parser.set_defaults(func=execute)
    return parser


def execute(args: argparse.Namespace) -> int:
    """Execute config command."""
    if not args.subcommand:
        print_error("No subcommand specified. Use 'ember config --help' for usage.")
        return 1
    
    if args.subcommand == "show":
        return show_config(args)
    elif args.subcommand == "set":
        return set_config(args)
    elif args.subcommand == "get":
        return get_config(args)
    elif args.subcommand == "path":
        return show_paths(args)
    elif args.subcommand == "validate":
        return validate_config(args)
    elif args.subcommand == "init":
        return init_config(args)
    else:
        print_error(f"Unknown subcommand '{args.subcommand}'")
        return 1


def show_config(args: argparse.Namespace) -> int:
    """Show current configuration."""
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        
        if args.key:
            # Show specific key
            keys = args.key.split(".")
            value = config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    print_error(f"Key '{args.key}' not found in configuration")
                    return 1
            
            if args.format == "json":
                import json
                print(json.dumps({args.key: value}, indent=2))
            elif args.format == "table":
                print_table([{"Key": args.key, "Value": str(value)}])
            else:  # yaml
                import yaml
                print(yaml.dump({args.key: value}, default_flow_style=False))
        
        else:
            # Show all config
            print_header("Ember Configuration")
            
            if args.format == "json":
                import json
                print(json.dumps(config, indent=2, default=str))
            elif args.format == "table":
                # Flatten config for table display
                items = []
                
                def flatten(d: Dict[str, Any], prefix: str = "") -> None:
                    for k, v in d.items():
                        key = f"{prefix}{k}" if prefix else k
                        if isinstance(v, dict):
                            flatten(v, f"{key}.")
                        else:
                            items.append({"Key": key, "Value": str(v)})
                
                flatten(config)
                print_table(items)
            else:  # yaml
                import yaml
                print(yaml.dump(config, default_flow_style=False))
        
        return 0
        
    except Exception as e:
        print_error(f"Error showing configuration: {e}")
        return 1


def set_config(args: argparse.Namespace) -> int:
    """Set a configuration value."""
    try:
        config_manager = ConfigManager()
        
        # Parse value
        value = args.value
        if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
        
        # Set value
        keys = args.key.split(".")
        config = config_manager.config
        
        # Navigate to parent
        parent = config
        for key in keys[:-1]:
            if key not in parent:
                parent[key] = {}
            parent = parent[key]
        
        # Set the value
        parent[keys[-1]] = value
        
        # Save config
        config_path = Path.home() / ".ember" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print_success(f"Set {args.key} = {value}")
        return 0
        
    except Exception as e:
        print_error(f"Error setting configuration: {e}")
        return 1


def get_config(args: argparse.Namespace) -> int:
    """Get a configuration value."""
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Navigate to value
        keys = args.key.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                print_error(f"Key '{args.key}' not found in configuration")
                return 1
        
        print(value)
        return 0
        
    except Exception as e:
        print_error(f"Error getting configuration: {e}")
        return 1


def show_paths(args: argparse.Namespace) -> int:
    """Show configuration file paths."""
    print_header("Configuration Paths")
    
    paths = [
        {"Location": "User Config", "Path": str(Path.home() / ".ember" / "config.yaml")},
        {"Location": "Project Config", "Path": "./ember.yaml or ./.ember/config.yaml"},
        {"Location": "Environment", "Path": "$EMBER_CONFIG_PATH (if set)"}]
    
    print_table(paths)
    
    # Check which files exist
    print("\nExisting files:")
    user_config = Path.home() / ".ember" / "config.yaml"
    if user_config.exists():
        print(f"  ✓ {user_config}")
    
    project_configs = [Path("ember.yaml"), Path(".ember/config.yaml")]
    for config_path in project_configs:
        if config_path.exists():
            print(f"  ✓ {config_path}")
    
    if os.environ.get("EMBER_CONFIG_PATH"):
        env_config = Path(os.environ["EMBER_CONFIG_PATH"])
        if env_config.exists():
            print(f"  ✓ {env_config} (from EMBER_CONFIG_PATH)")
    
    return 0


def validate_config(args: argparse.Namespace) -> int:
    """Validate configuration."""
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        
        print_header("Validating Configuration")
        
        # TODO: Implement proper validation against schema
        # For now, just check that it loads successfully
        
        print_success("Configuration is valid")
        
        # Show summary
        print("\nConfiguration summary:")
        print(f"  - Log level: {config.get('logging', {}).get('level', 'INFO')}")
        print(f"  - Cache enabled: {config.get('cache', {}).get('enabled', True)}")
        
        if "providers" in config:
            print(f"  - Configured providers: {', '.join(config['providers'].keys())}")
        
        return 0
        
    except Exception as e:
        print_error(f"Configuration validation failed: {e}")
        return 1


def init_config(args: argparse.Namespace) -> int:
    """Initialize configuration file."""
    config_path = Path.home() / ".ember" / "config.yaml"
    
    if config_path.exists() and not args.force:
        print_error(f"Configuration file already exists at {config_path}")
        print("Use --force to overwrite")
        return 1
    
    try:
        # Create directory
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create default config
        default_config = """# Ember Configuration

# Logging configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # Suppress noisy loggers
  suppress:
    - httpx
    - urllib3
    - requests

# Cache configuration
cache:
  enabled: true
  directory: ~/.ember/cache
  max_size_mb: 1000

# Model provider configuration
providers:
  openai:
    # api_key: YOUR_API_KEY_HERE  # Or use OPENAI_API_KEY env var
    default_model: gpt-3.5-turbo
  
  anthropic:
    # api_key: YOUR_API_KEY_HERE  # Or use ANTHROPIC_API_KEY env var
    default_model: claude-3-sonnet-20240229

# Default model settings
defaults:
  temperature: 0.7
  max_tokens: 1000
  timeout: 60

# Development settings
development:
  debug: false
  profile: false
"""
        
        with open(config_path, "w") as f:
            f.write(default_config)
        
        print_success(f"Created configuration file at {config_path}")
        print_info("Edit this file to customize your Ember setup")
        
        return 0
        
    except Exception as e:
        print_error(f"Error creating configuration: {e}")
        return 1