"""Import configuration from external tools.

This module provides functionality to import configuration from external
AI tools (OpenAI CLI, Anthropic CLI, etc.) into Ember's format. It handles
format detection, adaptation, validation, and safe migration with backups.

The import process:
1. Detects or uses specified external configuration file
2. Adapts external format to Ember's schema
3. Validates the migrated configuration
4. Creates backup of existing config (optional)
5. Saves the migrated configuration

Supported formats: YAML, JSON
Supported tools: Any tool using compatible provider configuration
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ember.core.config.compatibility_adapter import CompatibilityAdapter
from ember.core.config.loader import load_config, save_config
from ember._internal.context import EmberContext


def cmd_import(args):
    """Import configuration from external AI tools.
    
    This command facilitates migration from external tools to Ember by importing
    existing configuration files. It automatically detects common configuration
    locations, adapts the format, and preserves all provider settings.
    
    Args:
        args: Parsed command line arguments containing:
            - config_path (Path, optional): Source configuration file path
            - output_path (Path, optional): Target Ember config path
            - backup (bool): Whether to backup existing config (default: True)
            - dry_run (bool): Preview changes without saving (default: False)
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    
    Examples:
        # Auto-detect and import external configuration
        ember configure import
        
        # Import from specific file
        ember configure import --config-path ~/my-config.json
        
        # Preview migration without making changes
        ember configure import --dry-run
        
        # Import without backing up existing config
        ember configure import --no-backup
        
    Notes:
        - Searches common locations: ~/.config/openai, ~/.openai, etc.
        - Preserves all provider API keys and settings
        - Creates timestamped backup unless --no-backup specified
        - Shows required environment variables after import
    """
    config_path = args.path if hasattr(args, 'path') else getattr(args, 'codex_config_path', None)
    output_path = args.output_path
    backup = args.backup
    dry_run = args.dry_run
    
    # Find config file
    if not config_path:
        config_path = _find_external_config()
        if not config_path:
            print("Error: No external configuration found. Please specify --path", file=sys.stderr)
            return 1
    
    # Determine output path
    if not output_path:
        output_path = EmberContext.get_config_path()
    
    print(f"Importing config from: {config_path}")
    print(f"Target Ember config: {output_path}")
    
    try:
        # Load external configuration
        external_config = load_config(config_path)
        
        # Check if it needs adaptation
        if CompatibilityAdapter.needs_adaptation(external_config):
            print("Detected external configuration format. Adapting...")
        
        # Perform migration
        migrated_config = _migrate_config(external_config)
        
        if dry_run:
            print("\nMigrated configuration (dry run):")
            print(yaml.dump(migrated_config, default_flow_style=False))
            return 0
        
        # Backup existing config if requested
        if backup and output_path.exists():
            backup_path = _backup_config(output_path)
            print(f"Backed up existing config to: {backup_path}")
        
        # Save migrated configuration
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(migrated_config, output_path)
        
        print(f"\n✓ Successfully imported configuration to {output_path}")
        
        # Show summary of imported providers
        if "providers" in migrated_config:
            print("\nImported providers:")
            for name, provider in migrated_config["providers"].items():
                env_key = provider.get("envKey", provider.get("env_key", ""))
                print(f"  - {name}: Uses ${env_key}")
        
        # Remind about environment variables
        print("\nRemember to set your environment variables:")
        _show_required_env_vars(migrated_config)
        
        return 0
        
    except Exception as e:
        print(f"Error importing configuration: {e}", file=sys.stderr)
        return 1


def _find_external_config() -> Optional[Path]:
    """Find external tool configuration file.
    
    Searches common configuration directories for external AI tool configs.
    Checks multiple locations and formats in priority order.
    
    Returns:
        Optional[Path]: Path to found configuration file, None if not found
        
    Search order:
        1. ~/.config/openai/config.{yaml,yml,json}
        2. ~/.openai/config.{yaml,yml,json}
        3. ~/.config/anthropic/config.{yaml,yml,json}
        4. ~/.anthropic/config.{yaml,yml,json}
    """
    # Check common config directories
    config_dirs = [
        Path.home() / ".config" / "openai",
        Path.home() / ".openai",
        Path.home() / ".config" / "anthropic",
        Path.home() / ".anthropic",
    ]
    
    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        
        # Check for config.yaml first, then config.json
        for filename in ["config.yaml", "config.yml", "config.json"]:
            config_path = config_dir / filename
            if config_path.exists():
                return config_path
    
    return None


def _migrate_config(external_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate external configuration to Ember format.
    
    Transforms external tool configuration to Ember's schema while preserving
    all settings. Adds version information and migration metadata.
    
    Args:
        external_config: Configuration dict from external tool
        
    Returns:
        Dict[str, Any]: Migrated configuration in Ember format
        
    Migration steps:
        1. Adapt each provider to Ember's schema
        2. Add version field if missing
        3. Store migration metadata and original fields
        4. Preserve tool-specific settings in _migration section
    """
    migrated = external_config.copy()
    
    # Migrate all providers
    if "providers" in migrated:
        for name, provider in migrated["providers"].items():
            if isinstance(provider, dict):
                migrated["providers"][name] = CompatibilityAdapter.migrate_provider(provider)
    
    # Add version if not present
    if "version" not in migrated:
        migrated["version"] = "1.0"
    
    # Add migration metadata
    migrated["_migration"] = {
        "from": "external",
        "date": datetime.now().isoformat(),
        "original_fields": {}
    }
    
    # Preserve external tool-specific fields
    for field in CompatibilityAdapter.EXTERNAL_FIELDS:
        if field in external_config:
            migrated["_migration"]["original_fields"][field] = external_config[field]
    
    return migrated


def _backup_config(config_path: Path) -> Path:
    """Create timestamped backup of existing configuration.
    
    Args:
        config_path: Path to configuration file to backup
        
    Returns:
        Path: Path to created backup file
        
    Example:
        config.yaml -> config.20240115_143052.backup.yaml
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(f".{timestamp}.backup{config_path.suffix}")
    shutil.copy2(config_path, backup_path)
    return backup_path


def _show_required_env_vars(config: Dict[str, Any]) -> None:
    """Display required environment variables for imported configuration.
    
    Args:
        config: Migrated configuration dictionary
        
    Note:
        Extracts unique environment variable names from all providers
        and displays export commands for easy shell setup.
    """
    if "providers" not in config:
        return
    
    env_vars = set()
    for provider in config["providers"].values():
        if isinstance(provider, dict):
            if "envKey" in provider:
                env_vars.add(provider["envKey"])
            elif "env_key" in provider:
                env_vars.add(provider["env_key"])
    
    for var in sorted(env_vars):
        print(f"  export {var}='your-api-key-here'")