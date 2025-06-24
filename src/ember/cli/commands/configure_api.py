"""Configuration API for external tools.

Provides a secure interface for the npm setup wizard to save configuration
through the centralized context system. Reads sensitive data from stdin
to avoid exposing secrets in process lists.
"""

import json
import sys
from typing import Dict, Any

from ember._internal.context import EmberContext


def save_api_key(provider: str, api_key: str) -> bool:
    """Save API key through credential manager.
    
    Args:
        provider: Provider name (e.g., "openai")
        api_key: API key to save
        
    Returns:
        True if successful
        
    Raises:
        ValueError: Invalid provider name
    """
    if not provider or not provider.replace('_', '').isalnum():
        raise ValueError(f"Invalid provider name: {provider}")
        
    ctx = EmberContext.current()
    ctx.credential_manager.store(provider, api_key)
    return True


def save_config(config_updates: Dict[str, Any]) -> None:
    """Save configuration through context system.
    
    Args:
        config_updates: Configuration dictionary to merge
        
    Raises:
        ValueError: Invalid configuration structure
    """
    def _apply_nested(ctx: EmberContext, updates: Dict[str, Any], prefix: str = "") -> None:
        """Recursively apply nested configuration."""
        for key, value in updates.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                _apply_nested(ctx, value, full_key)
            else:
                ctx.set_config(full_key, value)
    
    ctx = EmberContext.current()
    _apply_nested(ctx, config_updates)
    ctx.save()


def main() -> None:
    """CLI interface for configuration API.
    
    Reads sensitive data from stdin to avoid exposing in process lists.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m ember.cli.commands.configure_api <command> [args]", file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "save-key":
            if len(sys.argv) != 3:
                print("Usage: configure_api save-key <provider>", file=sys.stderr)
                print("API key should be provided via stdin", file=sys.stderr)
                sys.exit(1)
                
            provider = sys.argv[2]
            api_key = sys.stdin.read().strip()
            
            if not api_key:
                print("No API key provided", file=sys.stderr)
                sys.exit(1)
                
            save_api_key(provider, api_key)
            
        elif command == "save-config":
            config_json = sys.stdin.read().strip()
            
            if not config_json:
                print("No configuration provided", file=sys.stderr)
                sys.exit(1)
                
            config = json.loads(config_json)
            save_config(config)
            
        else:
            print(f"Unknown command: {command}", file=sys.stderr)
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()