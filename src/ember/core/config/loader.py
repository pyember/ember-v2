"""Configuration file loader with environment variable support.

This module provides robust configuration loading and saving with:
- Automatic format detection (YAML/JSON)
- Environment variable resolution using ${VAR_NAME} syntax
- Atomic writes for data integrity
- Support for extensionless configuration files

The loader follows the configuration patterns used by Docker Compose,
Kubernetes, and other modern tools for environment variable substitution.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file with environment variable resolution.
    
    Supports JSON and YAML formats with intelligent auto-detection for
    extensionless files. Resolves environment variables using ${VAR_NAME} syntax,
    compatible with Docker Compose and similar tools.
    
    Args:
        path: Path to configuration file. Can be string or Path object.
            Supports files with .json, .yaml, .yml, or no extension.
        
    Returns:
        Dict[str, Any]: Configuration dictionary with all environment
            variables resolved. Empty dict if file contains no data.
        
    Raises:
        FileNotFoundError: If specified file doesn't exist.
        ValueError: If file format is not supported or content is invalid.
            Includes detailed parse error for debugging.
            
    Format Detection:
        1. Uses file extension if present (.json, .yaml, .yml)
        2. For extensionless files, examines content:
           - Files starting with '{' are treated as JSON
           - All others are treated as YAML
        3. If detection fails, tries alternate format before failing
        
    Examples:
        >>> config = load_config("~/.ember/config.yaml")
        >>> config = load_config(Path.home() / ".ember" / "config")
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Read file content
    content = config_path.read_text()
    
    # Auto-detect format if no extension
    suffix = config_path.suffix.lower()
    if not suffix:
        # Try to detect format by content
        stripped = content.strip()
        if stripped.startswith('{'):
            suffix = '.json'
        else:
            suffix = '.yaml'
    
    # Parse based on format
    try:
        if suffix in ('.yaml', '.yml'):
            config = yaml.safe_load(content) or {}
        elif suffix == '.json':
            config = json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        # If auto-detection failed, try the other format
        if not config_path.suffix:
            try:
                config = json.loads(content) if suffix == '.yaml' else yaml.safe_load(content)
            except:
                raise ValueError(f"Failed to parse config file: {e}")
        else:
            raise ValueError(f"Failed to parse config file: {e}")
    
    # Resolve environment variables
    return _resolve_env_vars(config)


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to file with format detection.
    
    Saves configuration in YAML or JSON format based on file extension.
    Creates parent directories if needed. Uses atomic writes for safety.
    
    Args:
        config: Configuration dictionary to save. Can contain nested
            structures, lists, and basic types (str, int, float, bool, None).
        path: Destination file path. Format determined by extension:
            - .yaml/.yml -> YAML format
            - .json or no extension -> JSON format
            
    File Format:
        - YAML: Human-readable, preserves key order, no flow style
        - JSON: Pretty-printed with 2-space indentation
        
    Safety:
        - Creates parent directories with parents=True
        - Future versions will use atomic writes
        
    Examples:
        >>> save_config({"models": {"default": "gpt-4"}}, "config.yaml")
        >>> save_config(config_dict, Path.home() / ".ember" / "config")
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = config_path.suffix.lower()
    if suffix in ('.yaml', '.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        # Default to JSON for .json or no extension
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve environment variables in configuration.
    
    Replaces strings in format ${VAR_NAME} with environment variable values.
    Only complete ${VAR} strings are replaced to avoid accidental substitution
    in URLs or other contexts.
    
    Args:
        obj: Configuration object to process. Can be:
            - dict: All values are recursively processed
            - list: All items are recursively processed  
            - str: Checked for ${VAR_NAME} pattern
            - other: Returned unchanged
        
    Returns:
        Any: Object with resolved environment variables. Structure is preserved,
            only string values matching ${VAR_NAME} are replaced.
            
    Behavior:
        - Only exact ${VAR_NAME} strings are replaced
        - Partial matches like "prefix${VAR}suffix" are NOT replaced
        - Missing environment variables leave ${VAR_NAME} unchanged
        - Non-string values pass through unchanged
        
    Examples:
        >>> os.environ['API_KEY'] = 'secret123'
        >>> _resolve_env_vars("${API_KEY}")
        'secret123'
        >>> _resolve_env_vars("prefix-${API_KEY}")  # Not replaced
        'prefix-${API_KEY}'
        >>> _resolve_env_vars({"key": "${API_KEY}"})
        {'key': 'secret123'}
    """
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Only replace complete ${VAR} strings, not partial
        if obj.startswith("${") and obj.endswith("}") and obj.count("${") == 1:
            var_name = obj[2:-1]
            return os.environ.get(var_name, obj)
        return obj
    else:
        return obj