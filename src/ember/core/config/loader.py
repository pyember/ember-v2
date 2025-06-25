"""Configuration file loader."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from file.
    
    Supports JSON and YAML formats based on extension.
    
    Args:
        path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If format not supported.
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    suffix = config_path.suffix.lower()
    
    with open(config_path) as f:
        if suffix in ('.yaml', '.yml'):
            return yaml.safe_load(f) or {}
        elif suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")