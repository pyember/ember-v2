"""Logging utilities for Ember."""

import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Get logger for module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def configure_logging(verbose: bool = False) -> None:
    """Configure logging for Ember.
    
    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    # Suppress noisy libraries
    for lib in ['httpx', 'httpcore', 'urllib3', 'requests']:
        logging.getLogger(lib).setLevel(logging.WARNING)


def set_component_level(component: str, level: str) -> None:
    """Set log level for specific component.
    
    Args:
        component: Component name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    logger = logging.getLogger(component)
    logger.setLevel(getattr(logging, level.upper()))