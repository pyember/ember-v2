"""
Logging Configuration Module for Ember

This module centralizes logging configuration for the Ember framework,
providing consistent logging behavior across all components while allowing
for appropriate verbosity control.

The design follows standard Python logging best practices:
- Libraries (like Ember) should configure loggers but not handlers
- Applications control how logs are displayed/stored
- Sensible defaults are provided but easily overridable

Usage:
    # To apply standard configuration with reduced verbosity:
    from ember.core.utils.logging import configure_logging
    configure_logging(verbose=False)

    # To adjust specific component verbosity:
    from ember.core.utils.logging import set_component_level
    set_component_level("model_discovery", logging.DEBUG)
"""

import logging
import os

# Component groups allow configuring related loggers together
COMPONENT_GROUPS = {
    "model_registry": [
        "ember.core.registry.model",
        "ember.core.registry.model.initialization",
        "ember.core.registry.model.base.registry.discovery",
        "ember.core.registry.model.base.registry.model_registry",
    ],
    "model_warnings": [
        # These loggers produce too many warnings during discovery
        "ember.core.registry.model.base.registry.discovery",
    ],
    "model_discovery": [
        "ember.core.registry.model.providers.anthropic.anthropic_discovery",
        "ember.core.registry.model.providers.openai.openai_discovery",
        "ember.core.registry.model.providers.deepmind.deepmind_discovery",
    ],
    "http": [
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpx",
        "urllib3",
        "urllib3.connectionpool",
        "urllib3.util.retry",
        "openai",
        "openai._base_client",
        "openai._http_client",
        "anthropic",
        "anthropic._base_client",
    ],
}

# Reverse mapping to find group by logger name
LOGGER_TO_GROUP = {}
for group_name, loggers in COMPONENT_GROUPS.items():
    for logger_name in loggers:
        LOGGER_TO_GROUP[logger_name] = group_name


def configure_logging(
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    verbose: bool = False,
) -> None:
    """
    Configure logging for the Ember framework.

    This function sets appropriate logging levels for all Ember components
    and external dependencies. It does not configure handlers (following Python
    best practices for libraries).

    Args:
        level: Base logging level for all components (default: INFO).
        format_str: Log format string (only used for root logger).
        verbose: If False, reduces verbosity for non-essential components.
    """
    # Set base logging level for Ember
    ember_logger = logging.getLogger("ember")
    ember_logger.setLevel(level)

    # Configure root logger format only if not already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_str))
        root_logger.addHandler(handler)

    # Set component-specific levels based on verbosity
    quiet_level = logging.INFO if verbose else logging.WARNING

    # Model registry and discovery are typically verbose and not needed at INFO level
    set_component_group_level("model_registry", quiet_level)
    set_component_group_level("model_discovery", quiet_level)

    # HTTP libraries are very verbose - use ERROR by default unless overridden
    http_log_level = os.environ.get("EMBER_HTTP_LOG_LEVEL", "ERROR")
    try:
        http_level = getattr(logging, http_log_level.upper())
    except AttributeError:
        http_level = logging.ERROR
    set_component_group_level("http", http_level)

    # Add NullHandler to HTTP libraries to prevent "no handler" warnings
    # and shutdown logging errors
    _configure_http_library_handlers()

    # Some components need even stricter logging in non-verbose mode
    if not verbose:
        # Discovery warnings are too numerous during model initialization
        set_component_group_level("model_warnings", logging.ERROR)


def set_component_level(component: str, level: int) -> None:
    """
    Set logging level for a specific Ember component.

    Args:
        component: Component name or logger name
        level: Logging level to set (e.g., logging.INFO, logging.WARNING)
    """
    # If component is a known group, set the entire group
    if component in COMPONENT_GROUPS:
        set_component_group_level(component, level)
        return

    # Otherwise set individual logger
    logger = logging.getLogger(component)
    logger.setLevel(level)


def set_component_group_level(group_name: str, level: int) -> None:
    """
    Set logging level for an entire component group.

    Args:
        group_name: Name of component group
        level: Logging level to set
    """
    if group_name not in COMPONENT_GROUPS:
        raise ValueError(f"Unknown component group: {group_name}")

    for logger_name in COMPONENT_GROUPS[group_name]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def _configure_http_library_handlers() -> None:
    """
    Configure HTTP libraries with appropriate log levels.

    Sets conservative log levels for HTTP client libraries to reduce verbosity
    in normal operation. The actual handling of closed streams during shutdown
    is now handled by code in conftest.py that runs earlier in the process.
    """
    # Get the configured HTTP log level from environment or use ERROR
    http_log_level = os.environ.get("EMBER_HTTP_LOG_LEVEL", "ERROR")
    try:
        http_level = getattr(logging, http_log_level.upper())
    except AttributeError:
        http_level = logging.ERROR
    
    # Extended list of HTTP libraries to suppress
    http_libraries = [
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpx",
        "urllib3",
        "urllib3.connectionpool",
        "urllib3.util.retry",
        "openai",
        "openai._base_client",
        "openai._http_client", 
        "anthropic",
        "anthropic._base_client",
        "requests",
        "requests.packages.urllib3",
    ]

    # Use configured level for all HTTP libraries
    for name in http_libraries:
        logger = logging.getLogger(name)
        logger.setLevel(http_level)
        # Add NullHandler to prevent propagation of unwanted logs
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())


def get_ember_logger(name: str) -> logging.Logger:
    """
    Get a logger with the ember namespace.

    Args:
        name: Logger name (will be prefixed with 'ember.' if not already)

    Returns:
        Configured logger instance
    """
    if not name.startswith("ember."):
        name = f"ember.{name}"
    return logging.getLogger(name)


# Context managers for temporary log level control

from contextlib import contextmanager
from typing import Optional, List, Union


@contextmanager
def suppress_logs(loggers: Union[str, List[str]], level: int = logging.ERROR):
    """
    Context manager to temporarily suppress logs from specific loggers.
    
    Args:
        loggers: Logger name(s) to suppress. Can be a string or list of strings.
        level: Minimum level to show (default: ERROR, so only ERROR and CRITICAL show)
        
    Example:
        with suppress_logs("ember.core.registry.model"):
            # Model discovery logs suppressed here
            registry = initialize_registry()
    """
    if isinstance(loggers, str):
        loggers = [loggers]
    
    # Store original levels
    original_levels = {}
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(level)
    
    try:
        yield
    finally:
        # Restore original levels
        for logger_name, original_level in original_levels.items():
            logging.getLogger(logger_name).setLevel(original_level)


@contextmanager
def log_level(level: Union[int, str], logger_name: Optional[str] = None):
    """
    Context manager to temporarily change log level.
    
    Args:
        level: Log level (e.g., logging.DEBUG, "DEBUG")
        logger_name: Specific logger to modify (None for root logger)
        
    Example:
        with log_level("DEBUG", "ember.xcs"):
            # XCS logs at DEBUG level here
            result = execute_graph(graph, inputs)
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Get the logger
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    
    # Store original level
    original_level = logger.level
    logger.setLevel(level)
    
    try:
        yield
    finally:
        # Restore original level
        logger.setLevel(original_level)


@contextmanager
def verbose_mode():
    """
    Context manager for temporary verbose logging.
    
    Enables INFO level for all Ember components temporarily.
    
    Example:
        with verbose_mode():
            # All Ember logs at INFO level or above
            models.list()
    """
    # Get all Ember component groups
    groups_to_adjust = ["model_registry", "model_discovery", "http"]
    
    # Store original levels
    original_levels = {}
    
    # Set INFO level for all groups
    for group in groups_to_adjust:
        if group in COMPONENT_GROUPS:
            for logger_name in COMPONENT_GROUPS[group]:
                logger = logging.getLogger(logger_name)
                original_levels[logger_name] = logger.level
                logger.setLevel(logging.INFO)
    
    try:
        yield
    finally:
        # Restore original levels
        for logger_name, original_level in original_levels.items():
            logging.getLogger(logger_name).setLevel(original_level)
