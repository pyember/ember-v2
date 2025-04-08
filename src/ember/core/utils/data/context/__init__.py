"""Context module for data system.

This package provides the DataContext class for explicit dependency management
and thread-safety in the data subsystem.
"""

from ember.core.utils.data.context.data_context import (
    DataConfig,
    DataContext,
    get_default_context,
    reset_default_context,
    set_default_context,
)

__all__ = [
    "DataConfig",
    "DataContext",
    "get_default_context",
    "reset_default_context",
    "set_default_context",
]
