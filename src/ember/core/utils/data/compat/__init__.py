"""Compatibility layer for data system.

This module provides backward compatibility for transitioning
from the global registry to the context-based system.
"""

from ember.core.utils.data.compat.registry_proxy import RegistryProxy

__all__ = [
    "RegistryProxy"]
