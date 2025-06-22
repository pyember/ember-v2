"""Registry module for ember components."""

from __future__ import annotations

# Import EmberModel for easy access
from ember.core.types import EmberModel

# Import only available subpackages
from . import specification

__all__ = [
    "specification",
    "EmberModel"]
