"""Core type definitions for Ember.

This module provides the base types used throughout Ember.

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

from pydantic import BaseModel

# EmberModel is a simple alias for Pydantic's BaseModel, providing
# full validation features with zero overhead.
EmberModel = BaseModel

__all__ = ["EmberModel"]