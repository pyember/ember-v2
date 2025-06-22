"""Ember data model base class.

EmberModel is a simple alias for Pydantic's BaseModel, providing
full validation features with zero overhead.

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

from pydantic import BaseModel

# Zero overhead facade
EmberModel = BaseModel

__all__ = ["EmberModel"]