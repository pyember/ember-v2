"""Ember: Framework for Compound AI Systems.

Ember provides a compositional framework for building and orchestrating
Compound AI Systems and Networks of Networks (NONs).

Basic usage:
    >>> from ember.api import models
    >>> response = models("gpt-4", "What is the capital of France?")
    >>> print(response.text)
    Paris

For more examples, see https://ember.ai/docs
"""

from __future__ import annotations

import importlib.metadata

from ember.api import models
from ember.api import operators
from ember.api import xcs

try:
    __version__ = importlib.metadata.version("ember-ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"






__all__ = [
    "__version__",
    "models",
    "operators", 
    "xcs",
]
