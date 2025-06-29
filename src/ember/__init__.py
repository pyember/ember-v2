"""Ember: Framework for Compound AI Systems.

Ember provides a compositional framework for building and orchestrating
Compound AI Systems and Networks of Networks (NONs).

Architecture Philosophy:
    Ember is designed around three core principles:
    1. **Composability**: All components (models, operators, data sources) are
       composable building blocks that integrate seamlessly.
    2. **Zero Configuration**: Sensible defaults with progressive disclosure of
       complexity. Simple tasks are simple, complex tasks are possible.
    3. **JAX-First**: Built on JAX for automatic differentiation, compilation,
       and hardware acceleration while maintaining Python's expressiveness.

Package Structure:
    - api: Public API surface providing simplified access to core functionality
    - models: Model providers and catalog for LLM integration
    - operators: Composable operators for building compound AI systems and pipelines
    - xcs: "accelerated compound systems" package for advanced operator composition
    - _internal: Implementation details (not part of public API)

Basic usage:
    >>> from ember.api import models
    >>> response = models("gpt-4", "What is the capital of France?")
    >>> print(response.text)
    Paris

For more examples, see https://ember.ai/docs
"""

from __future__ import annotations

import importlib.metadata

from ember.api import models, operators, xcs

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
