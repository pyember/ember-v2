"""Ember public API package.

This package provides the stable public interface to Ember functionality.
All public APIs are accessible through this single namespace.

Architecture Notes:
    The API package implements a facade pattern over Ember's internal architecture,
    providing simplified access while maintaining full power for advanced use cases.

    Design principles:
    1. **Single Import**: All common functionality available from ember.api
    2. **Progressive Disclosure**: Simple tasks require minimal imports and configuration
    3. **Type Safety**: Full type hints for IDE support and static analysis
    4. **Functional Style**: APIs designed for composition and chaining

    The API is organized into four core facades:
    - models: Direct LLM invocation with automatic provider selection
    - operators: Functional transformations and pipeline building
    - stream: Memory-efficient data processing with lazy evaluation
    - xcs: Cross-cutting concerns (caching, monitoring, fallbacks)

Why This Design:
    Traditional AI frameworks require verbose initialization and configuration.
    Ember's API eliminates boilerplate while preserving flexibility - achieving
    the "make simple things simple, complex things possible" principle.

    The functional design enables JAX transformations (vmap, pmap, jit) to be
    applied to entire pipelines of mixed learnable/diffentiable components,
    model invocations, and tool/mcp server calls. This provides automatic
    parallelization and optimization opportunities not available in
    object-oriented frameworks.

Basic usage:
    >>> from ember.api import models, stream, operators
    >>>
    >>> # Direct model invocation
    >>> response = models("gpt-4", "What's the capital of France?")
    >>> print(response.text)
    Paris
    >>>
    >>> # Stream data efficiently
    >>> for item in stream("mmlu").take(10):
    ...     print(item)
    >>>
    >>> # Create operators
    >>> @operators.op
    ... def summarize(text):
    ...     return models("gpt-4", f"Summarize: {text}").text
"""

# Module imports
import ember.api.eval as evaluation
import ember.api.exceptions as exceptions
import ember.api.operators as operators
import ember.api.types as types
import ember.api.validators as validators
import ember.api.xcs as xcs

# Specific imports
from ember.api.data import (
    DatasetInfo,
    DataSource,
    FileSource,
    HuggingFaceSource,
    StreamIterator,
    from_file,
    list_datasets,
    load,
    load_file,
    metadata,
    register,
    stream,
)
from ember.api.decorators import op
from ember.api.eval import EvaluationPipeline, Evaluator
from ember.api.models import models
from ember.api.validators import (
    ValidationHelpers,
    field_validator,
    model_validator,
)
from ember.models.catalog import Models

__all__ = [
    # Core facades
    "models",
    "Models",
    "op",
    # Module namespaces
    "evaluation",
    "operators",
    "types",
    "validators",
    "xcs",
    "exceptions",
    # Data API
    "stream",
    "load",
    "metadata",
    "list_datasets",
    "register",
    "from_file",
    "load_file",
    "DataSource",
    "DatasetInfo",
    "StreamIterator",
    "FileSource",
    "HuggingFaceSource",
    # Evaluation API
    "Evaluator",
    "EvaluationPipeline",
    # Validation API
    "field_validator",
    "model_validator",
    "ValidationHelpers",
]
