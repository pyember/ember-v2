"""Ember public API package.

This package provides the stable public interface to Ember functionality.
All public APIs are accessible through this single namespace.

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
from ember.models.catalog import Models

import ember.api.eval as evaluation
import ember.api.operators as operators
import ember.api.types as types
import ember.api.xcs as xcs


__all__ = [
    # Core facades
    "models",
    "Models",
    "op",
    # Module namespaces
    "evaluation",
    "operators",
    "types",
    "xcs",
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
]
