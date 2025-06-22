"""Ember public API package.

This package provides a clean, stable public interface to Ember functionality.
All public APIs are accessible through this package, allowing users to import
from a single location while implementation details remain encapsulated.

Examples:
    # Import the main facades
    from ember.api import models, datasets, operators

    # Use models with a clean, namespaced interface
    response = models.openai.gpt4o("What's the capital of France?")

    # Load datasets directly
    mmlu_data = datasets("mmlu")

    # Or use the builder pattern
    from ember.api import DatasetBuilder
    dataset = DatasetBuilder().split("test").sample(100).build("mmlu")

    # Use Network of Networks (NON) patterns
    from ember.api import non
    ensemble = non.UniformEnsemble(num_units=3, model_name="openai:gpt-4o")

    # Optimize with XCS
    from ember.api import xcs
    @xcs.jit
    def optimized_fn(x):
        return complex_computation(x)
"""

# Import module namespaces
import ember.api.eval as eval  # Evaluation module
from ember.api.models import models  # Import the singleton instance, not the module
# TODO: Fix non.py to use new operators_v2 system
# import ember.api.non as non  # Network of Networks patterns
import ember.api.operators as operators  # Operator system
import ember.api.types as types
import ember.api.xcs as xcs  # Execution optimization system

# Make operators available as both singular and plural for backward compatibility
operator = operators

# Import core components
from ember.core.context.ember_context import EmberContext

# Import data API components
from ember.api.data import (
    stream,
    load,
    metadata,
    list_datasets,
    register,
    from_file,
    load_file,
    DataSource,
    DatasetInfo,
    StreamIterator,
    FileSource,
    HuggingFaceSource,
)

# For backward compatibility, expose stream as 'data'
data = stream
from ember.api.eval import EvaluationPipeline  # Pipeline for batch evaluation
from ember.api.eval import Evaluator  # Evaluator for model outputs


# Import decorators
from ember.api.decorators import op

# Convenience function for creating model bindings
def model(model_id: str, **params):
    """Create a model binding for use in operators.
    
    This is a convenience function that wraps models.instance() for cleaner
    operator code.
    
    Args:
        model_id: Model identifier (e.g., "gpt-4", "claude-3")
        **params: Default parameters for all calls (temperature, etc.)
        
    Returns:
        ModelBinding that can be called multiple times
        
    Examples:
        >>> # In an operator
        >>> class MyOperator(ember.Operator):
        ...     model: ModelBinding
        ...     
        ...     def __init__(self):
        ...         self.model = ember.model("gpt-4", temperature=0.7)
    """
    return models.instance(model_id, **params)

# Public interface - export facades, modules, and direct API components
__all__ = [
    # Main facade objects
    "models",  # Model access (models.openai.gpt4o, etc.)
    "model",   # Convenience function for creating model bindings
    "op",      # Decorator for function-style operators
    "data",  # Data access (data("mmlu"), data.builder(), etc.)
    # Module namespaces
    "eval",  # Evaluation module
    "non",  # Network of Networks patterns
    "xcs",  # Execution optimization
    "operators",  # Operator system (plural)
    "operator",  # Operator system (singular, for backward compatibility)
    "types",  # Types system for Ember models and operators
    # Model API components
    "ModelAPI",  # High-level model API
    "ModelBuilder",  # Builder pattern for model configuration
    "ModelEnum",  # Type-safe model references
    # Data API components
    "stream",  # Stream data (main function)
    "load",  # Load data into memory
    "metadata",  # Get dataset metadata
    "list_datasets",  # List available datasets
    "register",  # Register custom data source
    "from_file",  # Stream from file
    "load_file",  # Load file into memory
    "DataSource",  # Protocol for data sources
    "DatasetInfo",  # Dataset metadata
    "StreamIterator",  # Iterator with chaining
    "FileSource",  # File data source
    "HuggingFaceSource",  # HuggingFace data source
    # Evaluation API components
    "Evaluator",  # Evaluator for model outputs
    "EvaluationPipeline",  # Pipeline for batch evaluation
]
