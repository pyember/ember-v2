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
import ember.api.non as non  # Network of Networks patterns
import ember.api.operators as operators  # Operator system
import ember.api.types as types
import ember.api.xcs as xcs  # Execution optimization system

# Make operators available as both singular and plural for backward compatibility
operator = operators

# Import core components
from ember.core.context.ember_context import EmberContext

# Import high-level API components
from ember.api.data import DataAPI  # Added
from ember.api.data import Dataset  # Dataset container class
from ember.api.data import DatasetBuilder  # Builder pattern for dataset configuration
from ember.api.data import DatasetConfig  # Configuration for dataset loading
from ember.api.data import DatasetEntry  # Individual dataset entry
from ember.api.data import DatasetInfo  # Dataset metadata
from ember.api.data import TaskType  # Enum of dataset task types

# Initialize DataAPI and expose its methods
data_api = DataAPI(EmberContext.current())
data = data_api  # Expose as 'data' for cleaner API usage
register_dataset = data_api.register
from ember.api.eval import EvaluationPipeline  # Pipeline for batch evaluation
from ember.api.eval import Evaluator  # Evaluator for model outputs

# Public interface - export facades, modules, and direct API components
__all__ = [
    # Main facade objects
    "models",  # Model access (models.openai.gpt4o, etc.)
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
    "DataAPI",  # Added
    "DatasetBuilder",  # Builder pattern for dataset loading
    "Dataset",  # Dataset container class
    "DatasetConfig",  # Configuration for dataset loading
    "TaskType",  # Enum of dataset task types
    "DatasetInfo",  # Dataset metadata
    "DatasetEntry",  # Individual dataset entry
    "register_dataset",  # Dataset registration function
    # Evaluation API components
    "Evaluator",  # Evaluator for model outputs
    "EvaluationPipeline",  # Pipeline for batch evaluation
]
