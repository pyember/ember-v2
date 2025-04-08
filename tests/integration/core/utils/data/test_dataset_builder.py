"""Integration tests for DatasetBuilder with thread-local context architecture.

Tests the DatasetBuilder API with the thread-safe context system, focusing on
behavioral correctness rather than implementation details.
"""

import pytest

from ember.api.data import DataAPI, DatasetBuilder 
from ember.core.context.ember_context import current_context
from ember.core.utils.data.base.models import TaskType
from ember.core.utils.data.context.data_context import DataContext


def test_builder_creates_from_context():
    """Test DatasetBuilder works with the context system."""
    # Create a test context
    ctx = DataContext.create_test_context()
    
    # Create builder with context
    builder = DatasetBuilder(context=ctx)
    
    # Test basic operations
    assert builder is not None
    assert hasattr(builder, "from_registry")
    
    # Verify methods return self for chaining
    assert builder.split("test") is builder
    assert builder.sample(100) is builder
    assert builder.seed(42) is builder


def test_dataset_registration_and_retrieval():
    """Test dataset registration and retrieval works with DataAPI."""
    # Create a test context
    ctx = DataContext.create_test_context()
    api = DataAPI(context=ctx)
    
    # Register a test dataset
    ctx.register_dataset(
        name="test-dataset",
        source="test/source",
        task_type=TaskType.MULTIPLE_CHOICE,
        description="Test dataset",
    )
    
    # List datasets and verify our test dataset is included
    datasets = api.list()
    assert "test-dataset" in datasets
    
    # Get dataset info
    info = api.info("test-dataset")
    assert info is not None
    assert info.name == "test-dataset"
    assert info.source == "test/source"
    assert info.task_type == TaskType.MULTIPLE_CHOICE


def test_dataset_builder_chain():
    """Test DatasetBuilder method chaining works correctly."""
    # Create context with test dataset
    ctx = DataContext.create_test_context()
    ctx.register_dataset(
        name="mmlu",
        source="test/source",
        task_type=TaskType.MULTIPLE_CHOICE,
    )
    
    # Create and configure builder
    api = DataAPI(context=ctx)
    builder = api.builder()
    builder.from_registry("mmlu")
    builder.subset("math")
    builder.split("test")
    builder.sample(10)
    
    # Verify internal state is correct
    assert builder._dataset_name == "mmlu"
    assert builder._config["subset"] == "math"
    assert builder._split == "test"
    assert builder._sample_size == 10


def test_builder_transformation():
    """Test that transformers are properly added to the builder."""
    ctx = DataContext.create_test_context()
    builder = DatasetBuilder(context=ctx)
    
    # Define a simple transformer function
    def transformer_fn(item):
        item["transformed"] = True
        return item
    
    # Add transformer
    builder.transform(transformer_fn)
    
    # Verify transformer was added
    assert len(builder._transformers) == 1