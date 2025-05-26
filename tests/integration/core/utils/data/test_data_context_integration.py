"""Integration tests for DataContext.

Tests integration between DataContext and other components of the data system.
"""

import pytest
from unittest import mock

from ember.api.data import DataAPI
from ember.core.utils.data.base.models import DatasetEntry, TaskType
from ember.core.utils.data.context.data_context import DataContext
from ember.core.utils.data.base.preppers import IDatasetPrepper


@pytest.fixture
def test_context() -> DataContext:
    """Fixture that creates a test context with a sample dataset."""
    context = DataContext.create_test_context()

    # Register a test dataset
    context.register_dataset(
        name="test_dataset",
        source="test/source",
        task_type=TaskType.MULTIPLE_CHOICE,
        description="Test dataset for integration tests")

    return context


class TestDataApiIntegration:
    """Tests for DataAPI integration with DataContext."""

    def test_data_api_with_context(self, test_context: DataContext):
        """Test creating a DataAPI with an explicit context."""
        # Create API with test context
        api = DataAPI(context=test_context)

        # Check that it can access the test dataset
        datasets = api.list()
        assert "test_dataset" in datasets

        # Check that the API can get dataset info
        info = api.info("test_dataset")
        assert info is not None
        assert info.name == "test_dataset"
        assert info.description == "Test dataset for integration tests"

    def test_builder_with_context(self, test_context):
        """Test using DatasetBuilder with an explicit context."""
        # Create API with test context
        api = DataAPI(context=test_context)

        # Use builder
        builder = api.builder()

        # Check that builder can access the test dataset
        builder.from_registry("test_dataset")

        # Configure builder
        builder.sample(10)
        builder.split("test")
        
        # Verify configuration is set
        assert builder._sample_size == 10
        assert builder._split == "test"


class SimpleTestPrepper(IDatasetPrepper):
    """Simple prepper for testing."""
    
    def prepare(self, data, config=None):
        """Return data unchanged."""
        return data
        
    def create_dataset_entries(self, dataset, config=None):
        """Create DatasetEntry objects."""
        return [
            DatasetEntry(
                id=f"test-{i}",
                query=f"Question {i}",
                choices={"A": "Option A", "B": "Option B"}
            )
            for i in range(5)
        ]
        
    def get_required_keys(self):
        """Return required keys."""
        return []


class TestIntegrationWithService:
    """Tests for integration with DatasetService."""
    
    def test_context_service_integration(self, test_context):
        """Test that context integrates with service layer."""
        # Register test dataset with a prepper
        test_context.register_dataset(
            name="test_prepper_dataset",
            source="test/source",
            task_type=TaskType.MULTIPLE_CHOICE,
            prepper_class=SimpleTestPrepper)
        
        # Verify prepper is registered with loader factory
        prepper_class = test_context.loader_factory.get_prepper_class(
            dataset_name="test_prepper_dataset"
        )
        assert prepper_class is SimpleTestPrepper
        
        # Create API with context
        api = DataAPI(context=test_context)
        
        # Build a builder - verify it can access the registered dataset
        builder = api.builder()
        builder.from_registry("test_prepper_dataset")
        
        # Verify loader factory is properly integrated
        assert "test_prepper_dataset" in test_context.loader_factory.list_registered_preppers()