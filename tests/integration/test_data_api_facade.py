"""Integration tests for the Ember Data API facade.

These tests verify that the data API facade in ember.api.data correctly
interacts with the underlying components, without requiring external API calls.
"""

import unittest
from typing import Any, Dict, List
from unittest import mock

import pytest

from ember.api.data import (
    DataAPI,
    Dataset,
    DatasetBuilder,
    DatasetEntry,
    DatasetInfo,
    TaskType)
from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.base.validators import DatasetValidator
from ember.core.utils.data.context.data_context import DataContext


@pytest.fixture(scope="module", autouse=True)
def patch_hf_dataset_loader():
    """Patch HuggingFaceDatasetLoader to avoid actual HF downloads."""
    with mock.patch(
        "ember.core.utils.data.base.loaders.HuggingFaceDatasetLoader.load"
    ) as mock_load:
        # Configure mock dataset
        mock_dataset = mock.MagicMock()
        mock_dataset_dict = mock.MagicMock()
        mock_dataset_dict.__getitem__.return_value = mock_dataset
        mock_dataset_dict.keys.return_value = ["test"]

        # Set up return value for the mock
        mock_load.return_value = mock_dataset_dict
        yield mock_load


@pytest.fixture
def register_test_datasets():
    """Register test datasets to ensure they're available for all tests."""

    from ember.core.utils.data.base.models import DatasetEntry
    from ember.core.utils.data.base.preppers import IDatasetPrepper
    from ember.core.utils.data.registry import DATASET_REGISTRY, RegisteredDataset

    # Define test prepper inline - this keeps the test self-contained
    # and avoids adding unnecessary classes to production code
    class TestDatasetPrepper(IDatasetPrepper):
        """Simple prepper implementation for tests only."""

        def get_required_keys(self) -> List[str]:
            """Return minimal required keys for test data."""
            return ["query"]

        def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
            """Create a simple dataset entry from test data."""
            return [
                DatasetEntry(
                    id=str(item.get("id", "test")),
                    query=item.get("query", ""),
                    metadata={
                        k: v for k, v in item.items() if k not in ["id", "query"]
                    })
            ]

    # Create and register test datasets that match those referenced in tests
    datasets_to_register = [
        ("preset_dataset", "test source", TaskType.SHORT_ANSWER),
        ("explicit_dataset", "test source", TaskType.SHORT_ANSWER),
        ("custom/test", "custom test", TaskType.SHORT_ANSWER)]

    # Save existing registry to restore later
    original_registry = {}
    for name, entry in DATASET_REGISTRY._registry.items():
        original_registry[name] = entry

    # Add test datasets
    for name, source, task_type in datasets_to_register:
        info = DatasetInfo(
            name=name,
            description=f"Test dataset {name}",
            source=source,
            task_type=task_type)
        DATASET_REGISTRY._registry[name] = RegisteredDataset(
            name=name, info=info, prepper=TestDatasetPrepper()
        )

    yield

    # Restore original registry
    DATASET_REGISTRY._registry = original_registry


class TestDataAPIFacade(unittest.TestCase):
    """Test the data API facade with custom in-memory datasets."""

    def setUp(self) -> None:
        """Set up test fixtures with minimal test doubles."""
        # Create a test context
        self.test_context = DataContext.create_test_context()
        # Create an API instance using the context
        self.api = DataAPI(context=self.test_context)

        # Register a dummy dataset in the context
        self.api.register(
            name="mmlu", source="dummy/mmlu", task_type=TaskType.MULTIPLE_CHOICE
        )
        self.api.register(
            name="dataset1", source="dummy/ds1", task_type=TaskType.GENERATION
        )
        self.api.register(
            name="dataset2", source="dummy/ds2", task_type=TaskType.GENERATION
        )

        # Mock HuggingFaceDatasetLoader.load method directly
        self.loader_patcher = mock.patch.object(HuggingFaceDatasetLoader, "load")

    def test_list_datasets(self) -> None:
        """list_datasets() should return dataset names."""
        # Act
        result = self.api.list()

        # Assert - verify function behavior, not implementation details
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(ds, str) for ds in result))
        self.assertIn("mmlu", result)
        self.assertIn("dataset1", result)
        self.assertIn("dataset2", result)

    def test_get_dataset_info(self) -> None:
        """get_dataset_info() should return dataset info object."""
        # Use a standard dataset that should exist (registered in setUp)
        # Act
        result = self.api.info(name="mmlu")

        # Assert on behavior
        self.assertIsNotNone(result)
        # Check fields rather than exact type to make the test more robust
        # against import path changes or class version differences
        self.assertEqual("mmlu", result.name)
        self.assertTrue(hasattr(result, "task_type"))
        self.assertTrue(hasattr(result, "source"))

    def test_builder_method(self) -> None:
        """builder() should return a DatasetBuilder instance."""
        # Act
        builder_instance = self.api.builder()

        # Assert
        # Import here to avoid issues if api.data itself is mocked
        from ember.api.data import DatasetBuilder

        self.assertIsInstance(builder_instance, DatasetBuilder)

    def test_dataset_methods(self) -> None:
        """Dataset methods should provide correct access to entries."""
        # Create a dataset directly without going through the API
        # Using correct DatasetEntry structure - it doesn't have an id field directly
        entries = [
            DatasetEntry(
                query="Question 1?", metadata={"id": "1", "answer": "Answer 1"}
            ),
            DatasetEntry(
                query="Question 2?", metadata={"id": "2", "answer": "Answer 2"}
            ),
            DatasetEntry(
                query="Question 3?", metadata={"id": "3", "answer": "Answer 3"}
            )]
        dataset = Dataset(entries=entries)

        # Assert - basic dataset behavior
        assert len(dataset) == 3

        # Assert - __getitem__
        assert dataset[0].query == "Question 1?"
        assert dataset[0].metadata["id"] == "1"
        assert dataset[1].query == "Question 2?"
        assert dataset[2].query == "Question 3?"

        # Assert - __iter__
        items = list(iter(dataset))
        assert len(items) == 3
        assert items[0].query == "Question 1?"
