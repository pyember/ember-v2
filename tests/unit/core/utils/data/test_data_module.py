"""Unit tests for the data module's high-level API.

These tests verify that the public API in ember.api.data works correctly
and interfaces properly with the underlying components.
"""

import unittest
from unittest import mock

from ember.api.data import (
    DataAPI,
    Dataset,
    DatasetConfig,
    DatasetEntry,
    DatasetInfo,
    TaskType)
from ember.core.utils.data import load_dataset_entries
from ember.core.utils.data.context.data_context import DataContext


class TestLoadDatasetEntriesBasic(unittest.TestCase):
    """Basic tests for the load_dataset_entries function that don't need complex mocking."""

    def test_load_dataset_entries_with_named_params(self) -> None:
        """load_dataset_entries() should require named parameters."""
        # Arrange & Act & Assert
        # This test doesn't need mocking as it fails before any functionality is executed
        with self.assertRaises(TypeError):
            load_dataset_entries("test_dataset")  # type: ignore


class TestDatasetClass(unittest.TestCase):
    """Test cases for the Dataset class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entries = [
            DatasetEntry(id="1", query="Question 1?", metadata={"answer": "Answer 1"}),
            DatasetEntry(id="2", query="Question 2?", metadata={"answer": "Answer 2"}),
            DatasetEntry(id="3", query="Question 3?", metadata={"answer": "Answer 3"})]
        self.dataset_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.SHORT_ANSWER)
        self.dataset = Dataset(entries=self.entries, info=self.dataset_info)

    def test_init(self) -> None:
        """Dataset initialization should store entries and info."""
        # Arrange & Act in setUp

        # Assert
        self.assertEqual(self.entries, self.dataset.entries)
        self.assertEqual(self.dataset_info, self.dataset.info)

    def test_getitem(self) -> None:
        """Dataset.__getitem__ should return entry at specified index."""
        # Arrange & Act & Assert
        self.assertEqual(self.entries[0], self.dataset[0])
        self.assertEqual(self.entries[1], self.dataset[1])
        self.assertEqual(self.entries[2], self.dataset[2])

    def test_getitem_out_of_range(self) -> None:
        """Dataset.__getitem__ should raise IndexError for out-of-range index."""
        # Arrange & Act & Assert
        with self.assertRaises(IndexError):
            _ = self.dataset[3]

    def test_iter(self) -> None:
        """Dataset.__iter__ should iterate over all entries."""
        # Arrange & Act
        entries = list(iter(self.dataset))

        # Assert
        self.assertEqual(self.entries, entries)

    def test_len(self) -> None:
        """Dataset.__len__ should return number of entries."""
        # Arrange & Act & Assert
        self.assertEqual(3, len(self.dataset))

    def test_empty_dataset(self) -> None:
        """Dataset should work with empty entries list."""
        # Arrange & Act
        empty_dataset = Dataset(entries=[])

        # Assert
        self.assertEqual(0, len(empty_dataset))
        self.assertEqual([], list(empty_dataset))


# DataFunction tests removed - will be reimplemented with proper test doubles


# Legacy API tests removed - APIs modernized


# Legacy API tests removed - APIs modernized


if __name__ == "__main__":
    unittest.main()
