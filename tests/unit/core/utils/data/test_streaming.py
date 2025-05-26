"""Tests for streaming dataset implementation.

These tests verify that the streaming dataset implementation works
correctly with minimal memory usage.
"""

from typing import Any, Dict

from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.streaming import StreamingDataset


class MockTransformer:
    """Mock transformer for testing."""

    def __init__(self, transform_fn):
        """Initialize with transform function."""
        self.transform_fn = transform_fn
        self.called = 0

    def transform_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single item."""
        self.called += 1
        return self.transform_fn(item)


class TestStreamingDataset:
    """Tests for StreamingDataset implementation."""

    def test_streaming_iteration(self):
        """Test basic iteration through a streaming dataset."""
        # Create mock data source
        source = [
            {"id": 1, "question": "Q1"},
            {"id": 2, "question": "Q2"},
            {"id": 3, "question": "Q3"}]

        # For the StreamingDataset.py implementation with context architecture
        from ember.core.utils.data.context.data_context import DataContext
        context = DataContext.create_test_context()
        
        # Create streaming dataset with context
        dataset = StreamingDataset(source, _data_context=context)

        # Iterate and collect items
        items = list(dataset)

        # Verify we got all items
        assert len(items) == 3
        
        # In the new architecture, check for interface rather than type
        for item in items:
            assert hasattr(item, "query")
            
        assert items[0].query == "Q1"
        assert items[1].query == "Q2"
        assert items[2].query == "Q3"

    def test_transformations(self):
        """Test transformations are applied properly."""
        # Create mock data source
        source = [
            {"id": 1, "question": "Q1"},
            {"id": 2, "question": "Q2"},
            {"id": 3, "question": "Q3"}]

        # For the StreamingDataset.py implementation with context architecture
        from ember.core.utils.data.context.data_context import DataContext
        context = DataContext.create_test_context()

        # Create transformer that adds a prefix
        transformer = MockTransformer(
            lambda item: {**item, "question": f"Modified: {item['question']}"}
        )

        # Create streaming dataset with transformer and context
        dataset = StreamingDataset(source, transformers=[transformer], _data_context=context)

        # Iterate and collect items
        items = list(dataset)

        # Verify transformations were applied
        assert len(items) == 3
        assert items[0].query == "Modified: Q1"
        assert items[1].query == "Modified: Q2"
        assert items[2].query == "Modified: Q3"

        # Verify transformer was called for each item
        assert transformer.called == 3

    def test_limit(self):
        """Test limiting the number of items."""
        # Create mock data source
        source = [
            {"id": 1, "question": "Q1"},
            {"id": 2, "question": "Q2"},
            {"id": 3, "question": "Q3"},
            {"id": 4, "question": "Q4"},
            {"id": 5, "question": "Q5"}]

        # Create streaming dataset
        dataset = StreamingDataset(source)

        # Apply limit
        limited = dataset.limit(2)

        # Iterate and collect items
        items = list(limited)

        # Verify we got only the limited number of items
        assert len(items) == 2
        assert items[0].query == "Q1"
        assert items[1].query == "Q2"

    def test_filter(self):
        """Test filtering items."""
        # Create mock data source
        source = [
            {"id": 1, "question": "Q1", "category": "math"},
            {"id": 2, "question": "Q2", "category": "science"},
            {"id": 3, "question": "Q3", "category": "math"},
            {"id": 4, "question": "Q4", "category": "history"},
            {"id": 5, "question": "Q5", "category": "math"}]

        # Create custom transformer for filtering
        class FilterMathTransformer:
            def transform_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
                if item.get("category") == "math":
                    return item
                return None

        # Create streaming dataset with filter
        dataset = StreamingDataset(source, transformers=[FilterMathTransformer()])

        # Iterate and collect items
        items = list(dataset)

        # Verify we got only math questions
        assert len(items) == 3
        assert items[0].query == "Q1"
        assert items[1].query == "Q3"
        assert items[2].query == "Q5"

        # Verify metadata was preserved
        assert items[0].metadata["category"] == "math"

    def test_multiple_transformations(self):
        """Test applying multiple transformations in sequence."""
        # Create mock data source
        source = [
            {"id": 1, "question": "Q1"},
            {"id": 2, "question": "Q2"}]

        # Create transformers
        transformer1 = MockTransformer(
            lambda item: {**item, "question": f"First: {item['question']}"}
        )
        transformer2 = MockTransformer(
            lambda item: {**item, "question": f"Second: {item['question']}"}
        )

        # Create streaming dataset with transformers
        dataset = StreamingDataset(source, transformers=[transformer1, transformer2])

        # Iterate and collect items
        items = list(dataset)

        # Verify transformations were applied in order
        assert len(items) == 2
        assert items[0].query == "Second: First: Q1"
        assert items[1].query == "Second: First: Q2"

        # Verify transformers were called for each item
        assert transformer1.called == 2
        assert transformer2.called == 2

    def test_reuse_iterator(self):
        """Test reusing a streaming dataset iterator."""
        # Create mock data source
        source = [
            {"id": 1, "question": "Q1"},
            {"id": 2, "question": "Q2"}]

        # Create streaming dataset
        dataset = StreamingDataset(source)

        # Iterate once
        first_items = list(dataset)
        assert len(first_items) == 2

        # Iterate again
        second_items = list(dataset)
        assert len(second_items) == 2

        # Verify items are the same
        assert first_items[0].query == second_items[0].query
        assert first_items[1].query == second_items[1].query

    def test_chain_operations(self):
        """Test chaining operations."""
        # Create mock data source
        source = [
            {"id": 1, "question": "Q1", "category": "math"},
            {"id": 2, "question": "Q2", "category": "science"},
            {"id": 3, "question": "Q3", "category": "math"},
            {"id": 4, "question": "Q4", "category": "history"},
            {"id": 5, "question": "Q5", "category": "math"}]

        # Define transformers directly for better control
        class MathFilter:
            def transform_item(self, item):
                return item if item.get("category") == "math" else None

        class AddPrefixTransformer:
            def transform_item(self, item):
                if item is None:
                    return None
                return {**item, "question": f"Math: {item['question']}"}

        # Create streaming dataset with chained operations
        dataset = StreamingDataset(
            source=source,
            transformers=[MathFilter(), AddPrefixTransformer()]).limit(2)

        # Iterate and collect items
        items = list(dataset)

        # Verify operations were applied correctly
        assert len(items) == 2
        assert items[0].query == "Math: Q1"
        assert items[1].query == "Math: Q3"
