"""Integration tests for the Data API with thread-local context architecture.

This file tests the public API contract of the data system focused on:
1. Thread isolation of contexts
2. API behaviors across context boundaries
3. Core user workflows with minimal mocking
"""

import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import pytest

from ember.api.data import DataAPI, DataItem, DatasetEntry
from ember.core.utils.data.base.models import TaskType
from ember.core.utils.data.context.data_context import DataContext


@pytest.fixture
def test_context():
    """Fixture providing a test context with basic datasets."""
    context = DataContext.create_test_context()
    
    # Register test datasets
    context.register_dataset(
        name="test_dataset",
        source="test/source",
        task_type=TaskType.MULTIPLE_CHOICE,
        description="Test dataset for integration testing",
    )
    
    return context


@pytest.fixture
def data_api(test_context):
    """Fixture providing a DataAPI instance with the test context."""
    return DataAPI(context=test_context)


class TestContextIsolation:
    """Tests for thread isolation in the context architecture."""
    
    def test_separate_contexts_have_isolated_datasets(self):
        """Different contexts should have completely isolated dataset registries."""
        # Create two independent contexts
        ctx_a = DataContext.create_test_context()
        ctx_b = DataContext.create_test_context()
        
        # Register different datasets in each context
        ctx_a.register_dataset(
            name="only_in_a", 
            source="test/a", 
            task_type=TaskType.MULTIPLE_CHOICE
        )
        
        ctx_b.register_dataset(
            name="only_in_b", 
            source="test/b", 
            task_type=TaskType.MULTIPLE_CHOICE
        )
        
        # Create APIs with each context
        api_a = DataAPI(context=ctx_a)
        api_b = DataAPI(context=ctx_b)
        
        # Verify isolation
        assert "only_in_a" in api_a.list()
        assert "only_in_b" not in api_a.list()
        
        assert "only_in_b" in api_b.list()
        assert "only_in_a" not in api_b.list()
    
    def test_context_isolation_across_threads(self):
        """Contexts should maintain isolation across thread boundaries."""
        results = queue.Queue()
        
        def thread_with_context(context_name):
            """Worker function that creates its own context and registers datasets."""
            # Create a fresh context in this thread
            ctx = DataContext.create_test_context()
            
            # Register a dataset specific to this thread
            dataset_name = f"thread_{context_name}_dataset"
            ctx.register_dataset(
                name=dataset_name,
                source="test/thread",
                task_type=TaskType.MULTIPLE_CHOICE,
            )
            
            # Create an API with this context
            api = DataAPI(context=ctx)
            
            # Verify this thread can see its dataset
            datasets = api.list()
            results.put((context_name, dataset_name, dataset_name in datasets))
        
        # Start multiple threads with different contexts
        with ThreadPoolExecutor(max_workers=3) as executor:
            for i in range(3):
                executor.submit(thread_with_context, f"worker_{i}")
        
        # Collect and verify results
        thread_results = []
        while not results.empty():
            thread_results.append(results.get())
        
        # Each thread should have been able to see its own dataset
        for name, dataset, found in thread_results:
            assert found, f"Thread {name} couldn't see its dataset {dataset}"
        
        # Verify we got results from all 3 threads
        assert len(thread_results) == 3


class TestDatasetOperations:
    """Tests for core dataset operations with the DataAPI."""
    
    def test_list_datasets(self, data_api, test_context):
        """API should list all datasets in the context."""
        # Register additional test datasets
        test_context.register_dataset(
            name="extra_dataset",
            source="test/extra",
            task_type=TaskType.SHORT_ANSWER,
        )
        
        # Get dataset list
        datasets = data_api.list()
        
        # Verify all registered datasets are listed
        assert "test_dataset" in datasets
        assert "extra_dataset" in datasets
    
    def test_dataset_info(self, data_api):
        """API should return correct dataset information."""
        # Get info for test dataset
        info = data_api.info("test_dataset")
        
        # Verify info matches what was registered
        assert info is not None
        assert info.name == "test_dataset"
        assert info.source == "test/source"
        assert info.task_type == TaskType.MULTIPLE_CHOICE
        assert info.description == "Test dataset for integration testing"
        
        # Non-existent dataset should return None
        assert data_api.info("nonexistent") is None
    
    def test_register_dataset(self, data_api):
        """API should register datasets in the context."""
        # Register a new dataset
        data_api.register(
            name="new_dataset",
            source="test/new",
            task_type=TaskType.SHORT_ANSWER,  # Use existing enum value
            description="Newly registered dataset",
        )
        
        # Verify it appears in the list
        assert "new_dataset" in data_api.list()
        
        # Verify info is correct
        info = data_api.info("new_dataset")
        assert info is not None
        assert info.name == "new_dataset"
        assert info.source == "test/new"
        assert info.task_type == TaskType.SHORT_ANSWER
        assert info.description == "Newly registered dataset"


class TestBuilderPattern:
    """Tests for the DatasetBuilder pattern."""
    
    def test_builder_access_to_context(self, data_api):
        """Builder should access datasets through its context."""
        # Create builder
        builder = data_api.builder()
        
        # Should be able to access registered datasets
        builder.from_registry("test_dataset")  # Should not raise error
        
        # Should raise for non-existent datasets
        with pytest.raises(ValueError, match="not found"):
            builder.from_registry("nonexistent_dataset")
    
    def test_builder_configuration(self, data_api):
        """Builder should apply and store configuration."""
        # Configure a builder with chaining
        builder = (
            data_api.builder()
            .from_registry("test_dataset")
            .split("test")
            .subset("mathematics")
            .sample(10)
            .seed(42)
        )
        
        # Check internal configuration state
        assert builder._dataset_name == "test_dataset"
        assert builder._split == "test"
        assert builder._config["subset"] == "mathematics"
        assert builder._sample_size == 10
        assert builder._seed == 42
    
    def test_builder_transformation(self, data_api):
        """Builder should store and manage transformations."""
        # Define transformation functions
        def capitalize_text(item):
            if isinstance(item, dict) and "text" in item:
                return {**item, "text": item["text"].upper()}
            return item
        
        def add_prefix(item):
            if isinstance(item, dict) and "text" in item:
                return {**item, "text": f"PREFIX: {item['text']}"}
            return item
        
        # Add transformations to builder
        builder = data_api.builder()
        builder.transform(capitalize_text)
        builder.transform(add_prefix)
        
        # Verify transformers are stored in order
        assert len(builder._transformers) == 2


class TestStreamingBehavior:
    """Tests for streaming behavior in the DataAPI."""
    
    def test_streaming_configuration(self, data_api):
        """Builder should configure streaming behavior."""
        # Default should be streaming
        builder = data_api.builder()
        assert builder._streaming is True
        
        # Disable streaming
        builder.streaming(False)
        assert builder._streaming is False
        
        # Re-enable streaming
        builder.streaming(True)
        assert builder._streaming is True
    
    def test_limit_configuration(self, data_api):
        """Builder should configure limits for both streaming and non-streaming."""
        # Using limit() method
        builder = data_api.builder()
        builder.limit(42)
        assert builder._sample_size == 42
        
        # Using sample() method (alias)
        builder = data_api.builder()
        builder.sample(24)
        assert builder._sample_size == 24


class TestDataItemAPI:
    """Tests for the DataItem wrapper."""
    
    def test_data_item_normalization(self):
        """DataItem should normalize access to common fields."""
        # Create a DatasetEntry to test DataItem's normalization
        entry = DatasetEntry(
            query="What is the capital of France?",
            choices={"A": "London", "B": "Paris", "C": "Berlin"},
            metadata={"correct_answer": "B"},
        )
        
        item = DataItem(entry)
        
        # Normalized access
        assert item.question == "What is the capital of France?"
        assert item.options == {"A": "London", "B": "Paris", "C": "Berlin"}
        # The correct_answer field is copied directly from metadata
        assert item.correct_answer == "B"
        
    def test_data_item_dict_access(self):
        """DataItem should work with dictionaries."""
        # Test with a dictionary
        item = DataItem({
            "question": "What is the capital of France?",
            "options": {"A": "London", "B": "Paris", "C": "Berlin"},
            "answer": "B",
        })
        
        # Access normalized fields
        assert item.question == "What is the capital of France?"
        assert item.options == {"A": "London", "B": "Paris", "C": "Berlin"}
        assert item.answer == "B"
        
        # Access through __getattr__
        with pytest.raises(AttributeError):
            _ = item.nonexistent_field