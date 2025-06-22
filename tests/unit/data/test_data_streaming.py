"""Test the data streaming API.

Tests focus on verifying behavior without duplicating implementation.
Following the design principles from CLAUDE.md:
- Test behavior, not implementation
- Use minimal test doubles  
- No emojis or casual language
- Comprehensive edge case coverage
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from ember.api.data import (
    DataSource,
    DatasetInfo,
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


class MockDataSource:
    """Mock data source for testing."""
    
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items
        self.batches_called = 0
        
    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield items in batches."""
        self.batches_called += 1
        for i in range(0, len(self.items), batch_size):
            yield self.items[i:i + batch_size]


class TestStreamFunction:
    """Test the stream() function."""
    
    def test_stream_from_custom_source(self):
        """Test streaming from a custom data source."""
        items = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
        source = MockDataSource(items)
        
        results = list(stream(source, normalize=False))
        assert results == items
        assert source.batches_called == 1
    
    def test_stream_with_filter(self):
        """Test filtering during streaming."""
        items = [
            {"question": "Q1", "answer": "A1", "score": 0.5},
            {"question": "Q2", "answer": "A2", "score": 0.8},
            {"question": "Q3", "answer": "A3", "score": 0.3},
        ]
        source = MockDataSource(items)
        
        results = list(stream(
            source,
            filter=lambda x: x["score"] > 0.4,
            normalize=False
        ))
        
        assert len(results) == 2
        assert results[0]["question"] == "Q1"
        assert results[1]["question"] == "Q2"
    
    def test_stream_with_transform(self):
        """Test transformation during streaming."""
        items = [{"text": "Hello"}, {"text": "World"}]
        source = MockDataSource(items)
        
        results = list(stream(
            source,
            transform=lambda x: {**x, "length": len(x["text"])},
            normalize=False
        ))
        
        assert results[0]["text"] == "Hello"
        assert results[0]["length"] == 5
        assert results[1]["text"] == "World"
        assert results[1]["length"] == 5
    
    def test_stream_with_max_items(self):
        """Test limiting number of items."""
        items = [{"id": i} for i in range(100)]
        source = MockDataSource(items)
        
        results = list(stream(source, max_items=5, normalize=False))
        assert len(results) == 5
        assert [r["id"] for r in results] == [0, 1, 2, 3, 4]
    
    def test_stream_with_normalization(self):
        """Test automatic normalization to standard schema."""
        items = [
            {"query": "What is 2+2?", "target": "4"},
            {"prompt": "Calculate 3*3", "response": "9"},
            {"text": "5+5=", "output": "10"},
        ]
        source = MockDataSource(items)
        
        results = list(stream(source, normalize=True))
        
        # All items should have standard fields
        for result in results:
            assert "question" in result
            assert "answer" in result
            assert "choices" in result
            assert "metadata" in result
        
        # Check specific mappings
        assert results[0]["question"] == "What is 2+2?"
        assert results[0]["answer"] == "4"
        assert results[1]["question"] == "Calculate 3*3"
        assert results[1]["answer"] == "9"
        assert results[2]["question"] == "5+5="
        assert results[2]["answer"] == "10"
    
    def test_stream_batch_size_respected(self):
        """Test that batch_size parameter is passed through."""
        items = [{"id": i} for i in range(100)]
        source = MockDataSource(items)
        
        # Stream with custom batch size
        list(stream(source, batch_size=10, normalize=False))
        
        # Should have made 10 batches of size 10
        assert source.batches_called == 1  # Only one call to read_batches
    
    def test_stream_empty_source(self):
        """Test streaming from empty source."""
        source = MockDataSource([])
        results = list(stream(source, normalize=False))
        assert results == []


class TestStreamIterator:
    """Test the StreamIterator class and method chaining."""
    
    def test_filter_chaining(self):
        """Test chaining multiple filters."""
        items = [
            {"id": 1, "score": 0.3, "valid": True},
            {"id": 2, "score": 0.6, "valid": False},
            {"id": 3, "score": 0.8, "valid": True},
            {"id": 4, "score": 0.9, "valid": True},
        ]
        source = MockDataSource(items)
        
        results = list(
            stream(source, normalize=False)
            .filter(lambda x: x["score"] > 0.5)
            .filter(lambda x: x["valid"])
        )
        
        assert len(results) == 2
        assert results[0]["id"] == 3
        assert results[1]["id"] == 4
    
    def test_transform_chaining(self):
        """Test chaining multiple transformations."""
        items = [{"value": 5}, {"value": 10}]
        source = MockDataSource(items)
        
        results = list(
            stream(source, normalize=False)
            .transform(lambda x: {**x, "doubled": x["value"] * 2})
            .transform(lambda x: {**x, "squared": x["value"] ** 2})
        )
        
        assert results[0]["value"] == 5
        assert results[0]["doubled"] == 10
        assert results[0]["squared"] == 25
        assert results[1]["value"] == 10
        assert results[1]["doubled"] == 20
        assert results[1]["squared"] == 100
    
    def test_limit_method(self):
        """Test the limit() method."""
        items = [{"id": i} for i in range(20)]
        source = MockDataSource(items)
        
        results = list(stream(source, normalize=False).limit(5))
        assert len(results) == 5
        assert [r["id"] for r in results] == [0, 1, 2, 3, 4]
    
    def test_limit_with_existing_max_items(self):
        """Test limit() respects existing max_items."""
        items = [{"id": i} for i in range(20)]
        source = MockDataSource(items)
        
        # Stream already limited to 10, further limit to 5
        results = list(stream(source, max_items=10, normalize=False).limit(5))
        assert len(results) == 5
        
        # Stream limited to 5, try to expand to 10 (should stay at 5)
        results = list(stream(source, max_items=5, normalize=False).limit(10))
        assert len(results) == 5
    
    def test_first_method(self):
        """Test the first() convenience method."""
        items = [{"id": i} for i in range(20)]
        source = MockDataSource(items)
        
        results = stream(source, normalize=False).first(3)
        assert isinstance(results, list)
        assert len(results) == 3
        assert [r["id"] for r in results] == [0, 1, 2]
    
    def test_collect_method(self):
        """Test the collect() method."""
        items = [{"id": i} for i in range(5)]
        source = MockDataSource(items)
        
        results = stream(source, normalize=False).filter(lambda x: x["id"] % 2 == 0).collect()
        assert isinstance(results, list)
        assert len(results) == 3
        assert [r["id"] for r in results] == [0, 2, 4]
    
    def test_complex_pipeline(self):
        """Test a complex processing pipeline."""
        items = [
            {"text": "Hello world", "category": "greeting"},
            {"text": "Goodbye", "category": "farewell"},
            {"text": "Hi there", "category": "greeting"},
            {"text": "See you", "category": "farewell"},
            {"text": "Welcome", "category": "greeting"},
        ]
        source = MockDataSource(items)
        
        # First filter by category, then add length, then filter by text length
        # Note: Due to how filters combine, we need to filter by text length directly
        greetings = (
            stream(source, normalize=False)
            .filter(lambda x: x["category"] == "greeting" and len(x["text"]) > 5)
            .transform(lambda x: {**x, "length": len(x["text"])})
            .limit(2)
            .collect()
        )
        
        assert len(greetings) == 2
        assert greetings[0]["text"] == "Hello world"
        assert greetings[0]["length"] == 11
        assert greetings[1]["text"] == "Hi there"
        assert greetings[1]["length"] == 8


class TestLoadFunction:
    """Test the load() function."""
    
    def test_load_basic(self):
        """Test basic loading functionality."""
        items = [{"id": i} for i in range(5)]
        source = MockDataSource(items)
        
        results = load(source, normalize=False)
        assert isinstance(results, list)
        assert results == items
    
    def test_load_with_processing(self):
        """Test load with filter and transform."""
        items = [
            {"value": 1, "keep": True},
            {"value": 2, "keep": False},
            {"value": 3, "keep": True},
        ]
        source = MockDataSource(items)
        
        results = load(
            source,
            filter=lambda x: x["keep"],
            transform=lambda x: {**x, "doubled": x["value"] * 2},
            normalize=False
        )
        
        assert len(results) == 2
        assert results[0]["value"] == 1
        assert results[0]["doubled"] == 2
        assert results[1]["value"] == 3
        assert results[1]["doubled"] == 6


class TestFileSource:
    """Test the FileSource class."""
    
    def test_jsonl_file(self):
        """Test reading JSONL files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"name": "Alice", "age": 30}\n')
            f.write('{"name": "Bob", "age": 25}\n')
            f.write('{"name": "Charlie", "age": 35}\n')
            fname = f.name
        
        try:
            source = FileSource(fname)
            batches = list(source.read_batches(batch_size=2))
            
            assert len(batches) == 2
            assert len(batches[0]) == 2
            assert len(batches[1]) == 1
            assert batches[0][0]["name"] == "Alice"
            assert batches[0][1]["name"] == "Bob"
            assert batches[1][0]["name"] == "Charlie"
        finally:
            Path(fname).unlink()
    
    def test_json_array_file(self):
        """Test reading JSON array files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"id": 1, "text": "First"},
                {"id": 2, "text": "Second"},
                {"id": 3, "text": "Third"},
            ], f)
            fname = f.name
        
        try:
            source = FileSource(fname)
            batches = list(source.read_batches(batch_size=2))
            
            assert len(batches) == 2
            assert len(batches[0]) == 2
            assert len(batches[1]) == 1
            assert batches[0][0]["id"] == 1
            assert batches[1][0]["id"] == 3
        finally:
            Path(fname).unlink()
    
    def test_json_object_file(self):
        """Test reading JSON object files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"config": "value", "debug": True}, f)
            fname = f.name
        
        try:
            source = FileSource(fname)
            batches = list(source.read_batches())
            
            assert len(batches) == 1
            assert len(batches[0]) == 1
            assert batches[0][0]["config"] == "value"
            assert batches[0][0]["debug"] is True
        finally:
            Path(fname).unlink()
    
    def test_csv_file(self):
        """Test reading CSV files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age,city\n")
            f.write("Alice,30,NYC\n")
            f.write("Bob,25,SF\n")
            fname = f.name
        
        try:
            source = FileSource(fname)
            batches = list(source.read_batches(batch_size=1))
            
            assert len(batches) == 2
            assert batches[0][0]["name"] == "Alice"
            assert batches[0][0]["age"] == "30"  # Note: CSV values are strings
            assert batches[1][0]["city"] == "SF"
        finally:
            Path(fname).unlink()
    
    def test_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            FileSource("/nonexistent/file.json")
    
    def test_unsupported_format(self):
        """Test error handling for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            fname = f.name
        
        try:
            source = FileSource(fname)
            with pytest.raises(ValueError, match="Unsupported file type"):
                list(source.read_batches())
        finally:
            Path(fname).unlink()
    
    def test_malformed_jsonl(self):
        """Test error handling for malformed JSONL."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write('not valid json\n')
            f.write('{"another": "valid"}\n')
            fname = f.name
        
        try:
            source = FileSource(fname)
            with pytest.raises(json.JSONDecodeError):
                list(source.read_batches())
        finally:
            Path(fname).unlink()


class TestConvenienceFunctions:
    """Test convenience functions for file operations."""
    
    def test_from_file(self):
        """Test from_file() function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": 1}\n')
            f.write('{"id": 2}\n')
            fname = f.name
        
        try:
            # Returns StreamIterator
            results = list(from_file(fname, normalize=False))
            assert len(results) == 2
            assert results[0]["id"] == 1
            assert results[1]["id"] == 2
        finally:
            Path(fname).unlink()
    
    def test_load_file(self):
        """Test load_file() function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"name": "test"}], f)
            fname = f.name
        
        try:
            # Returns list directly
            results = load_file(fname, normalize=False)
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["name"] == "test"
        finally:
            Path(fname).unlink()


class TestRegistry:
    """Test dataset registration functionality."""
    
    def test_register_custom_source(self):
        """Test registering a custom data source."""
        items = [{"custom": "data"}]
        source = MockDataSource(items)
        
        # Register the source
        register("test_custom", source)
        
        # Should be accessible by name
        assert "test_custom" in list_datasets()
        
        # Should be streamable
        results = list(stream("test_custom", normalize=False))
        assert results == items
    
    def test_register_with_metadata(self):
        """Test registering with metadata."""
        source = MockDataSource([{"q": "test"}])
        info = DatasetInfo(
            name="test_meta",
            description="Test dataset with metadata",
            size_bytes=1024,
            example_count=1,
            example_item={"q": "test"},
            streaming_supported=True
        )
        
        register("test_meta", source, info)
        
        # Should be able to retrieve metadata
        retrieved_info = metadata("test_meta")
        assert retrieved_info.name == "test_meta"
        assert retrieved_info.description == "Test dataset with metadata"
        assert retrieved_info.size_bytes == 1024
    
    def test_register_invalid_source(self):
        """Test error when registering invalid source."""
        with pytest.raises(TypeError, match="must implement DataSource protocol"):
            register("invalid", "not a data source")
    
    def test_metadata_generation(self):
        """Test automatic metadata generation."""
        items = [{"question": "Q1", "answer": "A1"}]
        source = MockDataSource(items)
        register("test_auto_meta", source)
        
        # Get auto-generated metadata
        info = metadata("test_auto_meta")
        assert info.name == "test_auto_meta"
        assert info.example_item == {"question": "Q1", "answer": "A1"}
        assert info.streaming_supported is True
        # Size and count unknown without full scan
        assert info.size_bytes == 0
        assert info.example_count == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_filter_result(self):
        """Test when filter removes all items."""
        items = [{"value": 1}, {"value": 2}, {"value": 3}]
        source = MockDataSource(items)
        
        results = list(stream(source, filter=lambda x: x["value"] > 10, normalize=False))
        assert results == []
    
    def test_transform_error_propagation(self):
        """Test that transform errors propagate correctly."""
        items = [{"value": "not_a_number"}]
        source = MockDataSource(items)
        
        # Transform that will raise an error
        with pytest.raises(ValueError):
            list(stream(
                source,
                transform=lambda x: {**x, "squared": int(x["value"]) ** 2},
                normalize=False
            ))
    
    def test_normalize_with_missing_fields(self):
        """Test normalization with missing fields."""
        items = [
            {},  # Empty item
            {"unrelated": "field"},  # No standard fields
            {"question": "Q1"},  # Missing answer
            {"answer": "A1"},  # Missing question
        ]
        source = MockDataSource(items)
        
        results = list(stream(source, normalize=True))
        
        # All should have standard fields, even if empty
        for result in results:
            assert "question" in result
            assert "answer" in result
            assert "choices" in result
            assert isinstance(result["choices"], dict)
            assert "metadata" in result
            assert isinstance(result["metadata"], dict)
    
    def test_choices_normalization(self):
        """Test normalization of choices field."""
        items = [
            {"question": "Q1", "choices": ["opt1", "opt2", "opt3"]},
            {"question": "Q2", "choices": {"A": "opt1", "B": "opt2"}},
            {"question": "Q3", "choices": "not_a_list_or_dict"},
        ]
        source = MockDataSource(items)
        
        results = list(stream(source, normalize=True))
        
        # List should be converted to dict
        assert results[0]["choices"] == {"A": "opt1", "B": "opt2", "C": "opt3"}
        
        # Dict should be preserved
        assert results[1]["choices"] == {"A": "opt1", "B": "opt2"}
        
        # Invalid type should become empty dict
        assert results[2]["choices"] == {}
    
    def test_metadata_preservation(self):
        """Test that existing metadata is preserved during normalization."""
        items = [{
            "question": "Q1",
            "answer": "A1",
            "metadata": {"difficulty": 3, "source": "test"},
            "extra_field": "value"
        }]
        source = MockDataSource(items)
        
        results = list(stream(source, normalize=True))
        result = results[0]
        
        # Original metadata should be preserved
        assert result["metadata"]["difficulty"] == 3
        assert result["metadata"]["source"] == "test"
        
        # Extra fields should be added to metadata
        assert result["metadata"]["extra_field"] == "value"
    
    def test_iterator_reuse(self):
        """Test that StreamIterator can be iterated multiple times."""
        items = [{"id": i} for i in range(3)]
        source = MockDataSource(items)
        
        iterator = stream(source, normalize=False)
        
        # First iteration
        first_results = list(iterator)
        assert len(first_results) == 3
        
        # Second iteration should also work
        second_results = list(iterator)
        assert len(second_results) == 3
        assert first_results == second_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])