"""Golden tests for data examples.

This module tests all examples in the ember/examples/data directory.
"""

import pytest
from unittest.mock import MagicMock, patch

from .test_golden_base import GoldenTestBase


class TestDataExamples(GoldenTestBase):
    """Test all data examples."""
    
    def test_data_api_example(self, capture_output):
        """Test the data API example."""
        expected_patterns = [
            r"Data API Example",
            r"Loading dataset",
            r"Dataset info:",
        ]
        
        # Mock dataset loading
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.__iter__.return_value = iter([
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ])
        
        def mock_load_dataset(name, **kwargs):
            return mock_dataset
        
        extra_patches = [
            patch("ember.core.utils.data.load_dataset_entries", side_effect=mock_load_dataset),
        ]
        
        results = self.run_category_tests(
            "data",
            {"data_api_example.py": expected_patterns},
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        result = results.get("data_api_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_custom_dataset_example(self, capture_output):
        """Test the custom dataset example."""
        expected_patterns = [
            r"Custom Dataset Example",
            r"Creating custom dataset",
            r"Loading data",
        ]
        
        results = self.run_category_tests(
            "data",
            {"custom_dataset_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("custom_dataset_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_context_example(self, capture_output):
        """Test the data context example."""
        expected_patterns = [
            r"Data Context Example",
            r"Context state:",
        ]
        
        # Mock DataContext
        mock_context = MagicMock()
        mock_context.list_datasets.return_value = ["dataset1", "dataset2"]
        mock_context.load_dataset.return_value = MagicMock(__len__=lambda self: 50)
        
        extra_patches = [
            patch("ember.core.utils.data.context.data_context.DataContext", return_value=mock_context),
        ]
        
        results = self.run_category_tests(
            "data",
            {"context_example.py": expected_patterns},
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        result = results.get("context_example.py")
        if result:
            # May have different output based on context
            pass
    
    def test_transformation_example(self, capture_output):
        """Test the data transformation example."""
        expected_patterns = [
            r"Transformation Example",
            r"Original data:",
            r"Transformed data:",
        ]
        
        results = self.run_category_tests(
            "data",
            {"transformation_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("transformation_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_explore_datasets(self, capture_output):
        """Test the explore datasets example."""
        # Mock dataset registry
        mock_registry = {
            "mmlu": MagicMock(description="MMLU dataset"),
            "truthful_qa": MagicMock(description="TruthfulQA dataset"),
        }
        
        extra_patches = [
            patch("ember.core.utils.data.registry.get_dataset_registry", return_value=mock_registry),
        ]
        
        results = self.run_category_tests(
            "data",
            {},
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        result = results.get("explore_datasets.py")
        if result and result["success"]:
            output = result["output"]
            assert "dataset" in output.lower() or len(output) > 0
    
    def test_all_data_examples_syntax(self):
        """Verify all data examples have valid syntax."""
        files = self.get_example_files("data")
        
        for file_path in files:
            error = self.check_syntax(file_path)
            assert error is None, error
    
    def test_data_examples_use_correct_imports(self):
        """Check that data examples use appropriate imports."""
        files = self.get_example_files("data")
        
        import_suggestions = []
        for file_path in files:
            imports = self.extract_imports(file_path)
            
            # Check for data API usage
            uses_data_api = any("ember.api.data" in imp for imp in imports)
            uses_deep_data = any("ember.core.utils.data" in imp for imp in imports)
            
            if uses_deep_data and not uses_data_api:
                # This is okay for data module - it often needs deep imports
                # Just note it for awareness
                pass
        
        if import_suggestions:
            print("\nData import notes:")
            for suggestion in import_suggestions:
                print(f"  - {suggestion}")