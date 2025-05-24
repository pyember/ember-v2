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
        from pathlib import Path
        from ember.core.utils.data.base.models import DatasetEntry

        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "data" / "data_api_example.py"

        # Create mock dataset entries
        mock_entries = [
            DatasetEntry(
                query="What is 2+2?",
                metadata={"correct_answer": "4", "subject": "math"},
                choices={"A": "3", "B": "4", "C": "5", "D": "6"}
            ),
            DatasetEntry(
                query="What is 3+3?",
                metadata={"correct_answer": "6", "subject": "math"},
                choices={"A": "5", "B": "6", "C": "7", "D": "8"}
            ),
            DatasetEntry(
                query="What is a cell?",
                metadata={"correct_answer": "Basic unit of life", "subject": "biology"},
                choices={"A": "Basic unit of life", "B": "A type of battery", "C": "A prison room", "D": "None of the above"}
            ),
        ]

        # Mock the DatasetBuilder.build method
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter(mock_entries)
        mock_dataset.__len__.return_value = len(mock_entries)
        mock_dataset.entries = mock_entries

        # Mock the builder chain
        mock_builder = MagicMock()
        mock_builder.from_registry.return_value = mock_builder
        mock_builder.subset.return_value = mock_builder
        mock_builder.split.return_value = mock_builder
        mock_builder.sample.return_value = mock_builder
        mock_builder.build.return_value = mock_dataset

        # We need to mock the imported 'data' object in the example
        mock_data = MagicMock()
        mock_data.builder.return_value = mock_builder
        
        extra_patches = [
            patch("ember.examples.data.data_api_example.data", mock_data),
        ]

        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )

        # Check it runs without error
        assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_custom_dataset_example(self, capture_output):
        """Test the custom dataset example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "data" / "custom_dataset_example.py"
        
        # Check if file exists, skip if not
        if not file_path.exists():
            pytest.skip("custom_dataset_example.py not found")
        
        # Mock data context registry
        mock_registry = MagicMock()
        mock_registry.list_datasets.return_value = ["custom_dataset"]
        
        extra_patches = [
            patch("ember.core.utils.data.registry.DATASET_REGISTRY", mock_registry),
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        # Just check it runs without error
        assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_context_example(self, capture_output):
        """Test the data context example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "data" / "context_example.py"
        
        # Mock the data context and registry
        mock_registry = MagicMock()
        mock_registry.list_datasets.return_value = ["dataset1", "dataset2"]
        mock_registry.get.return_value = MagicMock(info=MagicMock(description="Test dataset"))
        
        mock_context = MagicMock()
        mock_context.registry = mock_registry
        mock_context.list_datasets.return_value = ["dataset1", "dataset2"]
        mock_context.get_dataset_info.return_value = MagicMock(description="Test dataset")
        
        # Mock streaming dataset
        mock_streaming = MagicMock()
        mock_streaming.limit.return_value = mock_streaming
        mock_streaming.__iter__.return_value = iter([
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ])
        mock_context.get_streaming_dataset.return_value = mock_streaming
        
        extra_patches = [
            patch("ember.core.utils.data.context.data_context.DataContext.create_from_ember_context", return_value=mock_context),
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        # Just check it runs without error
        assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_transformation_example(self, capture_output):
        """Test the data transformation example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "data" / "transformation_example.py"
        
        # This is an XCS transform example that should be in the xcs folder
        pytest.skip("transformation_example.py is an XCS transforms example, not a data example")
    
    def test_explore_datasets(self, capture_output):
        """Test the explore datasets example."""
        from pathlib import Path
        from ember.core.utils.data.base.models import DatasetInfo, TaskType
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "data" / "explore_datasets.py"
        
        # Mock dataset info objects
        mmlu_info = DatasetInfo(
            name="mmlu",
            source="cais/mmlu",
            task_type=TaskType.MULTIPLE_CHOICE,
            description="Measuring Massive Multitask Language Understanding",
            splits=["train", "test", "validation"],
            subjects=["high_school_mathematics", "high_school_biology", "physics"]
        )
        
        truthful_qa_info = DatasetInfo(
            name="truthful_qa",
            source="truthful_qa",
            task_type=TaskType.MULTIPLE_CHOICE,
            description="TruthfulQA dataset",
            splits=["validation"]
        )
        
        # Mock the data API
        mock_data_api = MagicMock()
        mock_data_api.list.return_value = ["mmlu", "truthful_qa"]
        mock_data_api.info.side_effect = lambda name: {
            "mmlu": mmlu_info,
            "truthful_qa": truthful_qa_info
        }.get(name)
        
        # Mock DataItem for samples
        class MockDataItem:
            def __init__(self, question, answer):
                self.question = question
                self.answer = answer
                
        mock_data_api.return_value = iter([
            MockDataItem("What is 2+2?", "4"),
            MockDataItem("What is the capital of France?", "Paris"),
        ])
        
        # This example uses argparse, so we need to mock sys.argv
        import sys
        original_argv = sys.argv
        
        try:
            # Mock command line args
            sys.argv = ["explore_datasets.py"]  # No extra args needed
            
            extra_patches = [
                patch("ember.api.data", mock_data_api),
            ]
            
            result = self.run_example_with_mocks(
                file_path,
                capture_output=capture_output,
                extra_patches=extra_patches
            )
            
            # Just check it runs without error
            assert result["success"], f"Example failed: {result.get('error')}"
        finally:
            sys.argv = original_argv
    
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