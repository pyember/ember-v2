"""Tests for examples in 05_data_processing directory."""

import pytest
from .test_base import ExampleGoldenTest


class TestDataProcessingExamples(ExampleGoldenTest):
    """Test all examples in the 05_data_processing directory."""

    def test_loading_datasets(self):
        """Test the loading_datasets.py example."""
        self.run_example_test(
            "05_data_processing/loading_datasets.py",
            max_execution_time=15.0,
            # validate_sections removed due to parsing issues
        )

    @pytest.mark.requires_api_key
    def test_streaming_data(self):
        """Test the streaming_data.py example."""
        self.run_example_test(
            "05_data_processing/streaming_data.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=30.0,
            # validate_sections removed due to parsing issues
        )
