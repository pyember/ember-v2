"""Golden tests for 05_data_processing examples."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "05_data_processing"


class TestDataProcessingExamples:
    """Test suite for data processing examples."""
    
    def test_loading_datasets(self):
        """Test that loading_datasets.py runs successfully."""
        script_path = EXAMPLES_DIR / "loading_datasets.py"
        
        if not script_path.exists():
            pytest.skip("loading_datasets.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check it ran successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Check expected output
        assert "Working with Datasets" in result.stdout
        assert "Creating Simple Datasets" in result.stdout
        assert "Dataset Processing with Operators" in result.stdout
        assert "Streaming Large Datasets" in result.stdout
        assert "Filtering and Transformation" in result.stdout
        assert "Complete Data Processing Pipeline" in result.stdout
        assert "✅ Key Takeaways" in result.stdout
    
    def test_streaming_data(self):
        """Test that streaming_data.py runs successfully."""
        script_path = EXAMPLES_DIR / "streaming_data.py"
        
        if not script_path.exists():
            pytest.skip("streaming_data.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"