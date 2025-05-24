"""Golden tests for 10_evaluation_suite examples."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "10_evaluation_suite"


class TestEvaluationSuiteExamples:
    """Test suite for evaluation examples."""
    
    def test_accuracy_evaluation(self):
        """Test that accuracy_evaluation.py runs successfully."""
        script_path = EXAMPLES_DIR / "accuracy_evaluation.py"
        
        if not script_path.exists():
            pytest.skip("accuracy_evaluation.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check it ran successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Check expected output
        assert "Accuracy Evaluation Framework" in result.stdout
        assert "Creating Evaluation Datasets" in result.stdout
        assert "Systems to Evaluate" in result.stdout
        assert "Evaluation Metrics" in result.stdout
        assert "Complete Evaluation Pipeline" in result.stdout
        assert "System Comparison" in result.stdout
        assert "✅ Key Takeaways" in result.stdout
    
    def test_consistency_testing(self):
        """Test that consistency_testing.py runs successfully."""
        script_path = EXAMPLES_DIR / "consistency_testing.py"
        
        if not script_path.exists():
            pytest.skip("consistency_testing.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
    
    def test_benchmark_harness(self):
        """Test that benchmark_harness.py runs successfully."""
        script_path = EXAMPLES_DIR / "benchmark_harness.py"
        
        if not script_path.exists():
            pytest.skip("benchmark_harness.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"