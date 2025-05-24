"""Golden tests for 04_compound_ai examples."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "04_compound_ai"


class TestCompoundAIExamples:
    """Test suite for compound AI examples."""
    
    def test_simple_ensemble(self):
        """Test that simple_ensemble.py runs successfully."""
        script_path = EXAMPLES_DIR / "simple_ensemble.py"
        
        if not script_path.exists():
            pytest.skip("simple_ensemble.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check it ran successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Check expected output
        assert "Simple Ensemble System" in result.stdout
        assert "Sequential Expert Consultation" in result.stdout
        assert "Parallel Expert Consultation" in result.stdout
        assert "Building Consensus with Voting" in result.stdout
        assert "Ensemble Pipeline Results:" in result.stdout
        assert "✅ Key Takeaways" in result.stdout
    
    def test_judge_synthesis(self):
        """Test that judge_synthesis.py runs successfully."""
        script_path = EXAMPLES_DIR / "judge_synthesis.py"
        
        if not script_path.exists():
            pytest.skip("judge_synthesis.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"