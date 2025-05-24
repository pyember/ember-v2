"""Golden tests for 02_core_concepts examples."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "02_core_concepts"


class TestCoreConceptsExamples:
    """Test suite for core concepts examples."""
    
    def test_operators_basics(self):
        """Test that operators_basics.py runs successfully."""
        script_path = EXAMPLES_DIR / "operators_basics.py"
        
        if not script_path.exists():
            pytest.skip("operators_basics.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check it ran successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Check expected output
        assert "Understanding Operators" in result.stdout
        assert "Text Cleaner Results:" in result.stdout
        assert "Word Counter Results:" in result.stdout
        assert "Pipeline Results:" in result.stdout
        assert "Question Analysis:" in result.stdout
        assert "✅ Key Takeaways" in result.stdout
    
    def test_type_safety(self):
        """Test that type_safety.py runs successfully."""
        script_path = EXAMPLES_DIR / "type_safety.py"
        
        if not script_path.exists():
            pytest.skip("type_safety.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
    
    def test_context_management(self):
        """Test that context_management.py runs successfully."""
        script_path = EXAMPLES_DIR / "context_management.py"
        
        if not script_path.exists():
            pytest.skip("context_management.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
    
    def test_error_handling(self):
        """Test that error_handling.py runs successfully."""
        script_path = EXAMPLES_DIR / "error_handling.py"
        
        if not script_path.exists():
            pytest.skip("error_handling.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"