"""Golden tests for 09_practical_patterns examples."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "09_practical_patterns"


class TestPracticalPatternsExamples:
    """Test suite for practical pattern examples."""
    
    def test_rag_pattern(self):
        """Test that rag_pattern.py runs successfully."""
        script_path = EXAMPLES_DIR / "rag_pattern.py"
        
        if not script_path.exists():
            pytest.skip("rag_pattern.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check it ran successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Check expected output
        assert "RAG Pattern Implementation" in result.stdout
        assert "Document Processing" in result.stdout
        assert "Indexing with Simple Embeddings" in result.stdout
        assert "Semantic Retrieval" in result.stdout
        assert "Context-Aware Generation" in result.stdout
        assert "Complete RAG Pipeline" in result.stdout
        assert "✅ Key Takeaways" in result.stdout
    
    def test_chain_of_thought(self):
        """Test that chain_of_thought.py runs successfully."""
        script_path = EXAMPLES_DIR / "chain_of_thought.py"
        
        if not script_path.exists():
            pytest.skip("chain_of_thought.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
    
    def test_structured_output(self):
        """Test that structured_output.py runs successfully."""
        script_path = EXAMPLES_DIR / "structured_output.py"
        
        if not script_path.exists():
            pytest.skip("structured_output.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"