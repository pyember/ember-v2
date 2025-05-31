"""Golden tests for v2 examples.

This module tests the new v2 operator and module examples that showcase
the simplified APIs after the extensive refactoring.
"""

import pytest
from unittest.mock import MagicMock, patch

from .test_golden_base import GoldenTestBase


class TestV2Examples(GoldenTestBase):
    """Test the new v2 examples showcasing simplified APIs."""
    
    def test_module_v2_examples(self, capture_output):
        """Test the module v2 examples."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "module_v2_examples.py"
        
        # Mock the models API - it's a callable with instance() method
        mock_model_response = MagicMock(text="positive")
        mock_model = MagicMock(return_value=mock_model_response)
        mock_model.instance = MagicMock(return_value=mock_model)
        
        extra_patches = [
            patch("ember.api.models", mock_model)
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        # The file defines classes and functions but doesn't have a main()
        # Just check that it imports successfully
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check that it uses the new module system
        assert any("module_v2" in imp for imp in result["imports"])
        # Check that key classes are imported
        assert any("EmberModule" in imp or "ember.core.module_v2" in imp for imp in result["imports"])
    
    def test_operators_v2_examples(self, capture_output):
        """Test the operators v2 examples."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "operators_v2_examples.py"
        
        # Mock the models API
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(text="positive")
        mock_model.instance = MagicMock(return_value=mock_model)
        
        # Mock ensemble components
        mock_ensemble = MagicMock()
        mock_ensemble.return_value = ["story1", "story2", "story3"]
        
        extra_patches = [
            patch("ember.api.models", mock_model),
            patch("ember.core.operators_v2.ensemble.ensemble", mock_ensemble),
            patch("ember.core.operators_v2.selectors.most_common", return_value="story1"),
            patch("ember.core.operators_v2.selectors.best_of", return_value="story2")
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        # Check that it imports successfully
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check that it uses the new operators v2 system
        assert any("operators_v2" in imp for imp in result["imports"])
    
    def test_v2_examples_patterns(self):
        """Verify v2 examples follow best practices."""
        from pathlib import Path
        
        # Check module_v2_examples.py
        module_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "module_v2_examples.py"
        with open(module_path, "r") as f:
            content = f.read()
        
        # Should demonstrate key patterns
        assert "EmberModule" in content, "Should show EmberModule usage"
        assert "SignatureOperator" in content, "Should show SignatureOperator pattern"
        assert "@jit" in content or "jit(" in content, "Should demonstrate JIT optimization"
        assert "vmap" in content, "Should show batch processing with vmap"
        
        # Check operators_v2_examples.py
        ops_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "operators_v2_examples.py"
        with open(ops_path, "r") as f:
            content = f.read()
        
        # Should demonstrate simplified patterns
        assert "def " in content and "operator" not in content.lower() or "Any function is an operator" in content, \
               "Should show that any function can be an operator"
        assert "ensemble" in content, "Should demonstrate ensemble patterns"
        assert "jit(" in content, "Should show JIT usage"
        assert "vmap(" in content, "Should show batch processing"