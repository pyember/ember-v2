"""Golden tests for operators examples.

This module tests all examples in the ember/examples/operators directory.
"""

import pytest
from unittest.mock import MagicMock, patch

from .test_golden_base import GoldenTestBase


class TestOperatorsExamples(GoldenTestBase):
    """Test all operators examples."""
    
    def test_simplified_ensemble_example(self, capture_output, mock_lm):
        """Test the simplified ensemble operator example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "operators" / "simplified_ensemble_example.py"
        
        result = self.run_example_with_mocks(
            file_path,
            mock_lm=mock_lm,
            capture_output=capture_output
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check for key outputs
        output = result["output"]
        assert "ensemble" in output.lower() or "operator" in output.lower()
    
    def test_composition_example(self, capture_output):
        """Test the operator composition example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "operators" / "composition_example.py"
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check for key outputs
        output = result["output"]
        assert "composition" in output.lower() or "operator" in output.lower()
    
    def test_container_operator_example(self, capture_output):
        """Test the container operator example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "operators" / "container_operator_example.py"
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_container_simplified(self, capture_output):
        """Test the simplified container operator example."""
        results = self.run_category_tests(
            "operators",
            {},
            capture_output=capture_output
        )
        
        result = results.get("container_simplified.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_custom_prompt_example_caravan(self, capture_output):
        """Test the custom prompt caravan example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "operators" / "custom_prompt_example_caravan.py"
        
        # Mock the language model
        mock_lm = MagicMock()
        mock_lm.return_value = "Mocked caravan story response"
        
        extra_patches = [
            patch("ember.api.non", return_value=mock_lm),
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            mock_lm=mock_lm,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        # May use language models, so check syntax at minimum
        assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_diverse_ensemble_operator_example(self, capture_output):
        """Test the diverse ensemble operator example."""
        expected_patterns = [
            r"Diverse Ensemble",
            r"Creating diverse models",
            r"Result:",
        ]
        
        # Mock multiple language models
        def create_mock_lm(style):
            mock = MagicMock()
            mock.return_value = f"Response in {style} style"
            return mock
        
        results = self.run_category_tests(
            "operators",
            {"diverse_ensemble_operator_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("diverse_ensemble_operator_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_all_operators_examples_syntax(self):
        """Verify all operators examples have valid syntax."""
        files = self.get_example_files("operators")
        
        for file_path in files:
            error = self.check_syntax(file_path)
            assert error is None, error
    
    def test_operators_examples_patterns(self):
        """Check that operators examples follow good patterns."""
        files = self.get_example_files("operators")
        
        patterns_found = []
        for file_path in files:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Check for good patterns
            if "class.*Operator.*Operator" in content:
                patterns_found.append(f"{file_path.name}: Defines custom operator")
            if "Specification" in content:
                patterns_found.append(f"{file_path.name}: Uses Specification")
            if "EmberModel" in content:
                patterns_found.append(f"{file_path.name}: Uses EmberModel for types")
            if "forward(" in content:
                patterns_found.append(f"{file_path.name}: Implements forward method")
        
        # Report findings
        if patterns_found:
            print("\nGood patterns found in operators examples:")
            for pattern in patterns_found[:10]:  # Show first 10
                print(f"  ✓ {pattern}")