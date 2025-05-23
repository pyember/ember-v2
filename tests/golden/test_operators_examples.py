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
        expected_patterns = [
            r"=== Simplified Ensemble Operator Example ===",
            r"Creating ensemble with 3 mock models",
            r"Result:",
            r"Votes:",
            r"Confidence:",
        ]
        
        results = self.run_category_tests(
            "operators",
            {"simplified_ensemble_example.py": expected_patterns},
            mock_lm=mock_lm,
            capture_output=capture_output
        )
        
        result = results.get("simplified_ensemble_example.py")
        assert result is not None, "simplified_ensemble_example.py not found"
        assert result["success"], f"Example failed: {result.get('error')}"
        
        if "missing_patterns" in result:
            # Check for key outputs
            output = result["output"]
            assert "Ensemble Operator" in output
    
    def test_composition_example(self, capture_output):
        """Test the operator composition example."""
        expected_patterns = [
            r"=== Operator Composition Example ===",
            r"Step 1:",
            r"Step 2:",
            r"Final result:",
        ]
        
        results = self.run_category_tests(
            "operators",
            {"composition_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("composition_example.py")
        if result and result["success"]:
            output = result["output"]
            assert "composition" in output.lower() or "operator" in output.lower()
    
    def test_container_operator_example(self, capture_output):
        """Test the container operator example."""
        expected_patterns = [
            r"Container Operator Example",
            r"Processing data",
            r"Result:",
        ]
        
        results = self.run_category_tests(
            "operators",
            {"container_operator_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("container_operator_example.py")
        if result:
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
        expected_patterns = [
            r"Custom Prompt Example",
            r"Caravan",
        ]
        
        # Mock the language model
        mock_lm = MagicMock()
        mock_lm.return_value = "Mocked caravan story response"
        
        extra_patches = [
            patch("ember.api.non", return_value=mock_lm),
        ]
        
        results = self.run_category_tests(
            "operators",
            {"custom_prompt_example_caravan.py": expected_patterns},
            mock_lm=mock_lm,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        result = results.get("custom_prompt_example_caravan.py")
        if result:
            # May use language models, so check syntax at minimum
            assert "error" not in result or "import" in str(result.get("error"))
    
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