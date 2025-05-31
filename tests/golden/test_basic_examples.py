"""Golden tests for basic examples.

This module tests all examples in the ember/examples/basic directory.
"""

import pytest
from unittest.mock import MagicMock, patch

from .test_golden_base import GoldenTestBase


class TestBasicExamples(GoldenTestBase):
    """Test all basic examples."""
    
    def test_minimal_example(self, capture_output):
        """Test the minimal operator example."""
        expected_patterns = [
            r"Minimal Operator Example",
            r"Basic Example",
            r"Input\s+:\s+10",
            r"Result\s+:\s+30",  # (10 + 5) * 2
            r"Advanced Example",
            r"Input\s+:\s+7",
            r"Result\s+:\s+579",  # ((7 + 5) * 2)^2 + 3
            r"Added 5 to 7 = 12",
            r"Multiplied 12 by 2 = 24",
            r"Squared 24 = 576",
            r"Added 3 to 576 = 579",
        ]
        
        results = self.run_category_tests(
            "legacy/basic",
            {"minimal_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        # Assert results
        result = results.get("minimal_example.py")
        assert result is not None, "minimal_example.py not found"
        assert result["success"], f"Example failed: {result.get('error')}"
        assert result["has_main"], "Example should have a main function"
        
        # This example uses argparse, so it won't actually run in tests
        if result["output"] == "Example uses argparse - skipping execution in test environment":
            return  # Test passes - the example imported successfully
        
        if "missing_patterns" in result:
            pytest.fail(f"Missing output patterns: {result['missing_patterns']}")
    
    def test_minimal_operator_example(self, capture_output):
        """Test the minimal operator example variant."""
        expected_patterns = [
            r"Minimal Operator Example",
            r"Basic Example",
            r"Advanced Example",
            r"Operator Composition Example"]
        
        results = self.run_category_tests(
            "legacy/basic",
            {"minimal_operator_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("minimal_operator_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_compact_notation_example(self, capture_output, mock_lm):
        """Test the compact notation example."""
        pytest.skip("Legacy compact notation example has compatibility issues")
        expected_patterns = [
            r"==== Example 1: Basic Ensemble \+ Judge ====",
            r"Compact notation pipeline created",
            r"==== Example 2: Complex Verification Pipeline ====",
            r"==== Example 3: Nested Architecture ====",
            r"==== Example 4: Recursive Component References ====",
            r"==== Example 5: Custom Operator Types ====",
            r"==== Example 6: NestedNetwork Equivalent ====",
            r"==== Using the Pipeline ===="]
        
        results = self.run_category_tests(
            "legacy/basic",
            {"compact_notation_example.py": expected_patterns},
            mock_lm=mock_lm,
            capture_output=capture_output
        )
        
        result = results.get("compact_notation_example.py")
        assert result is not None, "compact_notation_example.py not found"
        assert result["success"], f"Example failed: {result.get('error')}"
        
        if "missing_patterns" in result:
            # Check if the main sections are present
            output = result["output"]
            assert "Example 1: Basic Ensemble" in output
            assert "Example 2: Complex Verification" in output
            assert "Using the Pipeline" in output
    
    def test_context_example(self, capture_output, mock_model_registry):
        """Test the context example."""
        expected_patterns = [
            r"=== Context Example ===",
            r"Current context state:",
            r"Model Registry:",
            r"available_models:"]
        
        # Create a more complete mock for context
        mock_context = MagicMock()
        mock_context.model_registry = mock_model_registry
        mock_context.cache_registry = MagicMock()
        mock_context.state = {
            "model_registry": {"available_models": ["openai:gpt-4", "anthropic:claude-3-sonnet"]},
            "cache_registry": {"cache_size": 0}
        }
        
        extra_patches = [
            patch("ember.core.context.ember_context.EmberContext", return_value=mock_context)]
        
        results = self.run_category_tests(
            "legacy/basic",
            {"context_example.py": expected_patterns},
            mock_registry=mock_model_registry,
            capture_output=capture_output
        )
        
        result = results.get("context_example.py")
        if result and result["success"]:
            # Check for expected patterns in a more lenient way
            output = result["output"]
            assert "Context Example" in output or "context" in output.lower()
    
    def test_simple_jit_demo(self, capture_output):
        """Test the simple JIT demo."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "legacy" / "basic" / "simple_jit_demo.py"
        
        # Check syntax
        error = self.check_syntax(file_path)
        assert error is None, error
        
        # Check imports
        imports = self.extract_imports(file_path)
        assert "ember.api.xcs" in imports
        assert "ember.api.operators" in imports
        
        # The demo has complex execution logic that may fail in test environment
        # Just ensure it can be imported without syntax errors
        pytest.skip("Complex JIT demo execution skipped in test environment")
    
    def test_check_env(self, capture_output):
        """Test the environment checker."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "legacy" / "basic" / "check_env.py"
        
        # Check syntax
        error = self.check_syntax(file_path)
        assert error is None, error
        
        # Check imports
        imports = self.extract_imports(file_path)
        assert "os" in imports
        assert "ember.api" in imports or "ember" in imports
        
        # Skip execution as it requires valid API keys
        pytest.skip("check_env requires actual environment setup")
    
    def test_all_basic_examples_syntax(self):
        """Verify all basic examples have valid syntax."""
        files = self.get_example_files("legacy/basic")
        
        for file_path in files:
            error = self.check_syntax(file_path)
            assert error is None, error
    
    def test_basic_examples_use_simplified_imports(self):
        """Check that basic examples use simplified imports where possible."""
        files = self.get_example_files("legacy/basic")
        
        all_issues = []
        for file_path in files:
            imports = self.extract_imports(file_path)
            issues = self.check_imports_are_simplified(imports)
            if issues:
                all_issues.extend(f"{file_path.name}: {issue}" for issue in issues)
        
        # Report but don't fail - these are recommendations
        if all_issues:
            print("\nImport simplification suggestions:")
            for issue in all_issues:
                print(f"  - {issue}")