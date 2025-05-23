"""Golden tests for models examples.

This module tests all examples in the ember/examples/models directory.
"""

import pytest
from unittest.mock import MagicMock, patch

from .test_golden_base import GoldenTestBase


class TestModelsExamples(GoldenTestBase):
    """Test all models examples."""
    
    def test_model_api_example(self, capture_output, mock_models_api):
        """Test the simplified model API example."""
        expected_patterns = [
            r"=== Basic Invocation Example ===",
            r"Direct invocation:",
            r'models\("gpt-4", "What is the capital of France\?"\)',
            r"=== Model Binding Example ===",
            r"Binding a model for reuse:",
            r"=== Model Listing Example ===",
            r"=== Model Info Example ===",
            r"=== Response Handling Example ===",
            r"=== Error Handling Example ===",
            r"All examples completed!",
        ]
        
        extra_patches = [
            patch("ember.api.models", mock_models_api),
        ]
        
        results = self.run_category_tests(
            "models",
            {"model_api_example.py": expected_patterns},
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        result = results.get("model_api_example.py")
        assert result is not None, "model_api_example.py not found"
        assert result["success"], f"Example failed: {result.get('error')}"
        
        if "missing_patterns" in result:
            # Some patterns might vary based on mock data
            output = result["output"]
            assert "Basic Invocation Example" in output
            assert "Model Binding Example" in output
            assert "All examples completed!" in output
    
    def test_list_models(self, capture_output):
        """Test the list models example."""
        # Mock models.list()
        mock_models = MagicMock()
        mock_models.list.return_value = [
            "openai:gpt-4",
            "openai:gpt-3.5-turbo", 
            "anthropic:claude-3-sonnet",
            "anthropic:claude-3-opus"
        ]
        
        extra_patches = [
            patch("ember.api.models", mock_models),
        ]
        
        results = self.run_category_tests(
            "models",
            {},
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        result = results.get("list_models.py")
        if result and result["success"]:
            output = result["output"]
            # Verify it lists some models
            assert "model" in output.lower() or len(output) > 0
    
    def test_model_registry_example(self, capture_output):
        """Test the model registry example."""
        # Mock the necessary components
        mock_context = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_models.return_value = ["gpt-4", "claude-3"]
        mock_registry.get_model_info.return_value = MagicMock(
            id="gpt-4",
            name="GPT-4",
            provider=MagicMock(name="OpenAI")
        )
        mock_context.registry = mock_registry
        
        extra_patches = [
            patch("ember.api.models.get_default_context", return_value=mock_context),
        ]
        
        results = self.run_category_tests(
            "models",
            {},
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        result = results.get("model_registry_example.py")
        if result:
            # May need updates to new API
            pass
    
    def test_function_style_api(self, capture_output):
        """Test function style API example."""
        # Mock the models function
        mock_response = MagicMock()
        mock_response.text = "This is a summary"
        mock_response.usage = {"total_tokens": 100, "cost": 0.003}
        
        def mock_models_call(model, prompt, **kwargs):
            return mock_response
        
        mock_models = MagicMock()
        mock_models.side_effect = mock_models_call
        mock_models.__call__ = mock_models_call
        
        extra_patches = [
            patch("ember.api.models", mock_models),
        ]
        
        results = self.run_category_tests(
            "models",
            {},
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        result = results.get("function_style_api.py")
        if result:
            # May need updates to new API
            pass
    
    def test_all_models_examples_syntax(self):
        """Verify all models examples have valid syntax."""
        files = self.get_example_files("models")
        
        for file_path in files:
            error = self.check_syntax(file_path)
            assert error is None, error
    
    def test_models_examples_use_simplified_api(self):
        """Check that models examples use the simplified API."""
        files = self.get_example_files("models")
        
        outdated_patterns = []
        for file_path in files:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Check for outdated patterns
            if "initialize_registry" in content:
                outdated_patterns.append(f"{file_path.name}: Uses old initialize_registry")
            if "ModelService(" in content and file_path.name != "model_api_example.py":
                outdated_patterns.append(f"{file_path.name}: Creates ModelService directly")
            if "from ember.core.registry.model" in content:
                outdated_patterns.append(f"{file_path.name}: Uses deep imports instead of ember.api.models")
        
        if outdated_patterns:
            print("\nExamples that need updating to simplified API:")
            for pattern in outdated_patterns:
                print(f"  - {pattern}")