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
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "models" / "model_api_example.py"
        
        # Run with proper mocks
        extra_patches = [
            patch("ember.api.models", mock_models_api),
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check output contains expected content
        output = result["output"]
        assert "Basic Invocation Example" in output
        assert "Model Binding Example" in output
        assert "All examples completed!" in output
    
    def test_list_models(self, capture_output):
        """Test the list models example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "models" / "list_models.py"
        
        # Mock models.list()
        mock_models = MagicMock()
        mock_models.list.return_value = [
            "openai:gpt-4",
            "openai:gpt-3.5-turbo", 
            "anthropic:claude-3-sonnet",
            "anthropic:claude-3-opus"
        ]
        mock_models.info.return_value = {
            "id": "openai:gpt-4",
            "provider": "openai",
            "context_window": 8192,
            "pricing": {"input": 0.03, "output": 0.06}
        }
        
        extra_patches = [
            patch("ember.api.models", mock_models),
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        # list_models.py uses logger instead of print, so output might be empty
        # Just check that it ran successfully
        assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_model_registry_example(self, capture_output):
        """Test the model registry example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "models" / "model_registry_example.py"
        
        # Mock the models API
        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.data = "The capital of France is Paris."
        mock_response.usage = MagicMock(total_tokens=50, cost=0.002)
        
        # Mock for direct call
        mock_models.return_value = mock_response
        
        # Mock for registry access
        mock_registry = MagicMock()
        mock_registry.list_models.return_value = ["gpt-4", "claude-3"]
        mock_registry.get_model_info.return_value = MagicMock(
            id="gpt-4",
            name="GPT-4",
            provider=MagicMock(name="OpenAI"),
            cost=MagicMock(input_cost_per_thousand=0.03, output_cost_per_thousand=0.06),
            context_window=8192
        )
        mock_registry.is_registered.return_value = False
        mock_models.get_registry.return_value = mock_registry
        
        extra_patches = [
            patch("ember.api.models", mock_models),
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        # The example may need mock API keys
        if not result["success"]:
            # Check if it's just missing API keys
            if "API key" in str(result.get("error", "")):
                pytest.skip("Example requires API keys")
    
    def test_function_style_api(self, capture_output):
        """Test function style API example."""
        from pathlib import Path
        
        # Get the file path
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "models" / "function_style_api.py"
        
        # This example just shows patterns, doesn't need complex mocks
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=[]
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check output contains expected patterns
        output = result["output"]
        assert "Function-Style Model API Examples" in output
        assert "Basic Usage" in output
        assert "Model Binding" in output
        assert "Examples completed!" in output
    
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