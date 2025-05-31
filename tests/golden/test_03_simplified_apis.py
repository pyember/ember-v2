"""Golden tests for simplified API examples (03_simplified_apis).

Tests the new examples that showcase Ember's simplified APIs after the refactoring.
"""

import pytest
from unittest.mock import MagicMock, patch

from .test_golden_base import GoldenTestBase


class TestSimplifiedAPIs(GoldenTestBase):
    """Test the simplified API examples."""
    
    def test_zero_config_jit(self, capture_output):
        """Test the zero-configuration JIT example."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "03_simplified_apis" / "zero_config_jit.py"
        
        # Mock the models API and stats
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(text="positive sentiment")
        
        # Mock environment to simulate API key presence
        mock_env = {"OPENAI_API_KEY": "test-key"}
        
        extra_patches = [
            patch("ember.api.models", return_value=mock_model),
            patch("ember.api.xcs.get_jit_stats", return_value={"cache_hits": 1, "cache_misses": 1}),
            patch("time.time", side_effect=[0, 0.1, 0.1, 0.05]),  # Simulate timing
            patch("os.environ.get", side_effect=lambda key, default=None: mock_env.get(key, default))
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check key concepts are demonstrated
        output = result["output"]
        assert "Basic JIT Example" in output
        assert "Complex Function JIT" in output
        assert "Conditional Logic with JIT" in output
        
        # Verify it uses simplified APIs
        assert "ember.api.xcs" in result["imports"]
        
        # Check that JIT examples ran (either the word "jit" or "JIT" should appear)
        assert "jit" in output.lower() or "optimized" in output
    
    def test_natural_api_showcase(self, capture_output):
        """Test the natural API showcase example."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "03_simplified_apis" / "natural_api_showcase.py"
        
        # Mock the models API
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(text="high urgency")
        mock_model.instance = MagicMock(return_value=mock_model)
        
        # Mock data module
        mock_data = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_data.load = MagicMock(return_value=mock_dataset)
        
        # Mock environment to simulate API key presence
        mock_env = {"OPENAI_API_KEY": "test-key"}
        
        extra_patches = [
            patch("ember.api.models", mock_model),
            patch("ember.api.data", mock_data),
            patch("os.environ.get", side_effect=lambda key, default=None: mock_env.get(key, default))
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check key concepts
        output = result["output"]
        assert "Functions as Operators" in output
        assert "Natural Composition" in output
        assert "Dynamic Behavior" in output
        
        # Verify natural API patterns
        assert "ember.api" in result["imports"]
        assert "just write python" in output.lower()
    
    def test_simplified_workflows(self, capture_output):
        """Test the simplified workflows example."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "03_simplified_apis" / "simplified_workflows.py"
        
        # Mock the models API with appropriate responses
        def mock_model_call(model_name, prompt):
            response = MagicMock()
            
            # Provide contextual responses based on prompt content
            if "toxic" in prompt.lower():
                response.text = "NO, not toxic"
            elif "spam" in prompt.lower():
                response.text = "NO, confidence: 0.2"
            elif "PII" in prompt.lower():
                response.text = "NO PII found"
            elif "difficulty" in prompt.lower():
                response.text = "easy"
            elif "topic" in prompt.lower():
                response.text = "math"
            elif "explain" in prompt.lower():
                response.text = "Basic arithmetic addition"
            elif "relevance" in prompt.lower():
                response.text = "0.8"
            elif "answer" in prompt.lower() and "context" in prompt.lower():
                response.text = "The Eiffel Tower is 330 meters tall."
            else:
                response.text = "Generic response"
                
            return response
        
        # Mock environment to simulate API key presence
        mock_env = {"OPENAI_API_KEY": "test-key"}
        
        extra_patches = [
            patch("ember.api.models", side_effect=mock_model_call),
            patch("os.environ.get", side_effect=lambda key, default=None: mock_env.get(key, default))
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check all workflows are demonstrated
        output = result["output"]
        assert "Content Moderation Workflow" in output
        assert "Data Processing Workflow" in output
        assert "Evaluation Workflow" in output
        assert "RAG Workflow" in output
        
        # Verify simplified patterns
        assert "Complex workflows are just composed functions" in output
    
    def test_simplified_api_patterns(self):
        """Verify simplified API examples follow best practices."""
        from pathlib import Path
        import os
        
        examples_dir = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "03_simplified_apis"
        
        for example_file in examples_dir.glob("*.py"):
            if example_file.name == "__init__.py":
                continue
                
            with open(example_file, "r") as f:
                content = f.read()
            
            # Check for simplified patterns
            assert "ember.api" in content, f"{example_file.name} should use ember.api imports"
            assert "@jit" in content or "jit(" in content, f"{example_file.name} should demonstrate JIT"
            
            # Should NOT use complex patterns
            assert "BaseOperator" not in content, f"{example_file.name} should not require base classes"
            assert "execution_options" not in content, f"{example_file.name} should not use old APIs"
            
            # Should have clear examples
            assert "def example_" in content or "def main" in content, f"{example_file.name} should have example functions"
            
            # Should handle missing API keys gracefully
            assert "OPENAI_API_KEY" in content or "API key" in content, f"{example_file.name} should check for API keys"