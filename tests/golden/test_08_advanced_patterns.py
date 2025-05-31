"""Golden tests for advanced patterns examples (08_advanced_patterns)."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from .test_golden_base import GoldenTestBase


class TestAdvancedPatterns(GoldenTestBase):
    """Test the advanced patterns examples."""
    
    def test_advanced_techniques(self, capture_output):
        """Test the advanced techniques example."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "08_advanced_patterns" / "advanced_techniques.py"
        
        # Mock the models API with contextual responses
        def mock_model_call(model_name, prompt, *args, **kwargs):
            mock_response = MagicMock()
            
            # Provide appropriate responses based on prompt content
            if "analyze sentiment" in prompt.lower():
                mock_response.text = "positive"
            elif "extract entities" in prompt.lower():
                mock_response.text = "AI, work, life"
            elif "summarize" in prompt.lower():
                mock_response.text = "AI transforms work and life"
            elif "keywords" in prompt.lower():
                mock_response.text = "AI, transformation, technology"
            elif "conversation history" in prompt.lower():
                mock_response.text = "Machine learning is a subset of AI that enables learning from data."
            elif "classify this query" in prompt.lower():
                if "derivative" in prompt:
                    mock_response.text = "math"
                elif "function" in prompt:
                    mock_response.text = "code"
                elif "haiku" in prompt:
                    mock_response.text = "creative"
                else:
                    mock_response.text = "general"
            elif "solve this math" in prompt.lower():
                mock_response.text = "The derivative is 2x + 3"
            elif "write python code" in prompt.lower():
                mock_response.text = "def reverse_string(s): return s[::-1]"
            elif "write creatively" in prompt.lower():
                mock_response.text = "Spring arrives softly / Cherry blossoms dance in wind / Nature awakens"
            elif "valid?" in prompt.lower() or "check if" in prompt.lower():
                mock_response.text = "yes"
            elif "classify sentiment" in prompt.lower() and "learn from" in prompt.lower():
                # Adaptive classifier - learn from corrections
                if "it's okay" in prompt.lower():
                    mock_response.text = "neutral"  # Learned from feedback
                else:
                    mock_response.text = "positive"
            else:
                mock_response.text = "Generic response"
                
            return mock_response
        
        # Mock model instance creation
        mock_instance = MagicMock(side_effect=mock_model_call)
        
        # Create a mock models object that acts as both callable and has instance method
        mock_models = MagicMock(side_effect=mock_model_call)
        mock_models.instance = MagicMock(return_value=mock_instance)
        
        # Mock environment to simulate API key presence
        mock_env = {"OPENAI_API_KEY": "test-key"}
        
        extra_patches = [
            patch("ember.api.models", mock_models),
            patch("time.time", return_value=1000),
            patch("time.sleep", return_value=None),
            patch("os.environ.get", side_effect=lambda key, default=None: mock_env.get(key, default))
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check all advanced patterns are demonstrated
        output = result["output"]
        assert "Streaming Responses" in output
        assert "State Management" in output
        assert "Dynamic Routing" in output
        assert "Hierarchical Processing" in output
        assert "Meta-Programming Patterns" in output
        assert "Adaptive System" in output
        
        # Check summary is included
        assert "Advanced Patterns Summary" in output
    
    def test_advanced_pattern_quality(self):
        """Verify advanced examples demonstrate sophisticated patterns."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "08_advanced_patterns" / "advanced_techniques.py"
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check for sophisticated patterns
        assert "Generator[" in content, "Should use generators for streaming"
        assert "@dataclass" in content, "Should use dataclasses for state"
        assert "class" in content, "Should demonstrate OOP patterns"
        assert "yield" in content, "Should use yield for streaming"
        
        # Check for dynamic behavior
        assert "dynamically" in content.lower() or "dynamic" in content.lower()
        assert "adaptive" in content.lower() or "adapt" in content.lower()
        
        # Check for advanced Python features
        assert "typing" in content, "Should use type hints"
        assert "Dict[" in content or "List[" in content, "Should use generic types"
        assert "@jit" in content or "jit(" in content, "Should optimize performance"
        assert "vmap(" in content, "Should use batch processing"
        
        # Check for meta-programming
        assert "Callable" in content, "Should work with callable types"
        assert "create_" in content and "return" in content, "Should create functions dynamically"