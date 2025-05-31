"""Golden tests for error handling examples (07_error_handling)."""

import pytest
from unittest.mock import MagicMock, patch, Mock

from .test_golden_base import GoldenTestBase


class TestErrorHandling(GoldenTestBase):
    """Test the error handling examples."""
    
    def test_robust_patterns(self, capture_output):
        """Test the robust error handling patterns example."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "07_error_handling" / "robust_patterns.py"
        
        # Mock the models API with various failure scenarios
        call_count = {"count": 0}
        
        def mock_model_call(model_name, prompt, *args, **kwargs):
            call_count["count"] += 1
            mock_response = MagicMock()
            
            # Simulate different responses and failures
            if "classify sentiment" in prompt.lower():
                mock_response.text = "positive"
            elif "translate" in prompt.lower():
                # Simulate failures for retry mechanism
                if call_count["count"] % 2 == 1:
                    raise ConnectionError("Network timeout")
                mock_response.text = "Bonjour le monde"
            elif "analyze" in prompt.lower():
                mock_response.text = "Analysis complete"
            else:
                mock_response.text = "Generic response"
                
            return mock_response
        
        # Mock random for circuit breaker demo
        random_values = [0.8, 0.9, 0.1, 0.8, 0.9, 0.2]  # Control failure pattern
        random_counter = {"index": 0}
        
        def mock_random():
            val = random_values[random_counter["index"] % len(random_values)]
            random_counter["index"] += 1
            return val
        
        # Mock environment to simulate API key presence
        mock_env = {"OPENAI_API_KEY": "test-key"}
        
        extra_patches = [
            patch("ember.api.models", side_effect=mock_model_call),
            patch("time.sleep", return_value=None),  # Skip sleep delays
            patch("time.time", return_value=1000),   # Fixed time for circuit breaker
            patch("random.random", side_effect=mock_random),
            patch("os.environ.get", side_effect=lambda key, default=None: mock_env.get(key, default))
        ]
        
        result = self.run_example_with_mocks(
            file_path,
            capture_output=capture_output,
            extra_patches=extra_patches
        )
        
        assert result["success"], f"Example failed: {result.get('error')}"
        
        # Check all error handling patterns are demonstrated
        output = result["output"]
        assert "Basic Error Handling" in output
        assert "Retry Mechanism" in output
        assert "Fallback Strategies" in output
        assert "Input Validation" in output
        assert "Circuit Breaker Pattern" in output
        assert "Error Aggregation and Reporting" in output
        
        # Check best practices are mentioned
        assert "Best Practices Summary" in output
    
    def test_error_handling_patterns(self):
        """Verify error handling examples follow best practices."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "07_error_handling" / "robust_patterns.py"
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check for key error handling patterns
        assert "try:" in content and "except" in content, "Should use try-except blocks"
        assert "retry" in content.lower(), "Should demonstrate retry logic"
        assert "fallback" in content.lower(), "Should show fallback strategies"
        assert "validate" in content or "validation" in content, "Should validate inputs"
        assert "circuit breaker" in content.lower(), "Should show circuit breaker pattern"
        
        # Check for structured error responses
        assert '"success":' in content, "Should return structured error responses"
        assert '"error":' in content, "Should include error messages"
        
        # Check for proper error types
        assert "Exception" in content, "Should handle exceptions"
        assert "ConnectionError" in content, "Should handle specific error types"