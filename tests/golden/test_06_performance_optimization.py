"""Golden tests for performance optimization examples (06_performance_optimization)."""

import pytest
from unittest.mock import MagicMock, patch
import time

from .test_golden_base import GoldenTestBase


class TestPerformanceOptimization(GoldenTestBase):
    """Test the performance optimization examples."""
    
    def test_optimization_techniques(self, capture_output):
        """Test the optimization techniques example."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "06_performance_optimization" / "optimization_techniques.py"
        
        # Mock the models API with timing simulation
        def mock_model_call(model_name, prompt):
            response = MagicMock()
            response.text = "positive sentiment"
            return response
        
        mock_model_call.return_value = MagicMock(text="positive sentiment")
        
        # Mock time.time to simulate performance improvements
        time_counter = {"index": 0}
        
        def mock_time():
            # Return incrementing time values
            val = time_counter["index"] * 0.1
            time_counter["index"] += 1
            return val
        
        # Mock environment to simulate API key presence
        mock_env = {"OPENAI_API_KEY": "test-key"}
        
        extra_patches = [
            patch("ember.api.models", side_effect=mock_model_call),
            patch("time.time", side_effect=mock_time),
            patch("time.sleep", return_value=None),  # Skip sleep delays
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
        assert "JIT Compilation Speedup" in output
        assert "Batch Processing with vmap" in output
        assert "Caching Pattern" in output
        assert "Optimized Pipeline" in output
        assert "Performance Optimization Tips" in output
    
    def test_performance_patterns(self):
        """Verify performance examples follow best practices."""
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "06_performance_optimization" / "optimization_techniques.py"
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check for key performance patterns
        assert "@jit" in content or "jit(" in content, "Should demonstrate JIT compilation"
        assert "vmap(" in content, "Should demonstrate batch processing"
        assert "cache" in content.lower(), "Should demonstrate caching"
        assert "time.time()" in content, "Should measure performance"
        
        # Check for optimization tips
        assert "Profile" in content or "profile" in content, "Should mention profiling"
        assert "Speedup" in content or "speedup" in content, "Should show performance gains"