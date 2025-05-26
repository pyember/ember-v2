"""Golden tests for XCS examples.

This module tests all examples in the ember/examples/xcs directory.
"""

import pytest
from unittest.mock import MagicMock, patch

from .test_golden_base import GoldenTestBase


class TestXCSExamples(GoldenTestBase):
    """Test all XCS examples."""
    
    def test_jit_example(self, capture_output):
        """Test the JIT compilation example."""
        expected_patterns = [
            r"JIT Example",
            r"Normal execution:",
            r"JIT execution:",
            r"Results match:"]
        
        results = self.run_category_tests(
            "xcs",
            {"jit_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("jit_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_example_simplified_xcs(self, capture_output):
        """Test the simplified XCS imports example."""
        expected_patterns = [
            r"Simplified XCS Example",
            r"Using jit:",
            r"Using transforms:"]
        
        results = self.run_category_tests(
            "xcs",
            {"example_simplified_xcs.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("example_simplified_xcs.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_auto_graph_example(self, capture_output):
        """Test the autograph example."""
        expected_patterns = [
            r"AutoGraph Example",
            r"Creating graph",
            r"Execution result:"]
        
        results = self.run_category_tests(
            "xcs",
            {"auto_graph_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("auto_graph_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_auto_graph_simplified(self, capture_output):
        """Test the simplified autograph example."""
        results = self.run_category_tests(
            "xcs",
            {},
            capture_output=capture_output
        )
        
        result = results.get("auto_graph_simplified.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_enhanced_jit_example(self, capture_output):
        """Test the enhanced JIT example."""
        expected_patterns = [
            r"Enhanced JIT",
            r"Performance comparison:"]
        
        results = self.run_category_tests(
            "xcs",
            {"enhanced_jit_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("enhanced_jit_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_simple_autograph_example(self, capture_output):
        """Test the simple autograph example."""
        results = self.run_category_tests(
            "xcs",
            {},
            capture_output=capture_output
        )
        
        result = results.get("simple_autograph_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_transforms_integration_example(self, capture_output):
        """Test the transforms integration example."""
        expected_patterns = [
            r"Transforms Integration",
            r"vmap example:",
            r"pmap example:"]
        
        results = self.run_category_tests(
            "xcs",
            {"transforms_integration_example.py": expected_patterns},
            capture_output=capture_output
        )
        
        result = results.get("transforms_integration_example.py")
        if result:
            assert result["success"], f"Example failed: {result.get('error')}"
    
    def test_all_xcs_examples_syntax(self):
        """Verify all XCS examples have valid syntax."""
        files = self.get_example_files("xcs")
        
        for file_path in files:
            error = self.check_syntax(file_path)
            assert error is None, error
    
    def test_xcs_examples_use_simplified_imports(self):
        """Check that XCS examples use ember.api.xcs imports."""
        files = self.get_example_files("xcs")
        
        good_patterns = []
        for file_path in files:
            imports = self.extract_imports(file_path)
            
            # Check for simplified XCS imports
            if any("ember.api.xcs" in imp for imp in imports):
                good_patterns.append(f"{file_path.name}: Uses ember.api.xcs ✓")
            elif any("ember.xcs" in imp for imp in imports):
                good_patterns.append(f"{file_path.name}: Consider using ember.api.xcs")
        
        if good_patterns:
            print("\nXCS import patterns:")
            for pattern in good_patterns[:10]:
                print(f"  {pattern}")