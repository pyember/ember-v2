"""Base classes and utilities for golden tests.

This module provides the foundation for testing all Ember examples
to ensure they work correctly and produce expected outputs.
"""

import ast
import importlib.util
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest


class GoldenTestBase:
    """Base class for golden tests of Ember examples."""
    
    @staticmethod
    def get_example_files(category: str) -> List[Path]:
        """Get all Python example files in a category."""
        examples_dir = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / category
        if not examples_dir.exists():
            return []
        
        # Get all .py files except __init__.py
        files = [f for f in examples_dir.glob("*.py") if f.name != "__init__.py"]
        return sorted(files)
    
    @staticmethod
    def extract_imports(file_path: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())
        
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    for alias in node.names:
                        imports.add(f"{node.module}.{alias.name}")
        
        return imports
    
    @staticmethod
    def check_syntax(file_path: Path) -> Optional[str]:
        """Check if a Python file has valid syntax."""
        try:
            with open(file_path, "r") as f:
                compile(f.read(), str(file_path), "exec")
            return None
        except SyntaxError as e:
            return f"Syntax error in {file_path}: {e}"
    
    @staticmethod
    def run_example_with_mocks(
        file_path: Path,
        mock_registry=None,
        mock_lm=None,
        capture_output=None,
        extra_patches=None
    ) -> Dict[str, Any]:
        """Run an example file with appropriate mocks."""
        result = {
            "success": False,
            "output": "",
            "error": None,
            "imports": set(),
            "has_main": False
        }
        
        # Check syntax first
        syntax_error = GoldenTestBase.check_syntax(file_path)
        if syntax_error:
            result["error"] = syntax_error
            return result
        
        # Extract imports
        result["imports"] = GoldenTestBase.extract_imports(file_path)
        
        # Load the module
        spec = importlib.util.spec_from_file_location("example_module", file_path)
        if not spec or not spec.loader:
            result["error"] = f"Failed to load module spec for {file_path}"
            return result
            
        module = importlib.util.module_from_spec(spec)
        
        # Prepare patches
        patches = []
        
        # Mock model registry if needed
        if mock_registry and any("model" in imp for imp in result["imports"]):
            patches.extend([
                patch("ember.api.models.initialize_registry", return_value=mock_registry),
                patch("ember.core.registry.model.initialization.initialize_registry", return_value=mock_registry),
            ])
        
        # Mock language model if needed
        if mock_lm and any("non" in imp or "lm" in imp for imp in result["imports"]):
            patches.extend([
                patch("ember.api.non", return_value=mock_lm),
                patch("ember.core.non", return_value=mock_lm),
            ])
        
        # Add any extra patches
        if extra_patches:
            patches.extend(extra_patches)
        
        # Execute with mocks
        try:
            for p in patches:
                p.start()
            
            # Capture output if provided
            if capture_output:
                with capture_output() as capture:
                    spec.loader.exec_module(module)
                    
                    # Run main if it exists
                    if hasattr(module, "main"):
                        result["has_main"] = True
                        module.main()
                    
                    result["output"] = capture.get_stdout()
                    result["success"] = True
            else:
                spec.loader.exec_module(module)
                
                # Run main if it exists
                if hasattr(module, "main"):
                    result["has_main"] = True
                    module.main()
                
                result["success"] = True
                
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            
        finally:
            for p in patches:
                p.stop()
        
        return result
    
    @staticmethod
    def validate_output(output: str, expected_patterns: List[str]) -> List[str]:
        """Validate output contains expected patterns."""
        missing_patterns = []
        
        for pattern in expected_patterns:
            if not re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                missing_patterns.append(pattern)
        
        return missing_patterns
    
    @staticmethod
    def check_imports_are_simplified(imports: Set[str]) -> List[str]:
        """Check if imports use the simplified API where possible."""
        issues = []
        
        # Map of deep imports to their simplified equivalents
        deep_to_simplified = {
            "ember.core.registry.model": "ember.api.models",
            "ember.core.utils.data": "ember.api.data",
            "ember.core.registry.operator": "ember.api.operators",
            "ember.xcs": "ember.api.xcs",
        }
        
        for imp in imports:
            for deep_pattern, simplified in deep_to_simplified.items():
                if imp.startswith(deep_pattern) and simplified not in imports:
                    issues.append(f"Using deep import '{imp}' - consider '{simplified}'")
        
        return issues
    
    def run_category_tests(
        self,
        category: str,
        expected_outputs: Dict[str, List[str]],
        mock_registry=None,
        mock_lm=None,
        capture_output=None
    ):
        """Run tests for all examples in a category."""
        files = self.get_example_files(category)
        
        if not files:
            pytest.skip(f"No example files found in {category}")
        
        results = {}
        
        for file_path in files:
            file_name = file_path.name
            
            # Run the example
            result = self.run_example_with_mocks(
                file_path,
                mock_registry=mock_registry,
                mock_lm=mock_lm,
                capture_output=capture_output
            )
            
            # Check for simplified imports
            import_issues = self.check_imports_are_simplified(result["imports"])
            if import_issues:
                result["import_issues"] = import_issues
            
            # Validate output if patterns provided
            if file_name in expected_outputs and result["success"]:
                missing = self.validate_output(
                    result["output"],
                    expected_outputs[file_name]
                )
                if missing:
                    result["missing_patterns"] = missing
            
            results[file_name] = result
        
        return results