"""Example runner with cost simulation and validation.

Ensures examples can be run safely without incurring costs during
development and testing.
"""

import ast
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from unittest.mock import Mock, patch

from ember.api.models import Response


class MockModelBinding:
    """Mock model for testing examples without API calls."""
    
    def __init__(self, model_id: str, **params):
        self.model_id = model_id
        self.params = params
        self.call_count = 0
        self.total_tokens = 0
        
        # Predefined responses for common prompts
        self.responses = {
            "capital of france": "Paris",
            "2+2": "4", 
            "hello": "Hello! How can I help you?",
            "sentiment": "positive",
            "three laws": "1. A robot may not harm humans\n2. A robot must obey humans\n3. A robot must protect itself"
        }
    
    def __call__(self, prompt: str, **kwargs) -> Response:
        """Simulate model call."""
        self.call_count += 1
        
        # Simple response selection
        prompt_lower = prompt.lower()
        response_text = "This is a mock response."
        
        for key, value in self.responses.items():
            if key in prompt_lower:
                response_text = value
                break
        
        # Simulate token usage
        prompt_tokens = len(prompt.split()) * 2
        completion_tokens = len(response_text.split()) * 2
        self.total_tokens += prompt_tokens + completion_tokens
        
        # Create mock response
        mock_response = Mock(spec=Response)
        mock_response.text = response_text
        mock_response.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": (prompt_tokens + completion_tokens) * 0.00001
        }
        mock_response.model_id = self.model_id
        
        return mock_response


class ExampleRunner:
    """Run examples with proper setup and monitoring."""
    
    def __init__(self, mock_models: bool = True):
        self.mock_models = mock_models
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "duration": 0.0
        }
    
    def run(self, example_path: str, capture_output: bool = True) -> Dict[str, Any]:
        """Run an example and return statistics."""
        path = Path(example_path)
        if not path.exists():
            raise FileNotFoundError(f"Example not found: {example_path}")
        
        # Reset stats
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0, 
            "total_cost": 0.0,
            "duration": 0.0
        }
        
        # Load module
        spec = importlib.util.spec_from_file_location("example", path)
        module = importlib.util.module_from_spec(spec)
        
        # Set up mocking if enabled
        if self.mock_models:
            mock_bindings = {}
            
            def mock_models_call(model_id: str, prompt: str, **kwargs):
                """Mock the models() direct call."""
                if model_id not in mock_bindings:
                    mock_bindings[model_id] = MockModelBinding(model_id)
                return mock_bindings[model_id](prompt, **kwargs)
            
            def mock_models_instance(model_id: str, **kwargs):
                """Mock the models.instance() call."""
                if model_id not in mock_bindings:
                    mock_bindings[model_id] = MockModelBinding(model_id, **kwargs)
                return mock_bindings[model_id]
            
            # Patch both models() and models.instance()
            with patch('ember.api.models', side_effect=mock_models_call) as mock:
                mock.instance = mock_models_instance
                
                # Run example
                start_time = time.time()
                try:
                    spec.loader.exec_module(module)
                    if hasattr(module, 'main'):
                        module.main()
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "stats": self.stats
                    }
                finally:
                    self.stats["duration"] = time.time() - start_time
                    
                    # Collect stats from mocks
                    for model in mock_bindings.values():
                        self.stats["total_calls"] += model.call_count
                        self.stats["total_tokens"] += model.total_tokens
                    
                    # Estimate cost
                    self.stats["total_cost"] = self.stats["total_tokens"] * 0.00001
        
        else:
            # Run with real models (be careful!)
            start_time = time.time()
            try:
                spec.loader.exec_module(module)
                if hasattr(module, 'main'):
                    module.main()
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "stats": self.stats
                }
            finally:
                self.stats["duration"] = time.time() - start_time
        
        return {
            "success": True,
            "stats": self.stats
        }
    
    def validate_example(self, example_path: str) -> Dict[str, Any]:
        """Validate an example meets quality criteria."""
        path = Path(example_path)
        
        with open(path, 'r') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        # Validation checks
        issues = []
        
        # Check for docstring
        if not ast.get_docstring(tree):
            issues.append("Missing module docstring")
        
        # Check for main function
        has_main = any(
            isinstance(node, ast.FunctionDef) and node.name == 'main'
            for node in ast.walk(tree)
        )
        if not has_main:
            issues.append("Missing main() function")
        
        # Check for if __name__ == "__main__"
        has_main_guard = any(
            isinstance(node, ast.If) and 
            isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == '__name__'
            for node in ast.walk(tree)
        )
        if not has_main_guard:
            issues.append("Missing if __name__ == '__main__' guard")
        
        # Check line count
        lines = content.split('\n')
        if len(lines) > 200:
            issues.append(f"Too long: {len(lines)} lines (max 200)")
        
        # Check for error handling
        has_try_except = any(
            isinstance(node, ast.Try)
            for node in ast.walk(tree)
        )
        if not has_try_except:
            issues.append("No error handling (try/except) found")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "lines": len(lines),
                "functions": sum(1 for node in ast.walk(tree) 
                               if isinstance(node, ast.FunctionDef)),
                "classes": sum(1 for node in ast.walk(tree) 
                             if isinstance(node, ast.ClassDef))
            }
        }


def main():
    """Test the runner with examples."""
    runner = ExampleRunner(mock_models=True)
    
    # Find all examples
    examples_dir = Path(__file__).parent
    example_files = list(examples_dir.glob("*.py"))
    example_files.extend(examples_dir.glob("**/*.py"))
    
    # Filter out non-examples
    example_files = [
        f for f in example_files 
        if f.name not in ['runner.py', '__init__.py']
    ]
    
    print(f"Found {len(example_files)} examples\n")
    
    # Run each example
    for example in sorted(example_files):
        print(f"Running {example.name}...")
        
        # Validate first
        validation = runner.validate_example(example)
        if not validation["valid"]:
            print(f"  ❌ Validation failed: {validation['issues']}")
            continue
        
        # Run example
        result = runner.run(example)
        
        if result["success"]:
            stats = result["stats"]
            print(f"  ✅ Success!")
            print(f"     Calls: {stats['total_calls']}")
            print(f"     Tokens: {stats['total_tokens']}")
            print(f"     Cost: ${stats['total_cost']:.4f}")
            print(f"     Time: {stats['duration']:.2f}s")
        else:
            print(f"  ❌ Failed: {result['error']}")
        
        print()


if __name__ == "__main__":
    main()