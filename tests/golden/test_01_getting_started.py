"""Golden tests for 01_getting_started examples."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "src" / "ember" / "examples" / "01_getting_started"


class TestGettingStartedExamples:
    """Test suite for getting started examples."""
    
    def test_hello_world(self):
        """Test that hello_world.py runs successfully."""
        script_path = EXAMPLES_DIR / "hello_world.py"
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check it ran successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Check expected output
        assert "Ember package imported successfully" in result.stdout
        assert "Core APIs imported successfully" in result.stdout
        assert "Basic operator creation successful" in result.stdout
        assert "Hello, World! Welcome to Ember." in result.stdout
        assert "Congratulations!" in result.stdout
    
    @pytest.mark.skipif(
        not any(Path().glob("**/OPENAI_API_KEY*")),
        reason="Requires OpenAI API key"
    )
    def test_first_model_call(self):
        """Test that first_model_call.py runs successfully."""
        script_path = EXAMPLES_DIR / "first_model_call.py"
        
        # Check if script exists first
        if not script_path.exists():
            pytest.skip("first_model_call.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60  # Allow more time for API calls
        )
        
        # Check it ran successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Check for expected output patterns
        assert "Model response:" in result.stdout or "Response:" in result.stdout
    
    def test_model_comparison(self):
        """Test that model_comparison.py runs successfully."""
        script_path = EXAMPLES_DIR / "model_comparison.py"
        
        if not script_path.exists():
            pytest.skip("model_comparison.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=90
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
    
    def test_basic_prompt_engineering(self):
        """Test that basic_prompt_engineering.py runs successfully."""
        script_path = EXAMPLES_DIR / "basic_prompt_engineering.py"
        
        if not script_path.exists():
            pytest.skip("basic_prompt_engineering.py not yet implemented")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"


class TestExampleImports:
    """Test that examples can be imported without errors."""
    
    def test_can_import_hello_world(self):
        """Test that hello_world module can be imported."""
        # Add examples directory to path
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        
        try:
            # This should not raise any errors
            from ember.examples._01_getting_started import hello_world
            assert hasattr(hello_world, 'main')
        except ImportError:
            # Try alternative import
            spec = __import__('importlib.util').util.spec_from_file_location(
                "hello_world", 
                str(EXAMPLES_DIR / "hello_world.py")
            )
            module = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert hasattr(module, 'main')
        finally:
            # Clean up sys.path
            if str(EXAMPLES_DIR.parent) in sys.path:
                sys.path.remove(str(EXAMPLES_DIR.parent))