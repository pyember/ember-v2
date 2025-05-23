"""Pytest configuration for golden tests.

This module provides shared fixtures and utilities for testing Ember examples.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing without API calls."""
    registry = MagicMock()
    
    # Mock common models
    mock_models = {
        "openai:gpt-4": MagicMock(
            model_info=MagicMock(
                id="openai:gpt-4",
                name="GPT-4",
                provider=MagicMock(name="OpenAI"),
                cost=MagicMock(
                    input_cost_per_thousand=0.01,
                    output_cost_per_thousand=0.03
                )
            )
        ),
        "anthropic:claude-3-sonnet": MagicMock(
            model_info=MagicMock(
                id="anthropic:claude-3-sonnet",
                name="Claude 3 Sonnet",
                provider=MagicMock(name="Anthropic"),
                cost=MagicMock(
                    input_cost_per_thousand=0.003,
                    output_cost_per_thousand=0.015
                )
            )
        )
    }
    
    def get_model(model_id: str):
        return mock_models.get(model_id)
    
    def is_registered(model_id: str):
        return model_id in mock_models
    
    def list_models():
        return list(mock_models.keys())
    
    def get_model_info(model_id: str):
        model = mock_models.get(model_id)
        return model.model_info if model else None
    
    registry.get_model.side_effect = get_model
    registry.is_registered.side_effect = is_registered
    registry.list_models.return_value = list_models()
    registry.get_model_info.side_effect = get_model_info
    registry.register_model.return_value = None
    
    return registry


@pytest.fixture
def mock_lm():
    """Mock language model for testing."""
    def mock_invoke(prompt: str, **kwargs):
        # Return predictable responses based on prompt
        if "capital" in prompt.lower():
            return "The capital of France is Paris."
        elif "quantum" in prompt.lower():
            return "Quantum computing uses quantum bits (qubits) to process information."
        else:
            return f"Mock response to: {prompt[:50]}..."
    
    lm = MagicMock()
    lm.side_effect = mock_invoke
    lm.__call__.side_effect = mock_invoke
    
    return lm


@pytest.fixture
def mock_environment():
    """Mock environment variables for API keys."""
    env_vars = {
        "OPENAI_API_KEY": "mock-openai-key",
        "ANTHROPIC_API_KEY": "mock-anthropic-key"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def capture_output():
    """Fixture to capture stdout/stderr output."""
    import io
    import contextlib
    
    class OutputCapture:
        def __init__(self):
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()
            
        def __enter__(self):
            self._stdout_context = contextlib.redirect_stdout(self.stdout)
            self._stderr_context = contextlib.redirect_stderr(self.stderr)
            self._stdout_context.__enter__()
            self._stderr_context.__enter__()
            return self
            
        def __exit__(self, *args):
            self._stdout_context.__exit__(*args)
            self._stderr_context.__exit__(*args)
            
        def get_stdout(self) -> str:
            return self.stdout.getvalue()
            
        def get_stderr(self) -> str:
            return self.stderr.getvalue()
    
    return OutputCapture


@pytest.fixture
def example_imports():
    """Common imports for examples."""
    return {
        "ember.api": ["non", "models", "data", "operators", "xcs"],
        "ember.api.models": ["ModelService", "UsageService", "initialize_registry"],
        "ember.api.data": ["DataContext", "load_dataset_entries"],
        "ember.api.xcs": ["jit", "pmap", "vmap"],
    }


def run_example_main(module_path: str, mock_registry=None, mock_lm=None):
    """Helper to run an example's main function with mocks."""
    import importlib.util
    
    # Load the module
    spec = importlib.util.spec_from_file_location("example_module", module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Apply mocks if provided
    patches = []
    if mock_registry:
        patches.append(patch("ember.api.models.initialize_registry", return_value=mock_registry))
    if mock_lm:
        patches.append(patch("ember.api.non", return_value=mock_lm))
    
    # Execute with mocks
    try:
        for p in patches:
            p.start()
        
        spec.loader.exec_module(module)
        
        # Run main if it exists
        if hasattr(module, "main"):
            module.main()
            
    finally:
        for p in patches:
            p.stop()
    
    return module


@pytest.fixture
def golden_output_dir(tmp_path):
    """Directory for storing golden output files."""
    output_dir = tmp_path / "golden_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir