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
    
    lm = MagicMock(side_effect=mock_invoke)
    return lm


@pytest.fixture
def mock_environment():
    """Mock environment variables for API keys."""
    env_vars = {
        "OPENAI_API_KEY": "mock-openai-key",
        "ANTHROPIC_API_KEY": "mock-anthropic-key",
        "GOOGLE_API_KEY": "mock-google-key"
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
        "ember.api.models": ["model", "complete", "configure", "Response"],
        "ember.api.data": ["DataContext", "load_dataset_entries"],
        "ember.api.xcs": ["jit", "pmap", "vmap", "autograph"],
        "ember.api.operators": ["Operator", "Specification"],
    }


@pytest.fixture
def mock_models_api():
    """Mock the simplified models API."""
    # Create a mock response object
    class MockResponse:
        def __init__(self, text):
            self.text = text
            self.usage = {
                "total_tokens": 100,
                "prompt_tokens": 20,
                "completion_tokens": 80,
                "cost": 0.003
            }
            self.model = "mock-model"
            self.raw = {"data": text}
    
    # Mock the models() function
    def mock_models_call(model_id, prompt, **kwargs):
        if "capital" in prompt.lower():
            return MockResponse("The capital of France is Paris.")
        elif "quantum" in prompt.lower():
            return MockResponse("Quantum computing uses quantum bits (qubits).")
        else:
            return MockResponse(f"Mock response for {model_id}: {prompt[:50]}...")
    
    # Mock models.bind()
    def mock_bind(model_id, **params):
        bound = MagicMock()
        bound.model_id = model_id
        bound.params = params
        bound.__call__ = lambda prompt, **kw: mock_models_call(model_id, prompt, **{**params, **kw})
        bound.__repr__ = lambda: f"ModelBinding(model_id='{model_id}', params={params})"
        return bound
    
    # Mock models.list()
    def mock_list(provider=None):
        all_models = [
            "openai:gpt-4", "openai:gpt-3.5-turbo",
            "anthropic:claude-3-sonnet", "anthropic:claude-3-opus",
            "google:gemini-pro"
        ]
        if provider:
            return [m for m in all_models if m.startswith(provider)]
        return all_models
    
    # Mock models.info()
    def mock_info(model_id):
        return {
            "id": f"provider:{model_id}" if ":" not in model_id else model_id,
            "provider": model_id.split(":")[0] if ":" in model_id else "unknown",
            "context_window": 8192,
            "pricing": {"input": 0.01, "output": 0.03}
        }
    
    # Create the mock models module
    mock_models = MagicMock()
    mock_models.side_effect = mock_models_call
    mock_models.__call__ = mock_models_call
    mock_models.bind = mock_bind
    mock_models.list = mock_list
    mock_models.info = mock_info
    
    return mock_models


@pytest.fixture
def golden_output_dir(tmp_path):
    """Directory for storing golden output files."""
    output_dir = tmp_path / "golden_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir