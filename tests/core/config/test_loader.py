"""Tests for enhanced configuration loader."""

import json
import os
import tempfile
from pathlib import Path
import pytest
import yaml

from ember.core.config.loader import load_config, save_config, _resolve_env_vars


class TestConfigLoader:
    """Test suite for ConfigLoader."""
    
    def test_load_yaml_config(self, tmp_path):
        """Test loading YAML configuration."""
        config = {
            "version": "1.0",
            "model": "gpt-4",
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1"
                }
            }
        }
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))
        
        loaded = load_config(config_file)
        assert loaded == config
    
    def test_load_json_config(self, tmp_path):
        """Test loading JSON configuration."""
        config = {
            "version": "1.0",
            "model": "gpt-4",
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1"
                }
            }
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config, indent=2))
        
        loaded = load_config(config_file)
        assert loaded == config
    
    def test_load_extensionless_yaml(self, tmp_path):
        """Test loading YAML from extensionless file."""
        config = {"model": "gpt-4", "temperature": 0.7}
        
        config_file = tmp_path / "config"
        config_file.write_text(yaml.dump(config))
        
        loaded = load_config(config_file)
        assert loaded == config
    
    def test_load_extensionless_json(self, tmp_path):
        """Test loading JSON from extensionless file."""
        config = {"model": "gpt-4", "temperature": 0.7}
        
        config_file = tmp_path / "config"
        config_file.write_text(json.dumps(config))
        
        loaded = load_config(config_file)
        assert loaded == config
    
    def test_file_not_found(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
    
    def test_invalid_format(self, tmp_path):
        """Test handling of invalid file content."""
        config_file = tmp_path / "config"
        config_file.write_text("invalid: yaml: content:")
        
        with pytest.raises(ValueError):
            load_config(config_file)
    
    def test_empty_yaml(self, tmp_path):
        """Test loading empty YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        
        loaded = load_config(config_file)
        assert loaded == {}
    
    def test_save_yaml_config(self, tmp_path):
        """Test saving configuration as YAML."""
        config = {
            "version": "1.0",
            "providers": {
                "openai": {"api_key": "test"}
            }
        }
        
        config_file = tmp_path / "config.yaml"
        save_config(config, config_file)
        
        # Verify file was created
        assert config_file.exists()
        
        # Load and verify content
        loaded = yaml.safe_load(config_file.read_text())
        assert loaded == config
    
    def test_save_json_config(self, tmp_path):
        """Test saving configuration as JSON."""
        config = {
            "version": "1.0",
            "providers": {
                "openai": {"api_key": "test"}
            }
        }
        
        config_file = tmp_path / "config.json"
        save_config(config, config_file)
        
        # Verify file was created
        assert config_file.exists()
        
        # Load and verify content
        loaded = json.loads(config_file.read_text())
        assert loaded == config
    
    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories."""
        config = {"test": "value"}
        
        config_file = tmp_path / "nested" / "dir" / "config.yaml"
        save_config(config, config_file)
        
        assert config_file.exists()
        assert config_file.parent.exists()


class TestEnvironmentVariableResolution:
    """Test suite for environment variable resolution."""
    
    def test_resolve_simple_env_var(self, monkeypatch):
        """Test resolving simple environment variable."""
        monkeypatch.setenv("TEST_VAR", "test-value")
        
        config = {
            "api_key": "${TEST_VAR}",
            "other": "static"
        }
        
        resolved = _resolve_env_vars(config)
        assert resolved["api_key"] == "test-value"
        assert resolved["other"] == "static"
    
    def test_resolve_missing_env_var(self):
        """Test handling of missing environment variable."""
        config = {"api_key": "${MISSING_VAR}"}
        
        resolved = _resolve_env_vars(config)
        # Should keep original value if env var not found
        assert resolved["api_key"] == "${MISSING_VAR}"
    
    def test_resolve_nested_env_vars(self, monkeypatch):
        """Test resolving environment variables in nested structures."""
        monkeypatch.setenv("OPENAI_KEY", "openai-123")
        monkeypatch.setenv("ANTHROPIC_KEY", "anthropic-456")
        
        config = {
            "providers": {
                "openai": {
                    "api_key": "${OPENAI_KEY}",
                    "base_url": "https://api.openai.com/v1"
                },
                "anthropic": {
                    "api_key": "${ANTHROPIC_KEY}"
                }
            },
            "default": "openai"
        }
        
        resolved = _resolve_env_vars(config)
        assert resolved["providers"]["openai"]["api_key"] == "openai-123"
        assert resolved["providers"]["anthropic"]["api_key"] == "anthropic-456"
        assert resolved["providers"]["openai"]["base_url"] == "https://api.openai.com/v1"
        assert resolved["default"] == "openai"
    
    def test_resolve_env_vars_in_list(self, monkeypatch):
        """Test resolving environment variables in lists."""
        monkeypatch.setenv("VAR1", "value1")
        monkeypatch.setenv("VAR2", "value2")
        
        config = {
            "items": ["${VAR1}", "static", "${VAR2}", "${MISSING}"]
        }
        
        resolved = _resolve_env_vars(config)
        assert resolved["items"] == ["value1", "static", "value2", "${MISSING}"]
    
    def test_resolve_partial_env_var(self, monkeypatch):
        """Test that partial env var syntax is not resolved."""
        monkeypatch.setenv("TEST", "value")
        
        config = {
            "partial1": "prefix${TEST}",
            "partial2": "${TEST}suffix",
            "partial3": "prefix${TEST}suffix",
            "complete": "${TEST}"
        }
        
        resolved = _resolve_env_vars(config)
        # Only complete ${VAR} should be resolved
        assert resolved["partial1"] == "prefix${TEST}"
        assert resolved["partial2"] == "${TEST}suffix"
        assert resolved["partial3"] == "prefix${TEST}suffix"
        assert resolved["complete"] == "value"
    
    def test_load_config_with_env_vars(self, tmp_path, monkeypatch):
        """Test loading config file with environment variable resolution."""
        monkeypatch.setenv("API_KEY", "secret-key-123")
        
        config = {
            "providers": {
                "openai": {
                    "api_key": "${API_KEY}",
                    "model": "gpt-4"
                }
            }
        }
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))
        
        loaded = load_config(config_file)
        assert loaded["providers"]["openai"]["api_key"] == "secret-key-123"
        assert loaded["providers"]["openai"]["model"] == "gpt-4"
    
    def test_resolve_env_vars_function(self):
        """Test the _resolve_env_vars function with string inputs."""
        os.environ["TEST_RESOLVE"] = "resolved"
        
        # Test complete variable
        assert _resolve_env_vars("${TEST_RESOLVE}") == "resolved"
        
        # Test missing variable
        assert _resolve_env_vars("${MISSING_VAR}") == "${MISSING_VAR}"
        
        # Test non-env var strings
        assert _resolve_env_vars("regular string") == "regular string"
        assert _resolve_env_vars("${partial") == "${partial"
        assert _resolve_env_vars("partial}") == "partial}"
        
        # Clean up
        del os.environ["TEST_RESOLVE"]


class TestRealWorldScenarios:
    """Test real-world configuration scenarios."""
    
    def test_codex_style_config(self, tmp_path, monkeypatch):
        """Test loading Codex-style configuration."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-123456")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-789012")
        
        config = {
            "model": "o4-mini",
            "provider": "openai",
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                    "api_key": "${OPENAI_API_KEY}"
                },
                "anthropic": {
                    "name": "Anthropic",
                    "baseURL": "https://api.anthropic.com/v1",
                    "envKey": "ANTHROPIC_API_KEY",
                    "api_key": "${ANTHROPIC_API_KEY}"
                }
            }
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config, indent=2))
        
        loaded = load_config(config_file)
        assert loaded["providers"]["openai"]["api_key"] == "sk-123456"
        assert loaded["providers"]["anthropic"]["api_key"] == "ant-789012"
    
    def test_mixed_format_config(self, tmp_path, monkeypatch):
        """Test configuration with mixed static and env var values."""
        monkeypatch.setenv("PROD_KEY", "production-key")
        
        config = {
            "environments": {
                "dev": {
                    "api_key": "dev-static-key",
                    "url": "http://localhost:8000"
                },
                "prod": {
                    "api_key": "${PROD_KEY}",
                    "url": "https://api.production.com"
                }
            }
        }
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))
        
        loaded = load_config(config_file)
        assert loaded["environments"]["dev"]["api_key"] == "dev-static-key"
        assert loaded["environments"]["prod"]["api_key"] == "production-key"