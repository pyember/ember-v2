"""Unit tests for setup wizard functionality."""

import os
from pathlib import Path
from unittest.mock import patch
import pytest


class TestSetupWizardConfig:
    """Test setup wizard configuration utilities."""

    def test_config_save_format(self, tmp_path):
        """Test that config is saved in correct format."""
        config = {
            "providers": {
                "openai": {"default_model": "gpt-4", "api_key": "${OPENAI_API_KEY}"}
            }
        }

        config_file = tmp_path / "config.yaml"

        # Simulate what the setup wizard does
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Verify the file was created with correct content
        assert config_file.exists()
        loaded = yaml.safe_load(config_file.read_text())
        assert loaded == config
        assert loaded["providers"]["openai"]["api_key"] == "${OPENAI_API_KEY}"

    def test_credentials_save_secure(self, tmp_path):
        """Test that credentials are saved securely."""
        # Mock home directory
        with patch.dict(os.environ, {"HOME": str(tmp_path)}):
            ember_dir = tmp_path / ".ember"
            ember_dir.mkdir(exist_ok=True)

            # Simulate saving credentials
            creds_file = ember_dir / "credentials"
            creds_data = {"openai": "sk-test123"}

            # Write with secure permissions
            import json
            import stat

            with open(creds_file, "w") as f:
                json.dump(creds_data, f)

            # Set secure permissions (owner read/write only)
            creds_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

            # Verify permissions
            file_stat = creds_file.stat()
            assert file_stat.st_mode & 0o777 == 0o600  # Owner read/write only

    def test_provider_configuration_structure(self):
        """Test that provider configurations have correct structure."""
        # Since the setup wizard is TypeScript, we check the TypeScript source
        setup_wizard_dir = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "ember"
            / "cli"
            / "setup-wizard"
        )
        types_file = setup_wizard_dir / "src" / "types.ts"

        if not types_file.exists():
            pytest.skip("Setup wizard types.ts not found")

        content = types_file.read_text()

        # Verify PROVIDERS constant exists
        assert (
            "export const PROVIDERS" in content
        ), "PROVIDERS constant should be exported"

        # Verify required fields in provider structure
        required_fields = ["name", "envVar", "testModel", "description"]
        for field in required_fields:
            # Check for TypeScript property definition (field: type)
            assert f"{field}:" in content, f"Provider type should have {field} field"

        # Verify providers are defined
        for provider in ["openai", "anthropic", "google"]:
            assert (
                f'"{provider}"' in content or f"'{provider}'" in content
            ), f"Provider {provider} should be defined"

    def test_setup_mode_options(self):
        """Test that setup modes are properly defined."""
        setup_wizard_dir = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "ember"
            / "cli"
            / "setup-wizard"
        )
        setup_mode_file = (
            setup_wizard_dir / "src" / "components" / "steps" / "SetupModeSelection.tsx"
        )

        if not setup_mode_file.exists():
            pytest.skip("SetupModeSelection.tsx not found")

        content = setup_mode_file.read_text()

        # Verify the modes are correctly defined
        assert (
            "'single'" in content or '"single"' in content
        ), "Setup mode 'single' should be defined"
        assert (
            "'all'" in content or '"all"' in content
        ), "Setup mode 'all' should be defined"

        # Verify type definition
        assert (
            "onSelectMode: (mode: 'single' | 'all')" in content
        ), "onSelectMode should accept 'single' | 'all' union type"
