"""Base class for golden testing of Ember examples.

This module provides a comprehensive testing framework for validating
that all examples work correctly, produce consistent output, and
maintain their educational value.
"""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pytest


class ExampleGoldenTest:
    """Base class for testing Ember examples with golden outputs.

    This class provides utilities for:
    - Running examples in both simulated and real modes
    - Capturing and validating output against golden files
    - Enforcing performance bounds
    - Validating imports and dependencies
    """

    @property
    def examples_root(self) -> Path:
        """Get the root directory of examples."""
        return Path(__file__).parent.parent.parent / "examples"

    @property
    def golden_root(self) -> Path:
        """Get the root directory of golden outputs."""
        return Path(__file__).parent / "golden_outputs"

    @contextmanager
    def capture_output(self):
        """Context manager to capture stdout and stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            yield stdout_capture, stderr_capture
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def run_example(
        self,
        example_path: str,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> Tuple[str, str, float, int]:
        """Run an example and return output, error, duration, and return code.

        Args:
            example_path: Path relative to examples directory
            env_vars: Additional environment variables
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (stdout, stderr, duration, return_code)
        """
        full_path = self.examples_root / example_path
        if not full_path.exists():
            raise FileNotFoundError(f"Example not found: {full_path}")

        # Prepare environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Run the example
        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(full_path)],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
            duration = time.time() - start_time
            return result.stdout, result.stderr, duration, result.returncode
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            raise TimeoutError(f"Example {example_path} exceeded timeout of {timeout}s")

    def load_golden_output(
        self, example_path: str, mode: str = "simulated"
    ) -> Dict[str, Any]:
        """Load golden output for an example.

        Args:
            example_path: Path relative to examples directory
            mode: Either "simulated" or "real"

        Returns:
            Golden output data
        """
        golden_name = example_path.replace("/", "_").replace(".py", f"_{mode}.json")
        golden_path = self.golden_root / golden_name

        if not golden_path.exists():
            raise FileNotFoundError(f"Golden output not found: {golden_path}")

        with open(golden_path, "r") as f:
            return json.load(f)

    def save_golden_output(
        self,
        example_path: str,
        output: str,
        mode: str,
        duration: float,
        sections: Optional[List[Dict[str, str]]] = None,
        api_keys_required: Optional[List[str]] = None,
    ):
        """Save golden output for an example.

        Args:
            example_path: Path relative to examples directory
            output: Raw stdout from example
            mode: Either "simulated" or "real"
            duration: Execution duration
            sections: Parsed sections from output
            api_keys_required: List of required API keys
        """
        golden_name = example_path.replace("/", "_").replace(".py", f"_{mode}.json")
        golden_path = self.golden_root / golden_name

        # Ensure golden directory exists
        golden_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse sections if not provided
        if sections is None:
            sections = self.parse_output_sections(output)

        # Count metrics
        full_path = self.examples_root / example_path
        with open(full_path, "r") as f:
            content = f.read()
            lines_of_code = len([l for l in content.splitlines() if l.strip()])
            api_calls = (
                content.count("@model")
                + content.count(".call(")
                + content.count(".stream(")
            )

        golden_data = {
            "version": "1.0",
            "example": example_path,
            "execution_mode": mode,
            "sections": sections,
            "total_time": round(duration, 2),
            "api_keys_required": api_keys_required or [],
            "metrics": {"lines_of_code": lines_of_code, "api_calls": api_calls},
        }

        with open(golden_path, "w") as f:
            json.dump(golden_data, f, indent=2)

    def parse_output_sections(self, output: str) -> List[Dict[str, str]]:
        """Parse output into sections based on headers.

        Args:
            output: Raw stdout from example

        Returns:
            List of sections with headers and content
        """
        sections = []
        current_section = None
        current_content = []

        for line in output.splitlines():
            # Detect section headers (lines starting with "=" or "-" decorations)
            if line.strip() and (line.startswith("===") or line.startswith("---")):
                # Save previous section
                if current_section:
                    sections.append(
                        {
                            "header": current_section,
                            "output": "\n".join(current_content).strip(),
                        }
                    )
                # Start new section
                current_section = None
                current_content = []
            elif line.strip() and current_section is None and not line.startswith(" "):
                # This might be a section header
                current_section = line.strip()
            else:
                current_content.append(line)

        # Save last section
        if current_section or current_content:
            sections.append(
                {
                    "header": current_section or "Output",
                    "output": "\n".join(current_content).strip(),
                }
            )

        return sections

    def validate_golden(
        self, output: str, golden_data: Dict[str, Any], strict: bool = False
    ) -> List[str]:
        """Validate output against golden data.

        Args:
            output: Actual output from example
            golden_data: Expected golden data
            strict: If True, require exact matches

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Parse actual sections
        actual_sections = self.parse_output_sections(output)
        expected_sections = golden_data.get("sections", [])

        if strict:
            # Exact match required
            if len(actual_sections) != len(expected_sections):
                errors.append(
                    f"Section count mismatch: expected {len(expected_sections)}, "
                    f"got {len(actual_sections)}"
                )

            for i, (actual, expected) in enumerate(
                zip(actual_sections, expected_sections)
            ):
                if actual["header"] != expected["header"]:
                    errors.append(
                        f"Section {i} header mismatch: expected '{expected['header']}', "
                        f"got '{actual['header']}'"
                    )
                if actual["output"] != expected["output"]:
                    errors.append(f"Section {i} output mismatch")
        else:
            # Fuzzy validation - check key sections exist
            expected_headers = {s["header"] for s in expected_sections}
            actual_headers = {s["header"] for s in actual_sections}

            missing_headers = expected_headers - actual_headers
            if missing_headers:
                errors.append(f"Missing expected sections: {missing_headers}")

        return errors

    def run_example_test(
        self,
        example_path: str,
        requires_api_keys: Optional[List[str]] = None,
        max_execution_time: float = 30.0,
        validate_sections: Optional[List[str]] = None,
        skip_real_mode: bool = False,
    ):
        """Run a complete test of an example.

        Args:
            example_path: Path relative to examples directory
            requires_api_keys: List of required API keys
            max_execution_time: Maximum allowed execution time
            validate_sections: Specific sections that must be present
            skip_real_mode: Skip testing with real API keys
        """
        # Test 1: Validate imports
        self._test_imports(example_path)

        # Test 2: Run in simulated mode
        print(f"\nTesting {example_path} in simulated mode...")
        # Force empty API keys for simulated mode
        env_vars = {key: "" for key in (requires_api_keys or [])}
        # Also clear common API keys just in case
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
            env_vars[key] = ""

        stdout, stderr, duration, returncode = self.run_example(
            example_path, env_vars=env_vars, timeout=max_execution_time
        )

        # Check execution succeeded
        if returncode != 0:
            pytest.fail(
                f"Example failed with return code {returncode}\nStderr: {stderr}"
            )

        # Check performance bounds
        if duration > 10.0:  # Simulated mode should be reasonably fast
            pytest.fail(f"Simulated execution too slow: {duration:.2f}s > 10.0s")

        # Validate against golden output
        try:
            golden_data = self.load_golden_output(example_path, "simulated")
            errors = self.validate_golden(stdout, golden_data)
            if errors:
                pytest.fail(f"Golden validation failed:\n" + "\n".join(errors))
        except FileNotFoundError:
            # No golden file yet - create it
            print(f"Creating golden output for {example_path}")
            self.save_golden_output(
                example_path,
                stdout,
                "simulated",
                duration,
                api_keys_required=requires_api_keys,
            )

        # Test 3: Run in real mode (if API keys available and not skipped)
        if not skip_real_mode and requires_api_keys:
            # Check if we have ANY of the major API keys, not just the specific ones requested
            # Check both environment variables AND Ember's config
            from ember._internal.config.manager import ConfigManager
            from ember.core.config.loader import load_config
            
            # Load config file directly
            try:
                config_data = load_config()
                creds = config_data.get("credentials", {})
            except:
                creds = {}
            
            available_api_keys = {
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or creds.get("openai_api_key"),
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY") or creds.get("anthropic_api_key"), 
                "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY") or creds.get("google_api_key"),
                "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY") or creds.get("gemini_api_key"),
            }
            has_any_key = any(available_api_keys.values())
            
            if has_any_key:
                print(f"Testing {example_path} in real mode...")
                stdout, stderr, duration, returncode = self.run_example(
                    example_path, timeout=max_execution_time
                )

                if returncode != 0:
                    pytest.fail(
                        f"Real mode failed with return code {returncode}\nStderr: {stderr}"
                    )

                if duration > max_execution_time:
                    pytest.fail(
                        f"Real execution too slow: {duration:.2f}s > {max_execution_time}s"
                    )
            else:
                available_keys = [k for k, v in available_api_keys.items() if v]
                pytest.skip(
                    f"Skipping real mode test - no API keys found. Looking for any of: {list(available_api_keys.keys())}"
                )

        # Test 4: Validate specific sections if requested
        if validate_sections:
            sections = self.parse_output_sections(stdout)
            section_headers = {s["header"] for s in sections}

            missing = set(validate_sections) - section_headers
            if missing:
                pytest.fail(f"Required sections missing from output: {missing}")

        print(f"✓ {example_path} passed all tests")

    def _test_imports(self, example_path: str):
        """Test that all imports in an example work correctly.

        Args:
            example_path: Path relative to examples directory
        """
        full_path = self.examples_root / example_path

        # Use ast to parse imports safely
        import ast

        with open(full_path, "r") as f:
            tree = ast.parse(f.read(), filename=str(full_path))

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if module:
                        imports.append(f"{module}.{alias.name}")
                    else:
                        imports.append(alias.name)

        # Test each import
        for imp in imports:
            if imp.startswith("_shared"):
                # Skip checking shared imports as they use sys.path manipulation
                continue

            try:
                if "." in imp:
                    # Test from X import Y style
                    parts = imp.split(".")
                    module = ".".join(parts[:-1])
                    exec(f"from {module} import {parts[-1]}")
                else:
                    # Test import X style
                    exec(f"import {imp}")
            except ImportError as e:
                pytest.fail(f"Import failed in {example_path}: {imp} - {e}")

    def update_golden_from_current(self, example_path: str, mode: str = "simulated"):
        """Update golden output from current execution.

        Args:
            example_path: Path relative to examples directory
            mode: Either "simulated" or "real"
        """
        print(f"Updating golden output for {example_path} ({mode} mode)...")

        # Determine environment based on mode
        if mode == "simulated":
            # Clear API keys to force simulated mode
            env_vars = {
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
                "GOOGLE_API_KEY": "",
            }
        else:
            env_vars = None  # Use actual environment

        # Run the example
        stdout, stderr, duration, returncode = self.run_example(
            example_path, env_vars=env_vars, timeout=60.0
        )

        if returncode != 0:
            raise RuntimeError(f"Example failed: {stderr}")

        # Save as new golden output
        self.save_golden_output(example_path, stdout, mode, duration)
        print(f"✓ Updated golden output for {example_path}")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
