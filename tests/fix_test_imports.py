#!/usr/bin/env python3
"""Fix test imports to use new API structure.

This script updates all test files to use the new simplified API structure
instead of the old registry-based imports.
"""

import re
from pathlib import Path

# Define import mappings
IMPORT_MAPPINGS = [
    # Model imports
    (
        r"from ember\.core\.registry\.model\.base\.schemas\.chat_schemas "
        r"import.*ChatRequest.*\n.*ChatResponse.*",
        "from ember.models.schemas import ChatResponse",
    ),
    (
        r"from ember\.core\.registry\.model\.base\.schemas\.chat_schemas import.*ChatRequest",
        "# ChatRequest removed - use dict or kwargs directly",
    ),
    (
        r"from ember\.core\.registry\.model\.base\.schemas\.chat_schemas import.*ChatResponse",
        "from ember.models.schemas import ChatResponse",
    ),
    (
        r"from ember\.core\.registry\.model\.providers\.base_provider import BaseProviderModel",
        "from ember.models.providers.base import BaseProvider",
    ),
    (
        r"from ember\.core\.registry\.model\.base\.registry\.model_registry import ModelRegistry",
        "from ember.models.registry import ModelRegistry",
    ),
    (
        r"from ember\.core\.registry\.model\.base\.services\.model_service import ModelService",
        "# ModelService integrated into ModelRegistry",
    ),
    (
        r"from ember\.core\.registry\.model\.model_module\.lm import LMModule.*",
        "# LMModule deprecated - use ember.api.models.Model instead",
    ),
    (
        r"from ember\.core\.registry\.model\.base\.schemas\..*",
        "# Schema imports need updating",
    ),
    # Operator imports
    (
        r"from ember\.core\.registry\.operator\.base\.operator_base import Operator",
        "from ember.operators import Operator",
    ),
    (
        r"from ember\.core\.registry\.operator\.core\.ensemble import EnsembleOperator",
        "from ember._internal.operators.ensemble import EnsembleOperator",
    ),
    (
        r"from ember\.core\.registry\.operator\.core\.most_common import MostCommonOperator",
        "from ember._internal.operators.selectors import MostCommonOperator",
    ),
    (
        r"from ember\.core\.registry\.operator\.core\.verifier import VerifierOperator",
        "from ember._internal.operators.judges import VerifierOperator",
    ),
    (
        r"from ember\.core\.registry\.operator\.base\._module import static_field",
        "# static_field removed - use regular class attributes",
    ),
    # Specification imports
    (
        r"from ember\.core\.registry\.specification\.specification import Specification",
        "from ember._internal.operators.specification import Specification",
    ),
    # Plugin system
    (
        r"from ember\.core\.plugin_system import registered_providers",
        "# Use test_providers from conftest instead",
    ),
    (
        r"from ember\.core\.plugin_system import.*",
        "from ember._internal.plugin_system import PluginSystem",
    ),
]


def fix_imports_in_file(file_path: Path) -> bool:
    """Fix imports in a single file.

    Returns True if any changes were made.
    """
    try:
        content = file_path.read_text()
        original_content = content

        # Apply all mappings
        for old_pattern, new_import in IMPORT_MAPPINGS:
            content = re.sub(old_pattern, new_import, content)

        # Fix class inheritance for test providers
        content = re.sub(
            r"class (\w+Provider)\(BaseProviderModel\):",
            r"class \1(BaseProvider):",
            content,
        )

        # Fix method names in providers
        content = re.sub(
            r"def forward\(self, request: ChatRequest\) -> ChatResponse:",
            r"def complete(self, prompt: str, model: str, **kwargs) -> ChatResponse:",
            content,
        )

        # Fix method implementations
        content = re.sub(r"request\.prompt", r"prompt", content)

        # Add _get_api_key_from_env method where needed
        if "class.*Provider.*BaseProvider" in content and "_get_api_key_from_env" not in content:
            # Find class definitions and add the method
            lines = content.split("\n")
            new_lines = []
            for i, line in enumerate(lines):
                new_lines.append(line)
                if re.match(r"class \w+Provider\(BaseProvider\):", line):
                    # Add the method after the docstring
                    j = i + 1
                    while j < len(lines) and (
                        lines[j].strip().startswith('"""') or lines[j].strip() == ""
                    ):
                        j += 1
                    if j < len(lines):
                        indent = "    "
                        new_lines.extend(
                            [
                                "",
                                f"{indent}def _get_api_key_from_env(self) -> Optional[str]:",
                                f'{indent}    """Test provider doesn\'t need API key."""',
                                f'{indent}    return "test-key"',
                                "",
                            ]
                        )
            content = "\n".join(new_lines)

        # Write back if changed
        if content != original_content:
            file_path.write_text(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main entry point."""
    test_dir = Path(__file__).parent

    # Find all Python test files
    test_files = []
    for pattern in ["test_*.py", "*_test.py"]:
        test_files.extend(test_dir.rglob(pattern))

    # Also check conftest files
    test_files.extend(test_dir.rglob("conftest.py"))

    print(f"Found {len(test_files)} test files to check")

    fixed_count = 0
    for test_file in sorted(test_files):
        if fix_imports_in_file(test_file):
            print(f"Fixed imports in: {test_file.relative_to(test_dir)}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
