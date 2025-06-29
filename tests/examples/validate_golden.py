#!/usr/bin/env python3
"""Validate golden outputs for consistency and completeness.

This script checks that:
1. All examples have corresponding golden outputs
2. Golden outputs are valid JSON
3. Golden outputs contain required fields
4. No orphaned golden files exist
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def find_all_examples(examples_root: Path) -> Set[str]:
    """Find all Python example files."""
    examples = set()

    for py_file in examples_root.rglob("*.py"):
        # Skip __init__.py and shared utilities
        if py_file.name == "__init__.py" or "_shared" in str(py_file):
            continue

        # Get relative path from examples root
        rel_path = py_file.relative_to(examples_root)
        examples.add(str(rel_path))

    return examples


def find_all_golden_files(golden_root: Path) -> Dict[str, List[Path]]:
    """Find all golden output files grouped by example."""
    golden_files = {}

    for json_file in golden_root.rglob("*.json"):
        # Read the JSON to get the actual example path
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                example_name = data.get("example")
                mode = data.get("execution_mode", "unknown")

                if example_name:
                    if example_name not in golden_files:
                        golden_files[example_name] = []
                    golden_files[example_name].append((mode, json_file))
                else:
                    print(f"Warning: Golden file missing 'example' field: {json_file}")
        except Exception as e:
            print(f"Warning: Could not read golden file {json_file}: {e}")

    return golden_files


def validate_golden_file(file_path: Path) -> List[str]:
    """Validate a single golden file."""
    errors = []

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return errors
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
        return errors

    # Check required fields
    required_fields = ["version", "example", "execution_mode", "sections", "total_time"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate version
    if data.get("version") != "1.0":
        errors.append(f"Invalid version: {data.get('version')} (expected 1.0)")

    # Validate execution mode
    if data.get("execution_mode") not in ["simulated", "real"]:
        errors.append(f"Invalid execution_mode: {data.get('execution_mode')}")

    # Validate sections
    sections = data.get("sections", [])
    if not isinstance(sections, list):
        errors.append("Sections must be a list")
    else:
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                errors.append(f"Section {i} must be a dict")
            elif "header" not in section or "output" not in section:
                errors.append(f"Section {i} missing header or output")

    # Validate metrics
    if "metrics" in data:
        metrics = data["metrics"]
        if not isinstance(metrics, dict):
            errors.append("Metrics must be a dict")
        else:
            if "lines_of_code" in metrics and not isinstance(
                metrics["lines_of_code"], int
            ):
                errors.append("Metrics.lines_of_code must be an integer")
            if "api_calls" in metrics and not isinstance(metrics["api_calls"], int):
                errors.append("Metrics.api_calls must be an integer")

    return errors


def main():
    """Main validation function."""
    # Setup paths
    script_dir = Path(__file__).parent
    examples_root = script_dir.parent.parent / "examples"
    golden_root = script_dir / "golden_outputs"

    # Find all examples and golden files
    all_examples = find_all_examples(examples_root)
    golden_files = find_all_golden_files(golden_root)

    print(f"Found {len(all_examples)} examples")
    print(f"Found {len(golden_files)} examples with golden outputs")
    print("=" * 60)

    errors_found = False

    # Check for missing golden outputs
    print("\nChecking for missing golden outputs...")
    missing_golden = all_examples - set(golden_files.keys())
    if missing_golden:
        errors_found = True
        print(f"\n❌ {len(missing_golden)} examples missing golden outputs:")
        for example in sorted(missing_golden):
            print(f"  - {example}")
    else:
        print("✅ All examples have golden outputs")

    # Check for orphaned golden files
    print("\nChecking for orphaned golden files...")
    orphaned = set(golden_files.keys()) - all_examples
    if orphaned:
        errors_found = True
        print(f"\n❌ {len(orphaned)} orphaned golden files:")
        for example in sorted(orphaned):
            print(f"  - {example}")
    else:
        print("✅ No orphaned golden files")

    # Validate each golden file
    print("\nValidating golden file contents...")
    validation_errors = []

    for example, modes in golden_files.items():
        for mode, file_path in modes:
            errors = validate_golden_file(file_path)
            if errors:
                validation_errors.append((example, mode, errors))

    if validation_errors:
        errors_found = True
        print(f"\n❌ {len(validation_errors)} golden files have validation errors:")
        for example, mode, errors in validation_errors:
            print(f"\n  {example} ({mode} mode):")
            for error in errors:
                print(f"    - {error}")
    else:
        print("✅ All golden files are valid")

    # Summary
    print("\n" + "=" * 60)
    if errors_found:
        print("❌ Validation failed - please fix the issues above")
        return 1
    else:
        print("✅ All validations passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
