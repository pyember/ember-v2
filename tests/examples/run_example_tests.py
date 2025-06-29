#!/usr/bin/env python3
"""Quick script to test the example testing infrastructure.

This script runs a subset of example tests to verify the infrastructure works.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run a subset of example tests."""
    print("Testing Example Infrastructure")
    print("=" * 60)

    # Change to project root
    project_root = Path(__file__).parent.parent.parent

    # Test commands
    commands = [
        # List available tests
        ["python3", "-m", "pytest", "tests/examples/", "--collect-only", "-q"],
        # Run specific test in simulated mode
        [
            "python3",
            "-m",
            "pytest",
            "tests/examples/test_01_getting_started.py::TestGettingStartedExamples::test_hello_world",
            "-v",
            "--no-api-keys",
        ],
        # Validate golden outputs exist
        ["python3", "tests/examples/validate_golden.py"],
        # Show how to update golden outputs
        ["python3", "tests/examples/update_golden.py", "--list"],
    ]

    for cmd in commands:
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"\n❌ Command failed with return code {result.returncode}")
            # Don't fail on golden validation errors initially
            if "validate_golden" not in " ".join(cmd):
                return 1

    print("\n" + "=" * 60)
    print("✅ Example infrastructure is working!")
    print("\nNext steps:")
    print("1. Run 'python tests/examples/update_golden.py' to generate golden outputs")
    print("2. Run 'pytest tests/examples/ --no-api-keys' to test all examples")
    print("3. Add API keys to test real execution modes")

    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
