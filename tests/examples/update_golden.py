#!/usr/bin/env python3
"""Script to update golden outputs for Ember examples.

This script helps maintain golden outputs by running examples and
capturing their output in a standardized format.

Usage:
    python update_golden.py                    # Update all golden outputs
    python update_golden.py --example PATH     # Update specific example
    python update_golden.py --mode real        # Update with real API calls
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from test_base import ExampleGoldenTest


class GoldenUpdater(ExampleGoldenTest):
    """Utility class for updating golden outputs."""

    def find_all_examples(self) -> List[str]:
        """Find all Python example files."""
        examples = []

        for py_file in self.examples_root.rglob("*.py"):
            # Skip __init__.py and shared utilities
            if py_file.name == "__init__.py" or "_shared" in str(py_file):
                continue

            # Get relative path from examples root
            rel_path = py_file.relative_to(self.examples_root)
            examples.append(str(rel_path))

        return sorted(examples)

    def update_example(self, example_path: str, mode: str = "simulated"):
        """Update golden output for a single example."""
        try:
            self.update_golden_from_current(example_path, mode)
        except Exception as e:
            print(f"✗ Failed to update {example_path}: {e}")
            return False
        return True

    def update_all(self, mode: str = "simulated"):
        """Update golden outputs for all examples."""
        examples = self.find_all_examples()
        total = len(examples)
        success = 0

        print(f"Found {total} examples to update")
        print("=" * 60)

        for i, example in enumerate(examples, 1):
            print(f"\n[{i}/{total}] Updating {example}...")
            if self.update_example(example, mode):
                success += 1

        print("\n" + "=" * 60)
        print(f"Updated {success}/{total} examples successfully")

        if success < total:
            return 1
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update golden outputs for Ember examples"
    )
    parser.add_argument(
        "--example",
        type=str,
        help="Update specific example (path relative to examples/)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulated", "real"],
        default="simulated",
        help="Execution mode for generating golden outputs",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available examples"
    )

    args = parser.parse_args()
    updater = GoldenUpdater()

    # Handle --list
    if args.list:
        examples = updater.find_all_examples()
        print("Available examples:")
        for example in examples:
            print(f"  {example}")
        return 0

    # Handle specific example
    if args.example:
        if updater.update_example(args.example, args.mode):
            return 0
        return 1

    # Update all examples
    return updater.update_all(args.mode)


if __name__ == "__main__":
    sys.exit(main())
