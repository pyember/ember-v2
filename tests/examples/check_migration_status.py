#!/usr/bin/env python3
"""Check migration status of all examples."""

import json
from pathlib import Path


def check_migration_status():
    """Check which examples are migrated, have golden outputs, and pass tests."""
    examples_dir = Path("examples")
    golden_dir = Path("tests/examples/golden_outputs")

    # Find all examples
    all_examples = set()
    for py_file in examples_dir.rglob("*.py"):
        if "__init__.py" not in str(py_file) and "_shared" not in str(py_file):
            rel_path = py_file.relative_to(examples_dir)
            all_examples.add(str(rel_path))

    # Find migrated examples
    migrated_examples = set()
    for py_file in examples_dir.rglob("*.py"):
        if "__init__.py" not in str(py_file) and "_shared" not in str(py_file):
            with open(py_file, "r") as f:
                content = f.read()
            if "from _shared.conditional_execution import conditional_llm" in content:
                rel_path = py_file.relative_to(examples_dir)
                migrated_examples.add(str(rel_path))

    # Find examples with golden outputs
    golden_examples = set()
    for json_file in golden_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        example_path = data.get("example", "")
        if example_path:
            golden_examples.add(example_path)

    # Organize by directory
    by_directory = {}
    for example in sorted(all_examples):
        dir_name = Path(example).parts[0] if "/" in example else "root"
        if dir_name not in by_directory:
            by_directory[dir_name] = []

        status = []
        if example in migrated_examples:
            status.append("✅ Migrated")
        else:
            status.append("❌ Not migrated")

        if example in golden_examples:
            status.append("📄 Has golden")
        else:
            status.append("⚠️  No golden")

        by_directory[dir_name].append((example, status))

    # Print status
    print("Ember Examples Migration Status")
    print("=" * 80)
    print(f"Total examples: {len(all_examples)}")
    print(
        f"Migrated: {len(migrated_examples)} ({len(migrated_examples)/len(all_examples)*100:.1f}%)"
    )
    print(
        f"With golden outputs: {len(golden_examples)} "
        f"({len(golden_examples)/len(all_examples)*100:.1f}%)"
    )
    print("=" * 80)

    for dir_name, examples in sorted(by_directory.items()):
        print(f"\n{dir_name}/")
        print("-" * 40)
        for example, status in examples:
            status_str = " | ".join(status)
            print(f"  {example:<50} {status_str}")

    # Summary of what needs to be done
    not_migrated = all_examples - migrated_examples
    migrated_no_golden = migrated_examples - golden_examples

    if not_migrated:
        print(f"\n\n📋 TODO: {len(not_migrated)} examples need migration:")
        for ex in sorted(not_migrated)[:10]:
            print(f"  - {ex}")
        if len(not_migrated) > 10:
            print(f"  ... and {len(not_migrated) - 10} more")

    if migrated_no_golden:
        print(f"\n\n⚠️  {len(migrated_no_golden)} migrated examples need golden outputs:")
        for ex in sorted(migrated_no_golden):
            print(f"  - {ex}")

    print("\n\n🎯 Next steps:")
    if not_migrated:
        print("1. Run: python3 migrate_examples.py")
        print("   Or manually migrate complex examples")
    if migrated_no_golden:
        print("2. Generate golden outputs: python3 tests/examples/update_golden.py")
    print("3. Run tests: pytest tests/examples/ --no-api-keys")


if __name__ == "__main__":
    check_migration_status()
