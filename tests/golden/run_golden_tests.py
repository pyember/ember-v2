#!/usr/bin/env python3
"""Runner script for golden tests.

This script runs all golden tests for Ember examples, including both
legacy and new structured examples.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Test files for new structure
NEW_STRUCTURE_TESTS = [
    "test_01_getting_started.py",
    "test_02_core_concepts.py",
    "test_03_operators.py",
    "test_04_compound_ai.py",
    "test_05_data_processing.py",
    "test_06_performance.py",
    "test_07_advanced_patterns.py",
    "test_08_integrations.py",
    "test_09_practical_patterns.py",
    "test_10_evaluation_suite.py",
]

# Legacy test files
LEGACY_TESTS = [
    "test_basic_examples.py",
    "test_data_examples.py",
    "test_models_examples.py",
    "test_operators_examples.py",
    "test_xcs_examples.py",
]


def run_test_file(test_file: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Run a single test file and return success status and output."""
    cmd = [sys.executable, "-m", "pytest", str(test_file)]
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"Test file {test_file.name} timed out"
    except Exception as e:
        return False, f"Error running {test_file.name}: {e}"


def check_example_updates(legacy_only: bool = False):
    """Check which examples might need updates."""
    print("\n" + "=" * 60)
    print("Checking Examples for Needed Updates")
    print("=" * 60)
    
    examples_dir = Path(__file__).parent.parent.parent / "src" / "ember" / "examples"
    
    # Patterns to check
    outdated_patterns = {
        "initialize_registry": "Uses old registry initialization",
        "ModelService(": "Creates ModelService directly",
        "from ember.core.registry.model": "Uses deep model imports",
        "poetry run": "Uses poetry instead of uv",
    }
    
    issues_found = []
    
    # Check legacy examples
    if legacy_only or (examples_dir / "legacy").exists():
        search_dirs = ["legacy/basic", "legacy/models", "legacy/operators", 
                      "legacy/data", "legacy/xcs", "legacy/advanced", "legacy/integration"]
    else:
        search_dirs = ["basic", "models", "operators", "data", "xcs", "advanced", "integration"]
    
    for category in search_dirs:
        category_dir = examples_dir / category
        if not category_dir.exists():
            continue
            
        for example_file in category_dir.glob("*.py"):
            if example_file.name == "__init__.py":
                continue
                
            with open(example_file, "r") as f:
                content = f.read()
            
            file_issues = []
            for pattern, description in outdated_patterns.items():
                if pattern in content:
                    file_issues.append(description)
            
            if file_issues:
                issues_found.append((category, example_file.name, file_issues))
    
    if issues_found:
        print("\nExamples that may need updates:")
        for category, filename, issues in issues_found:
            print(f"\n  {category}/{filename}:")
            for issue in issues:
                print(f"    - {issue}")
    else:
        print("\n✅ No obvious outdated patterns found in examples.")
    
    return len(issues_found)


def main():
    parser = argparse.ArgumentParser(description="Run Ember golden tests")
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Run only legacy tests"
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Run only new structure tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--specific",
        help="Run only a specific test file"
    )
    parser.add_argument(
        "--check-updates",
        action="store_true",
        help="Check for outdated patterns in examples"
    )
    
    args = parser.parse_args()
    
    # If only checking updates
    if args.check_updates:
        update_count = check_example_updates(args.legacy_only)
        return 0 if update_count == 0 else 1
    
    # Determine which tests to run
    tests_to_run: List[str] = []
    
    if args.specific:
        tests_to_run = [args.specific]
    elif args.legacy_only:
        tests_to_run = LEGACY_TESTS
    elif args.new_only:
        tests_to_run = NEW_STRUCTURE_TESTS
    else:
        # Run all tests
        tests_to_run = NEW_STRUCTURE_TESTS + LEGACY_TESTS
    
    # Run tests
    golden_dir = Path(__file__).parent
    failed_tests = []
    passed_tests = []
    skipped_tests = []
    
    print("=" * 60)
    print("Running Ember Golden Tests")
    print("=" * 60)
    
    for test_name in tests_to_run:
        test_path = golden_dir / test_name
        
        # Skip if test doesn't exist yet
        if not test_path.exists():
            skipped_tests.append(test_name)
            if args.verbose:
                print(f"⏭️  Skipping {test_name} (not implemented yet)")
            continue
        
        print(f"\n▶️  Running {test_name}...", end="", flush=True)
        success, output = run_test_file(test_path, args.verbose)
        
        if success:
            print(" ✅ PASSED")
            passed_tests.append(test_name)
        else:
            print(" ❌ FAILED")
            failed_tests.append(test_name)
            if args.verbose:
                print(f"\nOutput:\n{output}\n")
    
    # Check for updates if not in new-only mode
    update_count = 0
    if not args.new_only:
        update_count = check_example_updates(args.legacy_only)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total tests available: {len(tests_to_run)}")
    print(f"Tests run: {len(passed_tests) + len(failed_tests)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Skipped: {len(skipped_tests)}")
    
    if update_count > 0:
        print(f"\n⚠️  Found {update_count} examples that may need updates")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n🎉 All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())