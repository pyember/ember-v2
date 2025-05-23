"""Runner for golden tests.

This script runs all golden tests for Ember examples and generates a report.
"""

import sys
import subprocess
from pathlib import Path


def run_golden_tests():
    """Run all golden tests and report results."""
    print("=" * 60)
    print("Running Ember Examples Golden Tests")
    print("=" * 60)
    
    # Find test directory
    test_dir = Path(__file__).parent
    
    # Run pytest with coverage for golden tests
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "-k", "golden",
        "--no-header"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ All golden tests passed!")
    else:
        print("\n❌ Some golden tests failed.")
        print("Please review the output above for details.")
    
    return result.returncode


def check_example_updates():
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
    
    for category in ["basic", "models", "operators", "data", "xcs", "advanced", "integration"]:
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
    """Main entry point."""
    # Run tests
    test_result = run_golden_tests()
    
    # Check for needed updates
    update_count = check_example_updates()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if test_result == 0:
        print("✅ Golden tests: PASSED")
    else:
        print("❌ Golden tests: FAILED")
    
    if update_count > 0:
        print(f"⚠️  Found {update_count} examples that may need updates")
    else:
        print("✅ Examples appear up to date")
    
    print("=" * 60)
    
    # Exit with test result code
    sys.exit(test_result)


if __name__ == "__main__":
    main()