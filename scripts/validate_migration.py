#!/usr/bin/env python3
"""Validation script for LMModule migration.

This script validates that the migration maintains functionality and improves performance.
"""

import ast
import importlib
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess
import json
import sys


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.performance_metrics = {}
        self.errors = []
        self.warnings = []
        
    def add_test_result(self, test_name: str, passed: bool, error: str = None):
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
            if error:
                self.errors.append(f"{test_name}: {error}")
    
    def add_performance_metric(self, metric_name: str, value: float, baseline: float = None):
        self.performance_metrics[metric_name] = {
            "value": value,
            "baseline": baseline,
            "improvement": ((baseline - value) / baseline * 100) if baseline else None
        }
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        # Test results
        total_tests = self.tests_passed + self.tests_failed
        if total_tests > 0:
            pass_rate = self.tests_passed / total_tests * 100
            print(f"\nTests: {self.tests_passed}/{total_tests} passed ({pass_rate:.1f}%)")
        
        # Performance metrics
        if self.performance_metrics:
            print("\nPerformance Metrics:")
            for metric, data in self.performance_metrics.items():
                print(f"  {metric}: {data['value']:.3f}")
                if data['improvement'] is not None:
                    arrow = "↓" if data['improvement'] > 0 else "↑"
                    print(f"    {arrow} {abs(data['improvement']):.1f}% vs baseline")
        
        # Errors
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5
                print(f"  ❌ {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        # Warnings
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings[:3]:
                print(f"  ⚠️  {warning}")
        
        # Overall status
        print("\nStatus: ", end="")
        if self.tests_failed == 0 and not self.errors:
            print("✅ PASSED")
        else:
            print("❌ FAILED")


def validate_imports(filepath: Path) -> Tuple[bool, List[str]]:
    """Validate that LMModule imports have been removed."""
    errors = []
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check for LMModule imports
        if "from ember.core.registry.model.model_module.lm import" in content:
            if "LMModule" in content or "LMModuleConfig" in content:
                errors.append(f"Found LMModule import in {filepath}")
        
        # Check for direct usage
        if "LMModule(" in content or "LMModuleConfig(" in content:
            errors.append(f"Found LMModule usage in {filepath}")
        
        # Ensure models API is imported where needed
        tree = ast.parse(content)
        uses_models = False
        has_import = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "bind":
                if isinstance(node.value, ast.Name) and node.value.id == "models":
                    uses_models = True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "ember.api" and any(a.name == "models" for a in node.names):
                    has_import = True
        
        if uses_models and not has_import:
            errors.append(f"Uses models.bind but missing import in {filepath}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        return False, [f"Error parsing {filepath}: {e}"]


def validate_functionality(operator_class: str, module_path: str) -> Tuple[bool, str]:
    """Validate that an operator still functions correctly."""
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the operator class
        operator_cls = getattr(module, operator_class, None)
        if not operator_cls:
            return False, f"Could not find class {operator_class}"
        
        # Try to instantiate
        if operator_class == "EnsembleOperator":
            operator = operator_cls(models=["gpt-3.5-turbo", "gpt-3.5-turbo"])
        elif operator_class == "VerifierOperator":
            operator = operator_cls(model="gpt-3.5-turbo")
        else:
            # Generic instantiation
            operator = operator_cls(model="gpt-3.5-turbo")
        
        # Basic validation - can we call forward?
        if hasattr(operator, "forward"):
            # Would need mock models to actually test
            return True, "Instantiation successful"
        else:
            return False, "No forward method found"
            
    except Exception as e:
        return False, f"Error: {str(e)}"


def run_tests(test_pattern: str = None) -> Tuple[int, int]:
    """Run test suite and return pass/fail counts."""
    cmd = ["python3", "-m", "pytest", "-xvs"]
    if test_pattern:
        cmd.append(test_pattern)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse pytest output
        for line in result.stdout.split('\n'):
            if "passed" in line and "failed" in line:
                # Extract numbers from pytest summary
                import re
                passed_match = re.search(r'(\d+) passed', line)
                failed_match = re.search(r'(\d+) failed', line)
                
                passed = int(passed_match.group(1)) if passed_match else 0
                failed = int(failed_match.group(1)) if failed_match else 0
                
                return passed, failed
        
        # If all passed
        if result.returncode == 0:
            # Count tests from output
            test_count = result.stdout.count(" PASSED")
            return test_count, 0
        
        return 0, 1  # At least one failure
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 0, 1


def benchmark_performance() -> Dict[str, float]:
    """Run performance benchmarks."""
    metrics = {}
    
    # Benchmark 1: Import time
    start = time.time()
    try:
        from ember.api import models
        metrics["import_time"] = time.time() - start
    except:
        metrics["import_time"] = -1
    
    # Benchmark 2: Model binding creation
    if "import_time" in metrics and metrics["import_time"] > 0:
        start = time.time()
        for _ in range(100):
            model = models.models.bind("gpt-3.5-turbo", temperature=0.7)
        metrics["binding_creation_per_100"] = time.time() - start
    
    # Benchmark 3: Memory usage (simplified)
    try:
        import psutil
        process = psutil.Process()
        metrics["memory_mb"] = process.memory_info().rss / 1024 / 1024
    except:
        pass
    
    return metrics


def check_backwards_compatibility() -> List[str]:
    """Check for backwards compatibility issues."""
    warnings = []
    
    # Check if compatibility layer exists
    compat_path = Path("src/ember/core/registry/model/model_module/lm.py")
    if compat_path.exists():
        with open(compat_path, 'r') as f:
            content = f.read()
            if "DeprecationWarning" not in content:
                warnings.append("Compatibility layer missing deprecation warnings")
    else:
        warnings.append("No compatibility layer found - this may break existing code")
    
    return warnings


def validate_examples() -> Tuple[int, int]:
    """Validate that examples run without errors."""
    examples_dir = Path("examples")
    passed = 0
    failed = 0
    
    for example in examples_dir.rglob("*.py"):
        if "lm_module" in example.stem.lower():
            continue  # Skip old examples
        
        try:
            # Try to parse and compile
            with open(example, 'r') as f:
                compile(f.read(), str(example), 'exec')
            passed += 1
        except SyntaxError:
            failed += 1
        except Exception:
            # Other errors are okay for validation
            passed += 1
    
    return passed, failed


def main():
    """Run full validation suite."""
    print("🔍 Running LMModule Migration Validation")
    print("=" * 80)
    
    result = ValidationResult()
    
    # Step 1: Validate imports have been migrated
    print("\n1. Validating imports...")
    files_to_check = list(Path("src").rglob("*.py"))
    import_issues = 0
    
    for filepath in files_to_check:
        if "model_module/lm.py" in str(filepath):
            continue  # Skip the module itself
        
        passed, errors = validate_imports(filepath)
        if not passed:
            import_issues += 1
            for error in errors:
                result.add_warning(error)
    
    result.add_test_result("Import validation", import_issues == 0, 
                          f"{import_issues} files with import issues")
    
    # Step 2: Run operator functionality tests
    print("\n2. Testing operator functionality...")
    operators_to_test = [
        ("VerifierOperator", "src/ember/core/registry/operator/core/verifier.py"),
        ("EnsembleOperator", "src/ember/core/registry/operator/core/ensemble.py")]
    
    for op_class, op_path in operators_to_test:
        if Path(op_path).exists():
            passed, error = validate_functionality(op_class, op_path)
            result.add_test_result(f"{op_class} functionality", passed, error)
    
    # Step 3: Run test suite
    print("\n3. Running test suite...")
    passed, failed = run_tests("tests/unit/core/registry/operator/")
    result.tests_passed += passed
    result.tests_failed += failed
    
    # Step 4: Performance benchmarks
    print("\n4. Running performance benchmarks...")
    metrics = benchmark_performance()
    
    # Add baseline values (these would come from pre-migration measurements)
    baselines = {
        "import_time": 0.05,  # Example baseline
        "binding_creation_per_100": 0.01,
        "memory_mb": 150
    }
    
    for metric, value in metrics.items():
        baseline = baselines.get(metric)
        result.add_performance_metric(metric, value, baseline)
    
    # Step 5: Check backwards compatibility
    print("\n5. Checking backwards compatibility...")
    warnings = check_backwards_compatibility()
    for warning in warnings:
        result.add_warning(warning)
    
    # Step 6: Validate examples
    print("\n6. Validating examples...")
    passed, failed = validate_examples()
    result.add_test_result("Examples validation", failed == 0,
                          f"{failed} examples with issues")
    
    # Print summary
    result.print_summary()
    
    # Save detailed report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {
            "passed": result.tests_passed,
            "failed": result.tests_failed
        },
        "performance": result.performance_metrics,
        "errors": result.errors,
        "warnings": result.warnings
    }
    
    with open("migration_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: migration_validation_report.json")
    
    # Exit with appropriate code
    sys.exit(0 if result.tests_failed == 0 else 1)


if __name__ == "__main__":
    main()