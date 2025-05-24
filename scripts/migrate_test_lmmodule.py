#!/usr/bin/env python3
"""
Script to migrate test files from LMModule to the new models API pattern.

This script helps update test files to use the new pattern where operators
accept callable models instead of LMModule instances.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Set, Tuple


class TestMigrationVisitor(ast.NodeVisitor):
    """AST visitor to find LMModule usage patterns in tests."""
    
    def __init__(self):
        self.imports: Set[str] = set()
        self.lmmodule_usages: List[Tuple[int, str]] = []
        self.lmmoduleconfig_usages: List[Tuple[int, str]] = []
        
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track imports from ember modules."""
        if node.module and 'lm' in node.module:
            for alias in node.names:
                if alias.name in ['LMModule', 'LMModuleConfig']:
                    self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call):
        """Find LMModule and LMModuleConfig instantiations."""
        if isinstance(node.func, ast.Name):
            if node.func.id == 'LMModule':
                self.lmmodule_usages.append((node.lineno, ast.unparse(node)))
            elif node.func.id == 'LMModuleConfig':
                self.lmmoduleconfig_usages.append((node.lineno, ast.unparse(node)))
        self.generic_visit(node)


def find_test_files(root_dir: Path) -> List[Path]:
    """Find all test files that might use LMModule."""
    test_files = []
    for file_path in root_dir.rglob("test_*.py"):
        test_files.append(file_path)
    return test_files


def analyze_file(file_path: Path) -> dict:
    """Analyze a file for LMModule usage."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return {'error': str(e)}
    
    # Quick check for LMModule references
    if 'LMModule' not in content and 'lm_module' not in content:
        return {'has_lmmodule': False}
    
    try:
        tree = ast.parse(content)
        visitor = TestMigrationVisitor()
        visitor.visit(tree)
        
        # Also check for lm_module parameter usage
        lm_module_params = re.findall(r'lm_module\s*[=:]', content)
        
        return {
            'has_lmmodule': True,
            'imports': visitor.imports,
            'lmmodule_usages': visitor.lmmodule_usages,
            'lmmoduleconfig_usages': visitor.lmmoduleconfig_usages,
            'lm_module_params': len(lm_module_params),
            'content': content
        }
    except Exception as e:
        return {'error': str(e), 'has_lmmodule': True}


def generate_migration_suggestions(analysis: dict) -> List[str]:
    """Generate migration suggestions based on analysis."""
    suggestions = []
    
    if analysis.get('imports'):
        suggestions.append("Replace LMModule imports with mock model pattern")
        suggestions.append("Add MockResponse class with .text attribute")
    
    if analysis.get('lmmodule_usages'):
        suggestions.append("Replace LMModule instantiations with mock models")
        suggestions.append("Update to return response objects instead of strings")
    
    if analysis.get('lm_module_params', 0) > 0:
        suggestions.append("Change 'lm_module' parameters to 'model'")
        suggestions.append("Update method calls to use response.text")
    
    return suggestions


def main():
    """Main migration analysis function."""
    tests_dir = Path(__file__).parent.parent / "tests"
    
    print("Analyzing test files for LMModule usage...")
    print("=" * 80)
    
    files_to_migrate = []
    
    for test_file in find_test_files(tests_dir):
        analysis = analyze_file(test_file)
        
        if analysis.get('has_lmmodule'):
            files_to_migrate.append((test_file, analysis))
    
    if not files_to_migrate:
        print("No test files found using LMModule!")
        return
    
    print(f"\nFound {len(files_to_migrate)} test files using LMModule:\n")
    
    for file_path, analysis in files_to_migrate:
        relative_path = file_path.relative_to(tests_dir.parent)
        print(f"\n{relative_path}:")
        
        if 'error' in analysis:
            print(f"  ERROR: {analysis['error']}")
            continue
        
        print(f"  - LMModule imports: {len(analysis.get('imports', []))}")
        print(f"  - LMModule instantiations: {len(analysis.get('lmmodule_usages', []))}")
        print(f"  - LMModuleConfig instantiations: {len(analysis.get('lmmoduleconfig_usages', []))}")
        print(f"  - lm_module parameters: {analysis.get('lm_module_params', 0)}")
        
        suggestions = generate_migration_suggestions(analysis)
        if suggestions:
            print("  Migration suggestions:")
            for suggestion in suggestions:
                print(f"    - {suggestion}")
    
    print(f"\n\nTotal files to migrate: {len(files_to_migrate)}")
    
    # Generate migration checklist
    print("\n\nMigration Checklist:")
    print("=" * 80)
    for i, (file_path, _) in enumerate(files_to_migrate, 1):
        relative_path = file_path.relative_to(tests_dir.parent)
        print(f"{i}. [ ] {relative_path}")


if __name__ == "__main__":
    main()