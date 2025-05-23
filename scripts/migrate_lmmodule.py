#!/usr/bin/env python3
"""Semi-automated migration tool for LMModule to ModelBinding.

This script helps automate common migration patterns while allowing manual review.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import difflib
import argparse
from dataclasses import dataclass


@dataclass
class MigrationChange:
    """Represents a single change to be made."""
    line_num: int
    old_line: str
    new_line: str
    change_type: str
    confidence: float  # 0.0 to 1.0


class LMModuleMigrator(ast.NodeTransformer):
    """AST transformer to migrate LMModule usage to ModelBinding."""
    
    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.changes: List[MigrationChange] = []
        self.imports_to_add: Set[str] = set()
        self.imports_to_remove: Set[str] = set()
        
    def visit_ImportFrom(self, node):
        """Transform LMModule imports."""
        if node.module and "lm" in node.module:
            for alias in node.names:
                if alias.name in ["LMModule", "LMModuleConfig"]:
                    # Mark for removal
                    self.imports_to_remove.add(node.lineno)
                    # Add models import if not present
                    self.imports_to_add.add("from ember.api import models, ModelBinding")
        return node
    
    def visit_Call(self, node):
        """Transform LMModule instantiations."""
        if isinstance(node.func, ast.Name) and node.func.id == "LMModule":
            # Extract config parameter
            config_arg = None
            for keyword in node.keywords:
                if keyword.arg == "config":
                    config_arg = keyword.value
                    break
            
            if config_arg and isinstance(config_arg, ast.Call):
                # Extract model_id and params from LMModuleConfig
                model_id = None
                params = {}
                
                for keyword in config_arg.keywords:
                    if keyword.arg == "id":
                        if isinstance(keyword.value, ast.Constant):
                            model_id = keyword.value.value
                    else:
                        # Store other params for later
                        params[keyword.arg] = keyword
                
                if model_id:
                    # Create models.bind call
                    new_call = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='models', ctx=ast.Load()),
                            attr='bind',
                            ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=model_id)],
                        keywords=[ast.keyword(arg=k, value=v.value) for k, v in params.items()]
                    )
                    
                    # Record the change
                    old_text = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                    new_text = ast.unparse(new_call) if hasattr(ast, 'unparse') else "models.bind(...)"
                    
                    self.changes.append(MigrationChange(
                        line_num=node.lineno,
                        old_line=old_text,
                        new_line=new_text,
                        change_type="instantiation",
                        confidence=0.9
                    ))
                    
                    return new_call
        
        return self.generic_visit(node)


def migrate_file(filepath: Path, dry_run: bool = True) -> Tuple[bool, List[MigrationChange]]:
    """Migrate a single file from LMModule to ModelBinding."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
        
        # Parse AST
        tree = ast.parse(content, filename=str(filepath))
        
        # Apply transformations
        migrator = LMModuleMigrator(lines)
        new_tree = migrator.visit(tree)
        
        # Apply regex-based transformations for patterns AST might miss
        regex_changes = apply_regex_migrations(lines)
        migrator.changes.extend(regex_changes)
        
        if not dry_run and migrator.changes:
            # Apply changes
            new_content = apply_changes(content, migrator.changes, migrator.imports_to_add)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return True, migrator.changes
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False, []


def apply_regex_migrations(lines: List[str]) -> List[MigrationChange]:
    """Apply regex-based migrations for common patterns."""
    changes = []
    
    patterns = [
        # LMModule instantiation with inline config
        (
            r'LMModule\(config=LMModuleConfig\(id=["\']([\w\-\.]+)["\'](.*?)\)\)',
            r'models.bind("\1"\2)',
            "inline_instantiation"
        ),
        # Method calls that need .text
        (
            r'(\w+_module)\(prompt=([^)]+)\)',
            r'\1(\2).text',
            "method_call"
        ),
        # Simple variable assignments
        (
            r'self\.lm_module\s*=\s*LMModule\((.*?)\)',
            r'self.model = models.bind(\1)',
            "assignment"
        )
    ]
    
    for i, line in enumerate(lines):
        for pattern, replacement, change_type in patterns:
            match = re.search(pattern, line)
            if match:
                new_line = re.sub(pattern, replacement, line)
                changes.append(MigrationChange(
                    line_num=i + 1,
                    old_line=line.strip(),
                    new_line=new_line.strip(),
                    change_type=change_type,
                    confidence=0.7
                ))
    
    return changes


def apply_changes(content: str, changes: List[MigrationChange], imports_to_add: Set[str]) -> str:
    """Apply all changes to the file content."""
    lines = content.splitlines()
    
    # Add new imports after the first import block
    if imports_to_add:
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                # Find the end of import block
                j = i
                while j < len(lines) and (lines[j].startswith("import ") or 
                                         lines[j].startswith("from ") or 
                                         lines[j].strip() == ""):
                    j += 1
                # Insert new imports
                for imp in imports_to_add:
                    lines.insert(j, imp)
                break
    
    # Apply line changes (in reverse order to maintain line numbers)
    for change in sorted(changes, key=lambda x: x.line_num, reverse=True):
        if 0 <= change.line_num - 1 < len(lines):
            lines[change.line_num - 1] = lines[change.line_num - 1].replace(
                change.old_line.strip(), 
                change.new_line.strip()
            )
    
    return "\n".join(lines)


def show_diff(filepath: Path, changes: List[MigrationChange]):
    """Show a diff of proposed changes."""
    print(f"\n{'=' * 80}")
    print(f"Proposed changes for: {filepath}")
    print(f"{'=' * 80}")
    
    if not changes:
        print("No changes needed.")
        return
    
    with open(filepath, 'r') as f:
        original = f.readlines()
    
    # Create modified version
    modified = original.copy()
    for change in sorted(changes, key=lambda x: x.line_num, reverse=True):
        if 0 <= change.line_num - 1 < len(modified):
            modified[change.line_num - 1] = modified[change.line_num - 1].replace(
                change.old_line.strip(),
                change.new_line.strip()
            ) + "\n"
    
    # Show unified diff
    diff = difflib.unified_diff(
        original, 
        modified,
        fromfile=str(filepath),
        tofile=str(filepath) + " (migrated)",
        lineterm=""
    )
    
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            print(f"\033[92m{line}\033[0m")  # Green
        elif line.startswith("-") and not line.startswith("---"):
            print(f"\033[91m{line}\033[0m")  # Red
        else:
            print(line)
    
    print(f"\nTotal changes: {len(changes)}")
    for change in changes:
        print(f"  Line {change.line_num}: {change.change_type} (confidence: {change.confidence:.0%})")


def main():
    parser = argparse.ArgumentParser(description="Migrate LMModule to ModelBinding")
    parser.add_argument("files", nargs="+", help="Files to migrate")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show changes without applying them (default)")
    parser.add_argument("--apply", action="store_true",
                       help="Apply changes to files")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactively approve each file")
    
    args = parser.parse_args()
    
    if args.apply:
        args.dry_run = False
    
    total_changes = 0
    
    for file_pattern in args.files:
        # Handle wildcards
        for filepath in Path().glob(file_pattern):
            if not filepath.is_file() or not filepath.suffix == ".py":
                continue
            
            success, changes = migrate_file(filepath, dry_run=True)
            
            if success and changes:
                show_diff(filepath, changes)
                total_changes += len(changes)
                
                if args.interactive:
                    response = input(f"\nApply these changes to {filepath}? [y/N] ")
                    if response.lower() == "y":
                        migrate_file(filepath, dry_run=False)
                        print(f"✓ Migrated {filepath}")
                    else:
                        print(f"✗ Skipped {filepath}")
                elif not args.dry_run:
                    migrate_file(filepath, dry_run=False)
                    print(f"✓ Migrated {filepath}")
            elif success:
                print(f"✓ {filepath} - No changes needed")
    
    print(f"\n{'=' * 80}")
    print(f"Migration Summary: {total_changes} changes identified")
    if args.dry_run:
        print("Run with --apply to apply changes, or --interactive for file-by-file approval")


if __name__ == "__main__":
    main()