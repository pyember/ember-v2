#!/usr/bin/env python3
"""Remove unnecessary files from XCS directory following radical simplification."""

import os
import shutil
from pathlib import Path
from typing import List, Set, Dict

def find_file_usage(filename: str, search_path: str = "src/ember/xcs/") -> Set[str]:
    """Find where a file is imported or used."""
    base_name = filename.replace('.py', '')
    usage = set()
    
    # Search for imports
    import_patterns = [
        f"from .{base_name} import",
        f"from ..{base_name} import", 
        f"from ...{base_name} import",
        f"from ember.xcs.{base_name} import",
        f"import {base_name}",
        f"{base_name}.",
    ]
    
    for pattern in import_patterns:
        result = os.popen(f'grep -r "{pattern}" {search_path} --include="*.py" 2>/dev/null').read()
        if result:
            for line in result.strip().split('\n'):
                if line:
                    file_path = line.split(':')[0]
                    # Filter out self-references
                    if not file_path.endswith(filename):
                        usage.add(file_path)
    
    return usage

def categorize_files() -> Dict[str, List[str]]:
    """Categorize files for removal."""
    return {
        "old_backups": [
            "src/ember/xcs/graph/__init__old.py",
            "src/ember/xcs/jit/__init__old.py",
        ],
        "duplicate_graphs": [
            "src/ember/xcs/graph/clean_graph.py",
            "src/ember/xcs/graph/simple_graph.py", 
            "src/ember/xcs/graph/xcs_graph.py",
        ],
        "unused_jit": [
            "src/ember/xcs/jit/simple_jit.py",
            "src/ember/xcs/jit/simple_core.py",
            "src/ember/xcs/jit/simple_strategy_selector.py",
            "src/ember/xcs/jit/strategy_selector_v2.py",
        ],
        "test_in_src": [
            "src/ember/xcs/utils/test_structured_logging.py",
        ],
        "unused_utils": [
            "src/ember/xcs/utils/boundary.py",
            "src/ember/xcs/utils/tree_util.py",
            "src/ember/xcs/utils/model_conversion.py",
        ],
        "other_unused": [
            "src/ember/xcs/simple.py",
            "src/ember/xcs/ultimate.py",
            "src/ember/xcs/transforms/simple_transforms.py",
        ]
    }

def update_imports_after_removal(removed_file: str, replacement: str = None) -> None:
    """Update imports after removing a file."""
    base_name = Path(removed_file).stem
    
    # Find all files that import the removed file
    usage = find_file_usage(Path(removed_file).name, "src/")
    
    for file_path in usage:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Update imports
        if replacement:
            # Replace with new import
            content = content.replace(f"from .{base_name} import", f"from .{replacement} import")
            content = content.replace(f"from ..{base_name} import", f"from ..{replacement} import")
            content = content.replace(f"from ember.xcs.graph.{base_name} import", f"from ember.xcs.graph.{replacement} import")
        else:
            # Remove the import line
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if base_name not in line or "import" not in line:
                    new_lines.append(line)
            content = '\n'.join(new_lines)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  Updated imports in {file_path}")

def main():
    print("=== XCS Radical Simplification - Automated File Removal ===\n")
    
    categories = categorize_files()
    total_files = sum(len(files) for files in categories.values())
    
    print(f"Planning to remove {total_files} files\n")
    
    # Remove old backups - safe to remove
    print("1. Removing old backup files...")
    for file_path in categories["old_backups"]:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"  Removed {file_path}")
    
    # Remove test files in src - safe to remove
    print("\n2. Removing test files from src...")
    for file_path in categories["test_in_src"]:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"  Removed {file_path}")
    
    # Handle duplicate graph implementations
    print("\n3. Handling duplicate graph implementations...")
    for file_path in categories["duplicate_graphs"]:
        if Path(file_path).exists():
            usage = find_file_usage(Path(file_path).name)
            if usage:
                print(f"  {file_path} is used in {len(usage)} places")
                # Update imports to use graph.py instead
                update_imports_after_removal(file_path, "graph")
            os.remove(file_path)
            print(f"  Removed {file_path}")
    
    # Remove unused JIT files
    print("\n4. Removing unused JIT files...")
    for file_path in categories["unused_jit"]:
        if Path(file_path).exists():
            usage = find_file_usage(Path(file_path).name)
            if not usage:
                os.remove(file_path)
                print(f"  Removed {file_path}")
            else:
                print(f"  Keeping {file_path} - used in {len(usage)} places")
    
    # Remove unused utilities
    print("\n5. Removing unused utilities...")
    for file_path in categories["unused_utils"]:
        if Path(file_path).exists():
            usage = find_file_usage(Path(file_path).name)
            if not usage or all("test" in u for u in usage):
                os.remove(file_path)
                print(f"  Removed {file_path}")
            else:
                print(f"  Keeping {file_path} - used in {len(usage)} places")
    
    # Remove other unused files
    print("\n6. Removing other unused files...")
    for file_path in categories["other_unused"]:
        if Path(file_path).exists():
            usage = find_file_usage(Path(file_path).name)
            if not usage:
                os.remove(file_path)
                print(f"  Removed {file_path}")
            else:
                print(f"  Keeping {file_path} - used in {len(usage)} places")
    
    # Clean up empty directories
    print("\n7. Cleaning up empty directories...")
    for root, dirs, files in os.walk("src/ember/xcs", topdown=False):
        if not files and not dirs and "__pycache__" not in root:
            os.rmdir(root)
            print(f"  Removed empty directory: {root}")
    
    print("\n=== Removal complete ===")
    
    # Show remaining structure
    print("\nRemaining XCS Python files:")
    os.system("find src/ember/xcs -type f -name '*.py' | grep -v __pycache__ | wc -l")
    
    # Show what's left in each directory
    print("\nFiles per directory:")
    os.system("find src/ember/xcs -type d | grep -v __pycache__ | while read d; do echo -n \"$d: \"; find \"$d\" -maxdepth 1 -name '*.py' | wc -l; done | grep -v ': *0$'")

if __name__ == "__main__":
    main()