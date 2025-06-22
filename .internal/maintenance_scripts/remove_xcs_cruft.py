#!/usr/bin/env python3
"""Remove unnecessary files from XCS directory following radical simplification."""

import os
import shutil
from pathlib import Path
from typing import List, Set

def find_file_usage(filename: str) -> Set[str]:
    """Find where a file is imported or used."""
    base_name = filename.replace('.py', '')
    usage = set()
    
    # Search for imports
    import_patterns = [
        f"from .{base_name} import",
        f"from ..{base_name} import",
        f"from ember.xcs.{base_name} import",
        f"import {base_name}",
        f"{base_name}.",
    ]
    
    for pattern in import_patterns:
        result = os.popen(f'grep -r "{pattern}" src/ember/xcs/ --include="*.py" 2>/dev/null').read()
        if result:
            for line in result.strip().split('\n'):
                if line:
                    usage.add(line.split(':')[0])
    
    return usage

def remove_files_safely(files_to_remove: List[str]) -> None:
    """Remove files with safety checks."""
    for file_path in files_to_remove:
        path = Path(file_path)
        if not path.exists():
            print(f"Already removed: {file_path}")
            continue
            
        # Check usage
        usage = find_file_usage(path.name)
        # Filter out self-references
        usage = {u for u in usage if not u.endswith(path.name)}
        
        if usage:
            print(f"\nWARNING: {file_path} is still used in:")
            for use in sorted(usage):
                print(f"  - {use}")
            response = input("Remove anyway? (y/N): ")
            if response.lower() != 'y':
                print(f"Skipping {file_path}")
                continue
        
        print(f"Removing {file_path}")
        os.remove(path)

def remove_empty_directories() -> None:
    """Remove empty directories after file removal."""
    for root, dirs, files in os.walk("src/ember/xcs", topdown=False):
        if not files and not dirs:
            print(f"Removing empty directory: {root}")
            os.rmdir(root)

def main():
    print("=== XCS Radical Simplification - File Removal ===\n")
    
    # Files to remove based on analysis
    files_to_remove = [
        # Old/backup files
        "src/ember/xcs/graph/__init__old.py",
        "src/ember/xcs/jit/__init__old.py",
        
        # Duplicate graph implementations
        "src/ember/xcs/graph/clean_graph.py",
        "src/ember/xcs/graph/simple_graph.py",
        "src/ember/xcs/graph/xcs_graph.py",
        
        # Unused JIT files
        "src/ember/xcs/jit/simple_jit.py",
        "src/ember/xcs/jit/simple_core.py",
        "src/ember/xcs/jit/simple_strategy_selector.py",
        "src/ember/xcs/jit/strategy_selector_v2.py",
        
        # Test files in src
        "src/ember/xcs/utils/test_structured_logging.py",
        
        # Unused utilities
        "src/ember/xcs/utils/boundary.py",
        "src/ember/xcs/utils/tree_util.py",
        "src/ember/xcs/utils/model_conversion.py",
        
        # Other unused files
        "src/ember/xcs/simple.py",
        "src/ember/xcs/ultimate.py",
        "src/ember/xcs/transforms/simple_transforms.py",
    ]
    
    print(f"Planning to remove {len(files_to_remove)} files\n")
    
    # Remove files
    remove_files_safely(files_to_remove)
    
    # Clean up empty directories
    print("\nCleaning up empty directories...")
    remove_empty_directories()
    
    print("\n=== Removal complete ===")
    
    # Show remaining structure
    print("\nRemaining XCS structure:")
    os.system("find src/ember/xcs -type f -name '*.py' | grep -v __pycache__ | sort | head -20")

if __name__ == "__main__":
    main()