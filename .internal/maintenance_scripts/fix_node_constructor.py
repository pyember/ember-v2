#!/usr/bin/env python3
"""
Fix Node constructor usage in XCS tests after API simplification.

The Node class is now a simple dataclass:
@dataclass
class Node:
    id: str
    func: Callable
    deps: Tuple[str, ...] = ()

This script updates test files that still use the old Node constructor.
"""

import os
import re
from pathlib import Path

def fix_node_constructor_in_file(file_path: Path) -> bool:
    """Fix Node constructor usage in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Remove any args and kwargs parameters from Node constructors
        # The new Node dataclass only accepts: id, func, deps
        
        # Pattern 1: Remove args parameter
        content = re.sub(r'(\bNode\([^)]*),\s*args\s*=\s*\[[^\]]*\]', r'\1', content)
        
        # Pattern 2: Remove kwargs parameter  
        content = re.sub(r'(\bNode\([^)]*),\s*kwargs\s*=\s*\{[^}]*\}', r'\1', content)
        
        # Pattern 3: Convert dependencies to deps and list to tuple
        content = re.sub(r'dependencies\s*=\s*\[([^\]]*)\]', r'deps=(\1)', content)
        
        # Pattern 4: Fix empty deps lists to empty tuples
        content = re.sub(r'deps\s*=\s*\[\s*\]', 'deps=()', content)
        
        # Pattern 5: Fix deps with single item to proper tuple syntax
        content = re.sub(r'deps\s*=\s*\[([^,\]]+)\]', r'deps=(\1,)', content)
        
        # Pattern 6: Remove trailing commas before closing Node parenthesis
        content = re.sub(r',(\s*\))', r'\1', content)
        
        # Pattern 7: Remove empty deps parameter (use default)
        content = re.sub(r',\s*deps\s*=\s*\(\s*\)', '', content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix Node constructor usage in all XCS test files."""
    base_dir = Path(__file__).parent
    test_dirs = [
        base_dir / "tests" / "unit" / "xcs",
        base_dir / "tests" / "integration" / "xcs",
        base_dir / "tests" / "helpers"
    ]
    
    files_updated = 0
    files_processed = 0
    
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue
            
        for file_path in test_dir.rglob("*.py"):
            files_processed += 1
            if fix_node_constructor_in_file(file_path):
                files_updated += 1
                print(f"Updated: {file_path.relative_to(base_dir)}")
    
    print(f"\nFixed Node constructor usage in {files_updated}/{files_processed} files")

if __name__ == "__main__":
    main()