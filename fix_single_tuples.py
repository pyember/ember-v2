#!/usr/bin/env python3
"""
Fix single-item tuple syntax in XCS tests.
Python requires (item,) for single tuples, not (item).
"""

import re
from pathlib import Path

def fix_single_tuple_syntax(file_path: Path) -> bool:
    """Fix single-item tuple syntax in test files."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern: deps=(variable) -> deps=(variable,)
        # But only when variable is not already a tuple or list
        content = re.sub(r'deps\s*=\s*\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'deps=(\1,)', content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix single tuple syntax in all test files."""
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
            if fix_single_tuple_syntax(file_path):
                files_updated += 1
                print(f"Updated: {file_path.relative_to(base_dir)}")
    
    print(f"\nFixed single tuple syntax in {files_updated}/{files_processed} files")

if __name__ == "__main__":
    main()