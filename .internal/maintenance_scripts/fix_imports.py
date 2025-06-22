#!/usr/bin/env python3
"""
Fix import issues in XCS test files.
Remove imports that don't exist in the simplified XCS API.
"""

import re
from pathlib import Path

def fix_imports_in_file(file_path: Path) -> bool:
    """Fix import issues in test files."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Remove detect_patterns and simple_vmap imports
        content = re.sub(r', detect_patterns', '', content)
        content = re.sub(r', simple_vmap as vmap', '', content)
        content = re.sub(r', vmap', '', content)
        
        # Remove any standalone detect_patterns usage (likely test code that should be removed)
        content = re.sub(r'.*detect_patterns.*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'.*vmap.*\n', '', content, flags=re.MULTILINE)
        
        # Clean up double commas
        content = re.sub(r',\s*,', ',', content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix import issues in all test files."""
    base_dir = Path(__file__).parent
    test_dirs = [
        base_dir / "tests" / "unit" / "xcs",
        base_dir / "tests" / "integration" / "xcs",
        base_dir / "tests" / "integration" / "performance"
    ]
    
    files_updated = 0
    files_processed = 0
    
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue
            
        for file_path in test_dir.rglob("*.py"):
            if file_path.name.endswith('.backup'):
                continue  # Skip backup files
            files_processed += 1
            if fix_imports_in_file(file_path):
                files_updated += 1
                print(f"Updated: {file_path.relative_to(base_dir)}")
    
    print(f"\nFixed imports in {files_updated}/{files_processed} files")

if __name__ == "__main__":
    main()