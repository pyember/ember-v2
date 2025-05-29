#!/usr/bin/env python3
"""
Final fixes for Node usage after API simplification.
Handles remaining edge cases and test assertion updates.
"""

import re
from pathlib import Path

def fix_remaining_node_issues(file_path: Path) -> bool:
    """Fix remaining Node-related issues in test files."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Fix 1: Convert deps parameter lists to tuples
        content = re.sub(r'deps\s*=\s*\[([^\]]*)\]', r'deps=(\1)', content)
        
        # Fix 2: Remove node.args and node.kwargs assertions
        content = re.sub(r'\s*assert\s+node\.args\s*==.*\n', '', content)
        content = re.sub(r'\s*assert\s+node\.kwargs\s*==.*\n', '', content)
        
        # Fix 3: Fix single item tuples (add comma)
        content = re.sub(r'deps\s*=\s*\(([^,)]+)\)', r'deps=(\1,)', content)
        
        # Fix 4: Fix empty tuples
        content = re.sub(r'deps\s*=\s*\(\s*\)', 'deps=()', content)
        
        # Fix 5: Fix test assertions that check for list vs tuple
        content = re.sub(r'assert\s+node\.deps\s*==\s*\[([^\]]*)\]', r'assert node.deps == (\1)', content)
        
        # Fix 6: Update any remaining ValueError to TypeError for Node validation
        content = re.sub(
            r'with pytest\.raises\(ValueError,\s*match="Node function must be callable"\)',
            'with pytest.raises(TypeError, match="Expected callable")',
            content
        )
        
        # Fix 7: Fix any remaining args usage in function calls  
        content = re.sub(r'args\s*=\s*\[[^\]]*\]', '', content)
        
        # Fix 8: Clean up extra commas
        content = re.sub(r',\s*,', ',', content)
        content = re.sub(r',(\s*\))', r'\1', content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Apply final Node fixes to all test files."""
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
            if fix_remaining_node_issues(file_path):
                files_updated += 1
                print(f"Updated: {file_path.relative_to(base_dir)}")
    
    print(f"\nApplied final Node fixes to {files_updated}/{files_processed} files")

if __name__ == "__main__":
    main()