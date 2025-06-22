#!/usr/bin/env python3
"""Fix test imports after core module restructuring."""

import os
import re

# Mapping of old imports to new imports
import_mappings = {
    r'from ember\.core\.module import Module': 'from ember._internal.module import Module',
    r'from ember\.core\.operators import': 'from ember.operators import',
    r'from ember\.core\.exceptions import': 'from ember._internal.exceptions import',
    r'from ember\.core\.context\.ember_context import': 'from ember._internal.context.ember_context import',
    r'from ember\.core\.metrics\.metrics import': 'from ember._internal.metrics.metrics import',
    r'from ember\.core\.types\.ember_model import EmberModel': 'from ember._internal.types import EmberModel',
    r'from ember\._internal\.types\.ember_model import EmberModel': 'from ember._internal.types import EmberModel',
    r'from ember\.core\.utils\.': 'from ember.utils.',
    r'from ember\.core\.config\.': 'from ember._internal.config.',
    r'from ember\.core\.': 'from ember._internal.',
    r'import ember\.core\.': 'import ember._internal.',
}

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all import mappings
        for old_pattern, new_import in import_mappings.items():
            content = re.sub(old_pattern, new_import, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Fixed imports in: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix imports in all test files."""
    test_dir = "tests"
    fixed_count = 0
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_imports_in_file(filepath):
                    fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()