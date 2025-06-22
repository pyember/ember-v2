#!/usr/bin/env python3
"""Fix Graph API mismatch throughout the codebase.

Updates all uses of add_node() to use the new add() API.
"""

import re
import sys
from pathlib import Path


def fix_add_node_calls(content):
    """Convert add_node calls to add calls."""
    
    # Pattern 1: add_node(operator=..., name=...)
    # Convert to: add(operator)
    pattern1 = r'graph\.add_node\(\s*operator=([^,\)]+)(?:,\s*name=[^)]+)?\s*\)'
    content = re.sub(pattern1, r'graph.add(\1)', content)
    
    # Pattern 2: add_node with kwargs on multiple lines
    pattern2 = r'graph\.add_node\([^)]*\n[^)]*\)'
    
    def fix_multiline(match):
        text = match.group(0)
        # Extract operator value
        operator_match = re.search(r'operator=([^,\n]+)', text)
        if operator_match:
            operator = operator_match.group(1).strip()
            return f'graph.add({operator})'
        return text
    
    content = re.sub(pattern2, fix_multiline, content)
    
    return content


def main():
    # Find all Python files in xcs directory
    xcs_path = Path('src/ember/xcs')
    
    files_fixed = 0
    
    for py_file in xcs_path.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        content = py_file.read_text()
        
        if 'add_node' in content:
            print(f"Fixing {py_file}")
            new_content = fix_add_node_calls(content)
            
            if new_content != content:
                py_file.write_text(new_content)
                files_fixed += 1
    
    print(f"\nFixed {files_fixed} files")
    
    # Also need to update Graph class if it has metadata attribute
    graph_file = Path('src/ember/xcs/graph/graph.py')
    if graph_file.exists():
        content = graph_file.read_text()
        
        # Check if Graph needs metadata attribute
        if 'metadata' not in content and 'class Graph' in content:
            print("\nAdding metadata attribute to Graph class...")
            
            # Add metadata dict after __init__
            content = content.replace(
                'self._is_dag: Optional[bool] = None',
                'self._is_dag: Optional[bool] = None\n        self.metadata: Dict[str, Any] = {}'
            )
            
            # Update imports if needed
            if 'from typing import' in content and 'Any' not in content:
                content = content.replace(
                    'from typing import Union,',
                    'from typing import Union, Any,'
                )
            
            graph_file.write_text(content)
            print("Added metadata attribute to Graph")


if __name__ == '__main__':
    main()