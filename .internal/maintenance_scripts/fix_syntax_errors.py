#!/usr/bin/env python3
"""Fix syntax errors from graph API migration."""

import re

def fix_syntax_errors(filepath):
    """Fix extra parentheses in graph.add calls."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix patterns like: graph.add(operator))
    #                                       ^^ extra paren
    #         )
    # Replace with: graph.add(operator)
    
    # Pattern to match graph.add(...)) followed by a line with just )
    pattern = r'(graph\.add\([^)]+\)\))\s*\n\s*\)'
    content = re.sub(pattern, r'\1', content)
    
    # Also fix single line double parens
    pattern2 = r'graph\.add\(([^)]+)\)\)'
    content = re.sub(pattern2, r'graph.add(\1)', content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

if __name__ == "__main__":
    fix_syntax_errors("src/ember/xcs/graph/graph_builder.py")