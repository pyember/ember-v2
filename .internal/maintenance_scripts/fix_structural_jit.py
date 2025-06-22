#!/usr/bin/env python3
"""Fix duplicate code in structural_jit.py"""

import re
from pathlib import Path

def fix_structural_jit():
    """Fix the duplicate code in structural_jit.py."""
    file_path = Path("src/ember/xcs/tracer/structural_jit.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and remove the duplicate exception handling block
    # The pattern is that lines 583-590 are duplicates of lines 574-580
    lines = content.split('\n')
    
    # Find the duplicate section
    duplicate_start = None
    for i in range(len(lines) - 7):
        if (lines[i].strip() == "# For machinery errors, try to recover with cached result if available" and
            lines[i+1].strip().startswith("if hasattr(graph,") and
            i > 570):  # Make sure we're in the duplicate section
            duplicate_start = i
            break
    
    if duplicate_start:
        # Remove lines from duplicate_start to duplicate_start + 7
        lines = lines[:duplicate_start] + lines[duplicate_start + 8:]
        content = '\n'.join(lines)
    
    # Also fix the imports section which has a stray )
    content = re.sub(r'\n\s*\)\nfrom ember\.xcs\.graph import Graph', '\nfrom ember.xcs.graph import Graph', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed structural_jit.py")

def fix_tracer_decorator():
    """Fix syntax issues in tracer_decorator.py."""
    file_path = Path("src/ember/xcs/tracer/tracer_decorator.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the execute_graph import
    content = re.sub(
        r'# Import execution components\n\s*execute_graph\)',
        '# Import execution components\n                    from ember.xcs.graph import execute_graph',
        content
    )
    
    # Fix the scheduler parameter
    content = re.sub(
        r'scheduler=\(\)',
        'parallel=True',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed tracer_decorator.py")

def main():
    print("Fixing syntax errors...")
    fix_structural_jit()
    fix_tracer_decorator()
    print("Done!")

if __name__ == "__main__":
    main()