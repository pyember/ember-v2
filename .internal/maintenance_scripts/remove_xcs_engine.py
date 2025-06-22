#!/usr/bin/env python3
"""Remove xcs_engine.py and update all references to use Graph directly.

This script:
1. Updates execute_graph imports to use Graph.run() directly
2. Removes xcs_engine.py and related test files
3. Updates the XCS API to work with Graph directly
"""

import os
import re
from pathlib import Path
from typing import Set, Tuple

def find_files_to_update() -> Set[Path]:
    """Find all Python files that might need updating."""
    files = set()
    
    # Find files with execute_graph imports
    for root, _, filenames in os.walk("src"):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = Path(root) / filename
                with open(filepath, 'r') as f:
                    content = f.read()
                    if 'execute_graph' in content or 'xcs_engine' in content:
                        files.add(filepath)
    
    # Find test files
    for root, _, filenames in os.walk("tests"):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = Path(root) / filename
                with open(filepath, 'r') as f:
                    content = f.read()
                    if 'execute_graph' in content or 'xcs_engine' in content:
                        files.add(filepath)
    
    return files

def update_file(filepath: Path) -> bool:
    """Update a single file to remove xcs_engine dependencies."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Update imports of execute_graph
    content = re.sub(
        r'from ember\.xcs\.engine import execute_graph',
        'from ember.xcs.graph import Graph',
        content
    )
    
    content = re.sub(
        r'from ember\.xcs\.engine\.xcs_engine import execute_graph',
        'from ember.xcs.graph import Graph',
        content
    )
    
    content = re.sub(
        r'from \.\.\.engine import execute_graph',
        'from ...graph import Graph',
        content
    )
    
    content = re.sub(
        r'from \.\.engine import execute_graph',
        'from ..graph import Graph',
        content
    )
    
    content = re.sub(
        r'from \.engine import execute_graph',
        'from .graph import Graph',
        content
    )
    
    # Update execute_graph calls to graph.run()
    # Handle cases where graph is already a variable
    content = re.sub(
        r'execute_graph\((\w+),\s*([^)]+)\)',
        r'\1.run(\2)',
        content
    )
    
    # Remove other xcs_engine imports
    content = re.sub(
        r'from ember\.xcs\.engine\.xcs_engine import .*\n',
        '',
        content
    )
    
    content = re.sub(
        r'from \.\.\.engine\.xcs_engine import .*\n',
        '',
        content
    )
    
    # Remove ExecutionMetrics imports (we'll move this to graph.py if needed)
    content = re.sub(
        r'from ember\.xcs\.engine import ExecutionMetrics\n',
        '',
        content
    )
    
    # Update __init__.py exports
    if filepath.name == '__init__.py':
        # Remove execute_graph from exports
        content = re.sub(r'"execute_graph",?\s*\n?', '', content)
        content = re.sub(r'execute_graph,?\s*\n?', '', content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    
    return False

def update_xcs_api() -> None:
    """Update the XCS API to work directly with Graph."""
    api_file = Path("src/ember/xcs/api/core.py")
    
    with open(api_file, 'r') as f:
        content = f.read()
    
    # Update to use Graph directly
    content = re.sub(
        r'from \.\.engine import execute_graph',
        'from ..graph import Graph',
        content
    )
    
    # Update execute method to use graph.run()
    content = re.sub(
        r'return execute_graph\(graph, inputs\)',
        'return graph.run(inputs)',
        content
    )
    
    with open(api_file, 'w') as f:
        f.write(content)

def update_execution_utils() -> None:
    """Update jit/execution_utils.py to work with Graph directly."""
    utils_file = Path("src/ember/xcs/jit/execution_utils.py")
    
    with open(utils_file, 'r') as f:
        content = f.read()
    
    # Update imports
    content = re.sub(
        r'from \.\.engine import execute_graph',
        'from ..graph import Graph',
        content
    )
    
    # Update execute_compiled_graph to use graph.run()
    content = re.sub(
        r'result = execute_graph\(compiled_graph, inputs\)',
        'result = compiled_graph.run(inputs)',
        content
    )
    
    with open(utils_file, 'w') as f:
        f.write(content)

def add_execute_graph_to_graph() -> None:
    """Add a simple execute_graph function to graph.py for backward compatibility."""
    graph_file = Path("src/ember/xcs/graph/graph.py")
    
    with open(graph_file, 'r') as f:
        content = f.read()
    
    # Add execute_graph function at the end if not already present
    if 'def execute_graph' not in content:
        execute_graph_code = '''

def execute_graph(
    graph: Graph,
    inputs: Dict[str, Any],
    *,
    parallel: Union[bool, int] = True,
    timeout: Optional[float] = None) -> Dict[str, Any]:
    """Execute a computational graph.
    
    This is a compatibility wrapper for Graph.run().
    New code should use graph.run() directly.
    
    Args:
        graph: The computational graph to execute
        inputs: Input data for the graph's source nodes
        parallel: Controls parallel execution (passed to run)
        timeout: Optional timeout in seconds (currently ignored)
        
    Returns:
        Dictionary mapping node IDs to their execution results
    """
    # For now, ignore timeout - could be added to Graph.run() if needed
    if isinstance(parallel, bool):
        return graph.run(inputs) if parallel else graph.run(inputs, max_workers=1)
    else:
        return graph.run(inputs, max_workers=parallel)
'''
        content += execute_graph_code
        
        # Update imports if needed
        if 'from typing import Union' not in content:
            content = content.replace(
                'from typing import',
                'from typing import Union,'
            )
    
    with open(graph_file, 'w') as f:
        f.write(content)

def update_graph_init() -> None:
    """Update graph/__init__.py to export execute_graph."""
    init_file = Path("src/ember/xcs/graph/__init__.py")
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Add execute_graph to exports
    if 'execute_graph' not in content:
        content = content.replace(
            '"Graph",',
            '"Graph",\n    "execute_graph",'
        )
        
        content = content.replace(
            'from .graph import Graph',
            'from .graph import Graph, execute_graph'
        )
    
    with open(init_file, 'w') as f:
        f.write(content)

def update_xcs_init() -> None:
    """Update xcs/__init__.py to import execute_graph from graph."""
    init_file = Path("src/ember/xcs/__init__.py")
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Update import
    content = re.sub(
        r'from \.engine import execute_graph',
        'from .graph import execute_graph',
        content
    )
    
    with open(init_file, 'w') as f:
        f.write(content)

def remove_engine_files() -> None:
    """Remove xcs_engine.py and related files."""
    files_to_remove = [
        "src/ember/xcs/engine/xcs_engine.py",
        "tests/unit/xcs/engine/test_xcs_engine.py",
        "tests/unit/xcs/engine/test_xcs_parallel_scheduler.py", 
        "tests/unit/xcs/engine/test_xcs_noop_scheduler.py",
        "tests/integration/xcs/test_engine_integration.py",
    ]
    
    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            print(f"Removing {file_path}")
            os.remove(path)

def main():
    print("Starting xcs_engine removal...")
    
    # Step 1: Add execute_graph to graph.py for compatibility
    print("\n1. Adding execute_graph to graph.py for backward compatibility...")
    add_execute_graph_to_graph()
    update_graph_init()
    
    # Step 2: Update specific files
    print("\n2. Updating XCS API and execution utilities...")
    update_xcs_api()
    update_execution_utils()
    update_xcs_init()
    
    # Step 3: Find and update all files with execute_graph references
    print("\n3. Finding files to update...")
    files_to_update = find_files_to_update()
    
    print(f"Found {len(files_to_update)} files to check")
    
    updated_count = 0
    for filepath in sorted(files_to_update):
        if update_file(filepath):
            print(f"Updated {filepath}")
            updated_count += 1
    
    print(f"\nUpdated {updated_count} files")
    
    # Step 4: Remove xcs_engine.py and related files
    print("\n4. Removing xcs_engine.py and related test files...")
    remove_engine_files()
    
    # Step 5: Clean up engine/__init__.py
    print("\n5. Cleaning up engine/__init__.py...")
    engine_init = Path("src/ember/xcs/engine/__init__.py")
    with open(engine_init, 'w') as f:
        f.write('"""XCS engine module - simplified to just re-export Graph functionality."""\n\n')
        f.write('# The engine complexity has been removed. Use Graph directly.\n')
        f.write('from ..graph import Graph, execute_graph\n\n')
        f.write('__all__ = ["Graph", "execute_graph"]\n')
    
    print("\nDone! xcs_engine.py has been removed and all references updated.")
    print("The Graph class now handles all execution directly.")

if __name__ == "__main__":
    main()