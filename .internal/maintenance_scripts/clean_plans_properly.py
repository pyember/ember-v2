#!/usr/bin/env python3
"""Properly clean up plans.py to remove XCSTask and XCSPlan."""

import re
from pathlib import Path

def clean_plans_file():
    """Clean up plans.py to only keep ExecutionResult."""
    plans_file = Path("src/ember/xcs/common/plans.py")
    
    with open(plans_file, 'r') as f:
        content = f.read()
    
    # Find ExecutionResult class
    execution_result_match = re.search(
        r'(@dataclasses\.dataclass\s*)?class ExecutionResult.*?(?=\n(?:class|\Z))',
        content,
        re.DOTALL
    )
    
    if execution_result_match:
        # Keep only the imports and ExecutionResult class
        new_content = '''"""Execution results for XCS.

This module contains the ExecutionResult class used to represent
the results of graph execution.
"""

import dataclasses
from typing import Any, Dict, Optional

'''
        new_content += execution_result_match.group(0).strip()
        new_content += "\n"
        
        with open(plans_file, 'w') as f:
            f.write(new_content)
        
        print("Successfully cleaned plans.py")
    else:
        # If ExecutionResult not found, create a minimal version
        new_content = '''"""Execution results for XCS.

This module contains the ExecutionResult class used to represent
the results of graph execution.
"""

import dataclasses
from typing import Any, Dict, Optional


@dataclasses.dataclass
class ExecutionResult:
    """Result of executing a computation graph.
    
    Attributes:
        outputs: Dictionary mapping node IDs to their computed values
        metadata: Optional metadata about the execution
    """
    outputs: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
'''
        
        with open(plans_file, 'w') as f:
            f.write(new_content)
        
        print("Created minimal ExecutionResult in plans.py")

def verify_cleanup():
    """Verify the cleanup was successful."""
    plans_file = Path("src/ember/xcs/common/plans.py")
    
    with open(plans_file, 'r') as f:
        content = f.read()
    
    print(f"\nFinal plans.py size: {len(content)} characters")
    
    # Check for removed classes
    if "XCSTask" in content:
        print("WARNING: XCSTask still present!")
    if "XCSPlan" in content:
        print("WARNING: XCSPlan still present!")
    
    # Show the content if it's reasonable size
    if len(content) < 1000:
        print("\nFinal content:")
        print(content)

def main():
    print("Cleaning up plans.py...")
    clean_plans_file()
    verify_cleanup()
    print("\nDone!")

if __name__ == "__main__":
    main()