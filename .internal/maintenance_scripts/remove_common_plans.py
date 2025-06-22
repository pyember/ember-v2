#!/usr/bin/env python3
"""Remove XCSTask and XCSPlan from common/plans.py since they're no longer used."""

import os
from pathlib import Path

def update_common_init():
    """Update common/__init__.py to remove XCSTask and XCSPlan exports."""
    init_file = Path("src/ember/xcs/common/__init__.py")
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Remove imports
    content = content.replace("from ember.xcs.common.plans import ExecutionResult, XCSPlan, XCSTask", 
                            "from ember.xcs.common.plans import ExecutionResult")
    
    # Remove from __all__
    content = content.replace('"XCSPlan",\n', '')
    content = content.replace('"XCSTask",\n', '')
    
    with open(init_file, 'w') as f:
        f.write(content)
    
    print("Updated common/__init__.py")

def update_plans_file():
    """Update plans.py to remove XCSTask and XCSPlan classes."""
    plans_file = Path("src/ember/xcs/common/plans.py")
    
    with open(plans_file, 'r') as f:
        lines = f.readlines()
    
    # Find where XCSTask starts and where XCSPlan ends
    new_lines = []
    skip = False
    
    for i, line in enumerate(lines):
        if line.startswith("class XCSTask"):
            skip = True
            continue
        elif line.startswith("class XCSPlan"):
            skip = True
            continue
        elif skip and (line.strip() == "" or (not line.startswith(" ") and not line.startswith("\t"))):
            # End of class definition
            if i + 1 < len(lines) and not lines[i + 1].startswith("class XCSPlan"):
                skip = False
        
        if not skip:
            new_lines.append(line)
    
    with open(plans_file, 'w') as f:
        f.writelines(new_lines)
    
    print("Updated common/plans.py")

def remove_old_init():
    """Remove __init__old.py if it exists."""
    old_init = Path("src/ember/xcs/__init__old.py")
    
    if old_init.exists():
        os.remove(old_init)
        print("Removed __init__old.py")

def check_remaining():
    """Check what's left in common/plans.py."""
    plans_file = Path("src/ember/xcs/common/plans.py")
    
    print("\nRemaining content in common/plans.py:")
    with open(plans_file, 'r') as f:
        content = f.read()
        if len(content.strip()) < 100:
            print(content)
        else:
            print(f"File has {len(content.strip())} characters")

def main():
    print("Removing XCSTask and XCSPlan...")
    
    update_common_init()
    update_plans_file()
    remove_old_init()
    check_remaining()
    
    print("\nDone!")

if __name__ == "__main__":
    main()