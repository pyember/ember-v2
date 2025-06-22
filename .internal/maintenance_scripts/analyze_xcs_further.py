#!/usr/bin/env python3
"""Analyze XCS for further simplification opportunities."""

import os
from pathlib import Path
from collections import defaultdict

def analyze_directory_contents():
    """Analyze what's in each XCS subdirectory."""
    dirs = defaultdict(list)
    
    for root, _, files in os.walk("src/ember/xcs"):
        if "__pycache__" in root:
            continue
        
        py_files = [f for f in files if f.endswith('.py') and f != '__init__.py']
        if py_files:
            rel_path = root.replace("src/ember/xcs/", "")
            if root == "src/ember/xcs":
                rel_path = "root"
            dirs[rel_path] = py_files
    
    print("=== XCS Directory Analysis ===\n")
    
    for dir_name, files in sorted(dirs.items()):
        print(f"{dir_name}/ ({len(files)} files):")
        for f in sorted(files):
            print(f"  - {f}")
        print()

def check_cross_directory_imports():
    """Check imports between XCS subdirectories."""
    print("=== Cross-directory imports ===\n")
    
    # Check key imports
    patterns = [
        ("engine imports", "from ember.xcs.engine import"),
        ("common imports", "from ember.xcs.common import"),
        ("utils imports", "from ember.xcs.utils import"),
        ("tracer imports", "from ember.xcs.tracer import"),
    ]
    
    for name, pattern in patterns:
        result = os.popen(f'grep -r "{pattern}" src/ember/xcs/ --include="*.py" | wc -l').read().strip()
        print(f"{name}: {result} occurrences")

def suggest_consolidation():
    """Suggest directory consolidation."""
    print("\n=== Consolidation Suggestions ===\n")
    
    # Check engine directory
    engine_files = os.popen('ls src/ember/xcs/engine/*.py | grep -v __init__').read().strip().split('\n')
    print(f"Engine directory has {len([f for f in engine_files if f])} non-init files")
    
    # Check common directory  
    common_files = os.popen('ls src/ember/xcs/common/*.py | grep -v __init__').read().strip().split('\n')
    print(f"Common directory has {len([f for f in common_files if f])} non-init files")
    
    # Check utils directory
    utils_files = os.popen('ls src/ember/xcs/utils/*.py | grep -v __init__').read().strip().split('\n')
    print(f"Utils directory has {len([f for f in utils_files if f])} non-init files")
    
    print("\nRecommendations:")
    print("1. Engine directory: Only has execution_options.py - could move to root or graph/")
    print("2. Common directory: Only has plans.py with ExecutionResult - could move to api/types.py")
    print("3. Utils directory: Has 2 files - consider if they're really needed")

def check_circular_dependencies():
    """Check for circular import patterns."""
    print("\n=== Checking for circular dependencies ===\n")
    
    # Check if any subdirectory imports from its parent
    subdirs = ["api", "common", "engine", "graph", "jit", "tracer", "transforms", "utils"]
    
    for subdir in subdirs:
        # Check if subdir imports from xcs root
        result = os.popen(f'grep -r "from ember.xcs import" src/ember/xcs/{subdir}/ --include="*.py" | grep -v "__all__" | wc -l').read().strip()
        if result != "0":
            print(f"{subdir}/ imports from xcs root: {result} times")

def main():
    analyze_directory_contents()
    check_cross_directory_imports()
    suggest_consolidation()
    check_circular_dependencies()

if __name__ == "__main__":
    main()