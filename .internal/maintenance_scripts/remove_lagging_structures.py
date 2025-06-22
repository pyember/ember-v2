#!/usr/bin/env python3
"""Remove remaining lagging structures from XCS."""

import os
import shutil
from pathlib import Path

def check_file_usage(file_path: str, search_dirs: list) -> list:
    """Check where a file is used."""
    filename = Path(file_path).name
    basename = Path(file_path).stem
    usage = []
    
    for search_dir in search_dirs:
        # Search for imports
        result = os.popen(f'grep -r "{basename}" {search_dir} --include="*.py" 2>/dev/null | grep -v "{filename}:"').read()
        if result:
            for line in result.strip().split('\n'):
                if line and 'import' in line:
                    usage.append(line.split(':')[0])
    
    return usage

def remove_engine_directory():
    """Remove the engine directory - it's just a compatibility shim."""
    print("\n1. Removing engine directory...")
    
    engine_dir = Path("src/ember/xcs/engine")
    
    # First, update any imports that use engine
    print("   Updating imports from engine...")
    
    # Update XCS __init__.py
    xcs_init = Path("src/ember/xcs/__init__.py")
    with open(xcs_init, 'r') as f:
        content = f.read()
    
    # Remove duplicate Graph import
    lines = content.split('\n')
    new_lines = []
    seen_graph_import = False
    
    for line in lines:
        if "from ember.xcs.graph import Graph" in line or "from .graph import Graph" in line:
            if not seen_graph_import:
                new_lines.append(line)
                seen_graph_import = True
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    with open(xcs_init, 'w') as f:
        f.write(content)
    
    # Remove engine directory
    if engine_dir.exists():
        shutil.rmtree(engine_dir)
        print("   Removed engine/ directory")

def consolidate_utils():
    """Move utils files to more appropriate locations."""
    print("\n2. Consolidating utils directory...")
    
    # Move structured_logging.py to root xcs directory
    structured_logging_src = Path("src/ember/xcs/utils/structured_logging.py")
    structured_logging_dst = Path("src/ember/xcs/structured_logging.py")
    
    if structured_logging_src.exists():
        print("   Moving structured_logging.py to xcs root...")
        shutil.move(str(structured_logging_src), str(structured_logging_dst))
        
        # Update imports
        os.system('find src -name "*.py" -type f -exec sed -i "" "s/from ember.xcs.utils.structured_logging/from ember.xcs.structured_logging/g" {} + 2>/dev/null')
        os.system('find src -name "*.py" -type f -exec sed -i "" "s/from .utils.structured_logging/from .structured_logging/g" {} + 2>/dev/null')
    
    # Check boundary.py usage
    boundary_usage = check_file_usage("src/ember/xcs/utils/boundary.py", ["src/ember/xcs"])
    if not boundary_usage:
        print("   boundary.py is not imported anywhere - removing...")
        boundary_file = Path("src/ember/xcs/utils/boundary.py")
        if boundary_file.exists():
            os.remove(boundary_file)
    else:
        print(f"   boundary.py is used in {len(boundary_usage)} places - keeping for now")
    
    # Remove utils directory if empty
    utils_dir = Path("src/ember/xcs/utils")
    remaining_files = list(utils_dir.glob("*.py"))
    remaining_files = [f for f in remaining_files if f.name != "__init__.py"]
    
    if not remaining_files:
        if utils_dir.exists():
            shutil.rmtree(utils_dir)
            print("   Removed empty utils/ directory")

def clean_jit_directory():
    """Clean up JIT directory - remove modes.py if not needed."""
    print("\n3. Cleaning JIT directory...")
    
    # Check if modes.py is really needed
    modes_usage = os.popen('grep -r "from.*modes import" src/ember/xcs/ --include="*.py" | grep -v "modes.py:" | wc -l').read().strip()
    
    if modes_usage == "0":
        print("   modes.py is not imported directly - checking inline usage...")
        inline_usage = os.popen('grep -r "JITMode\\." src/ember/xcs/ --include="*.py" | grep -v "modes.py:" | wc -l').read().strip()
        
        if inline_usage == "0":
            modes_file = Path("src/ember/xcs/jit/modes.py")
            if modes_file.exists():
                os.remove(modes_file)
                print("   Removed unused modes.py")
        else:
            print(f"   modes.py has {inline_usage} inline usages - keeping")
    else:
        print(f"   modes.py is imported in {modes_usage} places - keeping")

def flatten_strategies():
    """Consider flattening the strategies directory."""
    print("\n4. Analyzing strategies directory...")
    
    strategies_dir = Path("src/ember/xcs/jit/strategies")
    strategy_files = list(strategies_dir.glob("*.py"))
    strategy_files = [f for f in strategy_files if f.name != "__init__.py"]
    
    print(f"   Found {len(strategy_files)} strategy files")
    
    # For now, keep strategies as a subdirectory since it's a logical grouping
    print("   Keeping strategies/ subdirectory - it's a logical grouping")

def remove_empty_examples():
    """Check if examples directory should be moved or removed."""
    print("\n5. Checking examples directory...")
    
    examples_dir = Path("src/ember/xcs/examples")
    example_files = list(examples_dir.glob("*.py"))
    
    if len(example_files) == 2:  # Only 2 files
        print("   Examples directory has only 2 files - consider moving to docs/xcs/examples/")
        # For now, keep them as they might be useful references
        print("   Keeping for now as reference implementations")

def update_imports():
    """Update any remaining import issues."""
    print("\n6. Cleaning up imports...")
    
    # Remove any duplicate imports in __init__ files
    init_files = [
        "src/ember/xcs/__init__.py",
        "src/ember/xcs/api/__init__.py",
        "src/ember/xcs/graph/__init__.py",
        "src/ember/xcs/jit/__init__.py",
        "src/ember/xcs/tracer/__init__.py",
        "src/ember/xcs/transforms/__init__.py",
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        if path.exists():
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # Remove duplicate lines while preserving order
            seen = set()
            new_lines = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and line_stripped not in seen:
                    seen.add(line_stripped)
                    new_lines.append(line)
                elif not line_stripped:  # Keep blank lines
                    new_lines.append(line)
            
            with open(path, 'w') as f:
                f.writelines(new_lines)

def final_cleanup():
    """Final cleanup pass."""
    print("\n7. Final cleanup...")
    
    # Remove any __pycache__ directories
    os.system('find src/ember/xcs -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null')
    
    # Remove any empty directories
    os.system('find src/ember/xcs -type d -empty -delete 2>/dev/null')
    
    print("   Removed __pycache__ and empty directories")

def show_final_structure():
    """Show the final XCS structure."""
    print("\n=== Final XCS Structure ===")
    os.system("find src/ember/xcs -type f -name '*.py' | grep -v __pycache__ | sort")
    
    print("\n=== Directory Summary ===")
    os.system("find src/ember/xcs -type d | grep -v __pycache__ | sort")

def main():
    print("=== Removing Lagging Structures from XCS ===")
    
    remove_engine_directory()
    consolidate_utils()
    clean_jit_directory()
    flatten_strategies()
    remove_empty_examples()
    update_imports()
    final_cleanup()
    show_final_structure()
    
    print("\n=== Cleanup Complete ===")

if __name__ == "__main__":
    main()