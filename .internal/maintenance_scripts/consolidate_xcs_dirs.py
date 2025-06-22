#!/usr/bin/env python3
"""Consolidate XCS directories to reduce complexity."""

import os
import shutil
from pathlib import Path

def move_file_and_update_imports(src: str, dst: str, old_import_path: str, new_import_path: str):
    """Move a file and update all imports."""
    print(f"Moving {src} -> {dst}")
    
    # Create destination directory if needed
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    
    # Move the file
    shutil.move(src, dst)
    
    # Update imports across the codebase
    print(f"  Updating imports: {old_import_path} -> {new_import_path}")
    
    # Find all Python files
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    content = f.read()
                
                original_content = content
                
                # Update various import patterns
                content = content.replace(f"from {old_import_path} import", f"from {new_import_path} import")
                content = content.replace(f"import {old_import_path}", f"import {new_import_path}")
                
                if content != original_content:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    print(f"    Updated {file_path}")

def consolidate_directories():
    """Main consolidation logic."""
    print("=== XCS Directory Consolidation ===\n")
    
    # 1. Move ExecutionResult from common/plans.py to api/types.py
    print("1. Consolidating ExecutionResult into api/types.py...")
    
    # Read ExecutionResult from plans.py
    plans_path = Path("src/ember/xcs/common/plans.py")
    if plans_path.exists():
        with open(plans_path, 'r') as f:
            plans_content = f.read()
        
        # Extract ExecutionResult class
        import re
        result_match = re.search(
            r'(@dataclasses\.dataclass\s*)?class ExecutionResult.*?(?=(?:\n@|\nclass|\Z))',
            plans_content,
            re.DOTALL
        )
        
        if result_match:
            execution_result = result_match.group(0)
            
            # Append to api/types.py
            types_path = Path("src/ember/xcs/api/types.py")
            with open(types_path, 'r') as f:
                types_content = f.read()
            
            # Add import if needed
            if "import dataclasses" not in types_content:
                types_content = "import dataclasses\n" + types_content
            
            # Add ExecutionResult
            types_content += "\n\n" + execution_result + "\n"
            
            with open(types_path, 'w') as f:
                f.write(types_content)
            
            print("  Added ExecutionResult to api/types.py")
            
            # Update imports
            os.system('find src -name "*.py" -type f -exec sed -i "" "s/from ember.xcs.common.plans import ExecutionResult/from ember.xcs.api.types import ExecutionResult/g" {} +')
            os.system('find src -name "*.py" -type f -exec sed -i "" "s/from ember.xcs.common import ExecutionResult/from ember.xcs.api.types import ExecutionResult/g" {} +')
            
            # Remove plans.py and common directory
            os.remove(plans_path)
            common_init = Path("src/ember/xcs/common/__init__.py")
            if common_init.exists():
                os.remove(common_init)
            
            # Remove empty directory
            common_dir = Path("src/ember/xcs/common")
            if common_dir.exists() and not list(common_dir.iterdir()):
                os.rmdir(common_dir)
            
            print("  Removed common/ directory")
    
    # 2. Move execution_options.py to root xcs directory
    print("\n2. Moving execution_options.py to xcs root...")
    
    exec_opts_src = Path("src/ember/xcs/engine/execution_options.py")
    exec_opts_dst = Path("src/ember/xcs/execution_options.py")
    
    if exec_opts_src.exists():
        move_file_and_update_imports(
            str(exec_opts_src),
            str(exec_opts_dst),
            "ember.xcs.engine.execution_options",
            "ember.xcs.execution_options"
        )
        
        # Update engine/__init__.py
        engine_init = Path("src/ember/xcs/engine/__init__.py")
        with open(engine_init, 'w') as f:
            f.write('"""XCS engine module - now just a compatibility layer."""\n\n')
            f.write('# Engine functionality has been simplified and moved to other modules.\n')
            f.write('from ..graph import Graph, execute_graph\n')
            f.write('from ..execution_options import ExecutionOptions\n\n')
            f.write('__all__ = ["Graph", "execute_graph", "ExecutionOptions"]\n')
    
    # 3. Check if we can remove utils directory
    print("\n3. Analyzing utils directory...")
    
    utils_files = list(Path("src/ember/xcs/utils").glob("*.py"))
    utils_files = [f for f in utils_files if f.name != "__init__.py"]
    
    print(f"  Found {len(utils_files)} files in utils/")
    
    for file in utils_files:
        # Check usage
        usage = os.popen(f'grep -r "{file.stem}" src/ember/xcs/ --include="*.py" | grep -v "{file.name}" | wc -l').read().strip()
        print(f"  - {file.name}: {usage} references")
    
    # 4. Clean up __init__.py files
    print("\n4. Updating __init__.py files...")
    
    # Update xcs/__init__.py
    xcs_init = Path("src/ember/xcs/__init__.py")
    with open(xcs_init, 'r') as f:
        content = f.read()
    
    # Update execution_options import
    content = content.replace(
        "from ember.xcs.engine.execution_options import ExecutionOptions",
        "from ember.xcs.execution_options import ExecutionOptions"
    )
    
    with open(xcs_init, 'w') as f:
        f.write(content)
    
    print("  Updated xcs/__init__.py")
    
    print("\n=== Consolidation complete ===")

def show_final_structure():
    """Show the final directory structure."""
    print("\nFinal XCS structure:")
    os.system("find src/ember/xcs -type d | grep -v __pycache__ | sort")
    
    print("\nFiles per directory:")
    os.system("find src/ember/xcs -type d | grep -v __pycache__ | while read d; do echo -n \"$d: \"; find \"$d\" -maxdepth 1 -name '*.py' | wc -l; done | grep -v ': *0$'")

def main():
    consolidate_directories()
    show_final_structure()

if __name__ == "__main__":
    main()