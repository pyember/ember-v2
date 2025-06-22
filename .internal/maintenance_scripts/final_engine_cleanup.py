#!/usr/bin/env python3
"""Final cleanup of the engine directory."""

import os
from pathlib import Path

def remove_test_file():
    """Remove the test file that shouldn't be there."""
    test_file = Path("src/ember/xcs/engine/test_xcs_parallel_scheduler.py")
    if test_file.exists():
        os.remove(test_file)
        print(f"Removed {test_file}")

def check_execution_options_usage():
    """Check if ExecutionOptions is still being used."""
    print("\nChecking ExecutionOptions usage...")
    result = os.popen('grep -r "ExecutionOptions" src/ember/xcs/ --include="*.py" | grep -v "test_" | grep -v "execution_options.py"').read()
    
    if result:
        print("ExecutionOptions is still used in:")
        print(result)
    else:
        print("ExecutionOptions is not used anywhere else")

def check_execution_context_usage():
    """Check if execution_context.py is still being used."""
    print("\nChecking execution_context.py usage...")
    result = os.popen('grep -r "execution_context" src/ember/xcs/ --include="*.py" | grep -v "test_"').read()
    
    if result:
        print("execution_context is still used in:")
        print(result)
    else:
        print("execution_context is not used anywhere")

def remove_unused_files():
    """Remove files that are no longer used."""
    files_to_check = [
        "src/ember/xcs/engine/execution_context.py",
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            # Check if it's used
            filename = path.name
            usage = os.popen(f'grep -r "{filename[:-3]}" src/ember/xcs/ --include="*.py" | grep -v "{filename}"').read()
            
            if not usage:
                os.remove(path)
                print(f"Removed unused file: {file_path}")
            else:
                print(f"Keeping {file_path} - still in use")

def main():
    print("Final engine directory cleanup...")
    
    remove_test_file()
    check_execution_options_usage()
    check_execution_context_usage()
    remove_unused_files()
    
    # List final contents
    print("\nFinal engine directory contents:")
    os.system("ls -la src/ember/xcs/engine/")

if __name__ == "__main__":
    main()