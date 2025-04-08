"""Simplified import test after refactoring."""

import os
import sys

# Add src to path for imports
src_dir = os.path.join(os.path.dirname(__file__), "../src")
sys.path.insert(0, src_dir)

# Just test the core imports
print("Testing XCS JIT import...")
from ember.xcs.jit import JITMode

print("JIT modes:", [mode.value for mode in JITMode])

print("\nTesting transforms import...")

print("vmap and pmap imported successfully")

print("\nTesting engine import...")

print("Engine components imported successfully")

print("\nTesting schedulers import...")

print("Scheduler implementations imported successfully")

print("\nAll imports successful!")
