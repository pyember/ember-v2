"""Test basic imports after refactoring."""

import inspect
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from ember.xcs.engine.unified_engine import execute_graph

# Now import the modules to verify they still work
from ember.xcs.jit import JITMode, jit
from ember.xcs.schedulers.unified_scheduler import (
    ParallelScheduler,
    SequentialScheduler,
)
from ember.xcs.transforms import vmap

# Print basic info to confirm imports work
print("JIT modes:", [mode.value for mode in JITMode])
print("JIT function signature:", inspect.signature(jit))
print("vmap function signature:", inspect.signature(vmap))
print("execute_graph function signature:", inspect.signature(execute_graph))
print("Scheduler types:", SequentialScheduler.__name__, ParallelScheduler.__name__)

print("All imports successful!")
