"""Debug natural API behavior."""

import sys
sys.path.insert(0, 'src')

from ember.api import xcs

# Test individual transformations
print("Testing individual transformations...\n")

# 1. Just vmap
@xcs.vmap
def compute_vmap_only(x, factor=2):
    return x * factor

result = compute_vmap_only([1, 2, 3])
print(f"vmap only: compute([1, 2, 3]) = {result}")

# 2. Just jit
@xcs.jit
def compute_jit_only(x, factor=2):
    return x * factor

result = compute_jit_only(5)
print(f"jit only: compute(5) = {result}")

# 3. Composition order 1: jit(vmap)
print("\nTesting jit(vmap)...")
vmap_first = xcs.vmap(lambda x, factor=2: x * factor)
jit_vmap = xcs.jit(vmap_first)

try:
    result = jit_vmap([1, 2, 3])
    print(f"jit(vmap): {result}")
except Exception as e:
    print(f"jit(vmap) failed: {e}")

# 4. Composition order 2: vmap(jit)
print("\nTesting vmap(jit)...")
jit_first = xcs.jit(lambda x, factor=2: x * factor)
vmap_jit = xcs.vmap(jit_first)

try:
    result = vmap_jit([1, 2, 3])
    print(f"vmap(jit): {result}")
except Exception as e:
    print(f"vmap(jit) failed: {e}")

# 5. Check what the jitted vmap function looks like
print("\nInspecting jit(vmap) function...")
print(f"Type: {type(jit_vmap)}")
print(f"Name: {jit_vmap.__name__ if hasattr(jit_vmap, '__name__') else 'N/A'}")
print(f"Is JIT compiled: {getattr(jit_vmap, '_is_jit_compiled', False)}")
print(f"Is vmapped: {getattr(jit_vmap, '_is_vmapped', False)}")

# 6. Try direct call with proper signature
print("\nTrying direct calls...")
try:
    # Try calling with inputs dict
    result = jit_vmap(inputs={'args': [[1, 2, 3]]})
    print(f"Dict call result: {result}")
except Exception as e:
    print(f"Dict call failed: {e}")