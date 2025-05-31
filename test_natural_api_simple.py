"""Simple test for Natural API."""

import sys
sys.path.insert(0, 'src')

from ember.api import xcs


# Test 1: Natural JIT
print("Testing Natural JIT...")

@xcs.jit
def add(x, y):
    return x + y

result = add(2, 3)
print(f"add(2, 3) = {result}")
assert result == 5

@xcs.jit
def multiply(x, y=2):
    return x * y

result = multiply(5)
print(f"multiply(5) = {result}")
assert result == 10

result = multiply(5, 3)
print(f"multiply(5, 3) = {result}")
assert result == 15

print("✓ Natural JIT works!\n")


# Test 2: Natural VMap
print("Testing Natural VMap...")

@xcs.vmap
def square(x):
    return x * x

result = square([1, 2, 3, 4])
print(f"square([1, 2, 3, 4]) = {result}")
assert result == [1, 4, 9, 16]

@xcs.vmap
def add_vmap(x, y):
    return x + y

result = add_vmap([1, 2, 3], [4, 5, 6])
print(f"add_vmap([1, 2, 3], [4, 5, 6]) = {result}")
assert result == [5, 7, 9]

# Keyword arguments
result = add_vmap(x=[1, 2], y=[3, 4])
print(f"add_vmap(x=[1, 2], y=[3, 4]) = {result}")
assert result == [4, 6]

print("✓ Natural VMap works!\n")


# Test 3: Combined transformations
print("Testing Combined Transformations...")

@xcs.jit
@xcs.vmap
def compute(x, factor=2):
    return x * factor

result = compute([1, 2, 3])
print(f"compute([1, 2, 3]) = {result}")
assert result == [2, 4, 6]

result = compute([1, 2, 3], factor=3)
print(f"compute([1, 2, 3], factor=3) = {result}")
assert result == [3, 6, 9]

print("✓ Combined transformations work!\n")

print("🎉 All Natural API tests passed!")