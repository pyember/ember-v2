"""Debug how static JAX arrays are handled."""

import jax
import jax.numpy as jnp
from ember.core.module import Module
from ember.core.operators import Operator


class TestStaticArrays(Operator):
    """Test operator with arrays that should be static."""
    # These JAX arrays should be static (not learnable)
    mean: jnp.ndarray
    std: jnp.ndarray
    
    # This should be dynamic (learnable)
    scale: jnp.ndarray
    
    def __init__(self):
        self.mean = jnp.zeros(3)  # Should be static
        self.std = jnp.ones(3)    # Should be static
        self.scale = jnp.array([1.0, 2.0, 3.0])  # Should be dynamic
    
    def forward(self, x):
        return ((x - self.mean) / self.std) * self.scale


# Test gradient computation
op = TestStaticArrays()

def loss(op, x):
    return jnp.sum(op(x) ** 2)

x = jnp.ones(3)
grads = jax.grad(loss)(op, x)

print("Original operator:")
print(f"  mean: {op.mean}")
print(f"  std: {op.std}")
print(f"  scale: {op.scale}")

print("\nGradients:")
print(f"  mean grad: {grads.mean}")
print(f"  std grad: {grads.std}")
print(f"  scale grad: {grads.scale}")

# Check tree structure
leaves, treedef = jax.tree_util.tree_flatten(op)
print(f"\nNumber of leaves: {len(leaves)}")
print(f"Tree structure: {treedef}")