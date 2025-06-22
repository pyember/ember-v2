"""Debug Module gradient behavior."""

import jax
import jax.numpy as jnp
from ember.core.module import Module


class TestOp(Module):
    name: str
    weight: jnp.ndarray
    
    def __init__(self):
        self.name = "test"
        self.weight = jnp.ones(3)


def test_gradients():
    op = TestOp()
    
    def loss(module, x):
        return jnp.sum(module.weight * x)
    
    x = jnp.array([1.0, 2.0, 3.0])
    grad_fn = jax.grad(loss)
    grads = grad_fn(op, x)
    
    print(f"Original op: {op}")
    print(f"Gradient type: {type(grads)}")
    print(f"Gradient: {grads}")
    print(f"Has name: {hasattr(grads, 'name')}")
    print(f"Has weight: {hasattr(grads, 'weight')}")
    
    if hasattr(grads, 'name'):
        print(f"grads.name = {grads.name}")
    if hasattr(grads, 'weight'):
        print(f"grads.weight = {grads.weight}")
    
    # Check tree structure of grads
    leaves, treedef = jax.tree_util.tree_flatten(grads)
    print(f"\nGrad leaves: {leaves}")
    print(f"Grad treedef: {treedef}")


if __name__ == "__main__":
    test_gradients()