"""Test that our Module implementation satisfies all design principles.

Following CLAUDE.md principles - verify the implementation works as intended.
"""

import jax
import jax.numpy as jnp
from ember.core.module import Module


class TestOperator(Module):
    """Test operator with mixed field types."""
    # These should be automatically static
    name: str
    config: dict
    activation: str
    dropout_rate: float
    
    # These should be automatically dynamic
    weights: jnp.ndarray
    bias: jnp.ndarray
    
    def __init__(self, dim: int):
        self.name = "test"
        self.config = {"dim": dim}
        self.activation = "relu"
        self.dropout_rate = 0.1
        self.weights = jnp.ones((dim, dim))
        self.bias = jnp.zeros(dim)


def test_static_by_default():
    """Test that non-JAX fields are static by default."""
    op = TestOperator(4)
    
    # Check that Module is using our metaclass
    assert hasattr(Module, '__class__')
    assert 'EmberModuleMeta' in str(type(Module))
    
    # Tree flatten to see structure
    leaves, treedef = jax.tree_util.tree_flatten(op)
    
    print(f"Leaves: {[type(l).__name__ for l in leaves]}")
    print(f"Tree structure:\n{treedef}")
    
    # Only JAX arrays should be leaves
    assert len(leaves) == 2, f"Expected 2 leaves (weights, bias), got {len(leaves)}"
    assert all(isinstance(l, jnp.ndarray) for l in leaves)
    
    # Static fields should be in the treedef
    treedef_str = str(treedef)
    assert "test" in treedef_str  # name
    assert "relu" in treedef_str  # activation
    
    print("✓ Static by default: PASSED")


def test_dynamic_for_jax_arrays():
    """Test that JAX arrays are dynamic automatically."""
    op = TestOperator(3)
    
    # Define a simple loss
    def loss_fn(module, x):
        return jnp.sum(module.weights @ x + module.bias)
    
    # Compute gradient
    x = jnp.ones(3)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(op, x)
    
    # Should have gradients for JAX arrays
    assert hasattr(grads, 'weights')
    assert hasattr(grads, 'bias')
    assert grads.weights.shape == op.weights.shape
    assert grads.bias.shape == op.bias.shape
    
    # Static fields are preserved in structure but not differentiated
    assert hasattr(grads, 'name')
    assert grads.name == op.name  # Static value preserved
    assert hasattr(grads, 'activation')
    assert grads.activation == op.activation
    
    # Check that only JAX arrays are leaves (dynamic)
    leaves, _ = jax.tree_util.tree_flatten(grads)
    assert len(leaves) == 2  # Only weights and bias
    assert all(isinstance(l, jnp.ndarray) for l in leaves)
    
    print("✓ Dynamic for JAX arrays: PASSED")


def test_zero_configuration():
    """Test that no decorators or special syntax needed."""
    # The TestOperator class above has no decorators or field() calls
    # yet it works correctly
    
    class SimpleOp(Module):
        text: str  # Automatically static
        values: jnp.ndarray  # Automatically dynamic
        
        def __init__(self):
            self.text = "hello"
            self.values = jnp.array([1.0, 2.0, 3.0])
    
    op = SimpleOp()
    leaves, _ = jax.tree_util.tree_flatten(op)
    
    # Only values should be a leaf
    assert len(leaves) == 1
    assert jnp.array_equal(leaves[0], op.values)
    
    print("✓ Zero configuration: PASSED")


def test_full_jax_compatibility():
    """Test all JAX transformations work naturally."""
    
    class VmapOp(Module):
        scale: jnp.ndarray
        name: str
        
        def __init__(self, scale):
            self.scale = scale
            self.name = "vmap_test"
        
        def __call__(self, x):
            return x * self.scale
    
    # Test vmap
    scales = jnp.array([1.0, 2.0, 3.0])
    ops = jax.vmap(VmapOp)(scales)
    
    x = jnp.ones(4)
    results = jax.vmap(lambda op, x: op(x))(ops, jnp.stack([x, x, x]))
    assert results.shape == (3, 4)
    
    # Test jit
    @jax.jit
    def jitted_fn(op, x):
        return op(x)
    
    single_op = VmapOp(jnp.array(2.0))
    result = jitted_fn(single_op, x)
    assert jnp.allclose(result, x * 2.0)
    
    # Test grad
    def loss(op, x):
        return jnp.sum(op(x) ** 2)
    
    grad_fn = jax.grad(loss)
    grads = grad_fn(single_op, x)
    assert hasattr(grads, 'scale')
    assert hasattr(grads, 'name')  # Structure preserved
    assert grads.name == single_op.name  # Static value unchanged
    
    print("✓ Full JAX compatibility: PASSED")


if __name__ == "__main__":
    print("Testing Module implementation against design principles...\n")
    
    test_static_by_default()
    test_dynamic_for_jax_arrays()
    test_zero_configuration()
    test_full_jax_compatibility()
    
    print("\n✅ All desiderata satisfied!")