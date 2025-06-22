"""Debug deep network construction."""

import jax
import jax.numpy as jnp
from tests.unit.xcs.test_xcs_jax_grad_nested import DeepNetworkOperator

# Create network
key = jax.random.PRNGKey(42)
network = DeepNetworkOperator(
    input_dim=10,
    hidden_dims=[20, 15, 10],
    output_dim=5,
    key=key
)

print("Network layers:")
for i, layer in enumerate(network.layers):
    print(f"{i}: {type(layer).__name__}")
    if hasattr(layer, 'weight'):
        print(f"   Weight shape: {layer.weight.shape}")
    if hasattr(layer, 'scale'):
        print(f"   Scale shape: {layer.scale.shape}")

# Test forward pass
x = jnp.ones(10)
print(f"\nInput shape: {x.shape}")

try:
    for i, layer in enumerate(network.layers):
        print(f"\nLayer {i}: {type(layer).__name__}")
        print(f"  Input shape: {x.shape}")
        x = layer(x)
        print(f"  Output shape: {x.shape}")
except Exception as e:
    print(f"Error at layer {i}: {e}")