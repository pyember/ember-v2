"""Debug why Operator fields need explicit initialization."""

from ember.core.operators import Operator
import jax.numpy as jnp

class TestOp(Operator):
    """Simple test operator."""
    value: float
    
    def __init__(self, value: float):
        self.value = value
        # Do we need to set input_spec/output_spec?
    
    def forward(self, x):
        return x * self.value


# Try creating without setting specs
try:
    op = TestOp(2.0)
    print(f"Success! Created operator: {op}")
    print(f"input_spec: {op.input_spec}")
    print(f"output_spec: {op.output_spec}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")

# Check what fields Operator expects
import dataclasses
if hasattr(Operator, '__dataclass_fields__'):
    print(f"\nOperator dataclass fields: {list(Operator.__dataclass_fields__.keys())}")
else:
    print("\nOperator is not a dataclass")

# Check Module fields
from ember.core.module import Module
if hasattr(Module, '__dataclass_fields__'):
    print(f"Module dataclass fields: {list(Module.__dataclass_fields__.keys())}")