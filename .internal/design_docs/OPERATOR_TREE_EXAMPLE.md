# Operator Tree Transformation Example

## The Architecture

The new operator system is built on top of EmberModule (similar to JAX Equinox), which provides:

1. **Immutability** - Operators are frozen dataclasses
2. **Tree Transformability** - Can be mapped, flattened, transformed
3. **Static/Dynamic Separation** - Some fields participate in transformations, others don't

## Example: How It Works

```python
from ember.core.operators.base_v2 import op, ModuleOperator
from ember.core.module_v2 import tree_map, tree_flatten, static_field
from dataclasses import dataclass

# Simple function operator
@op
def scale(x: float) -> float:
    return x * 2.0

# Stateful operator with EmberModule
@dataclass
class ScalingOperator(ModuleOperator):
    scale_factor: float  # Dynamic field - participates in transformations
    name: str = static_field(default="scaler")  # Static - preserved
    
    def forward(self, x: float) -> float:
        return x * self.scale_factor

# Create instance
scaler = ScalingOperator(scale_factor=2.0, name="my_scaler")

# Tree operations work!
doubled_scaler = tree_map(lambda x: x * 2, scaler)
# Now doubled_scaler.scale_factor == 4.0
# But doubled_scaler.name == "my_scaler" (unchanged)

# Flatten for JAX-style transformations
leaves, treedef = tree_flatten(scaler)
# leaves = [2.0]  # Only dynamic fields
# treedef contains reconstruction info + static fields
```

## Why This Matters

1. **JAX Integration** - Operators can be vmapped, pmapped, jitted
2. **Immutable Updates** - `scaler.replace(scale_factor=3.0)` creates new instance
3. **Composition** - Tree operations work recursively on nested operators
4. **Performance** - Static fields don't participate in expensive transformations

## The Connection to Equinox

Just like Equinox modules:
- Our operators ARE modules (not just use them)
- They're immutable dataclasses  
- They support tree transformations
- They separate static (non-trainable) from dynamic (trainable) data

The key insight from the original `_module.py` was that operators need to be first-class participants in the transformation system, not just wrappers around it.