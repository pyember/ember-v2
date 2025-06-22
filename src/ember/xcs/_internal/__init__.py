"""Internal XCS implementation.

This module is private. Users should not import from here.
All public API is in ember.xcs.
"""

# Auto-register pytrees on import to fix JAX warnings
from ember.xcs._internal.pytree_registration import register_ember_pytrees
register_ember_pytrees()

# Private module - not part of public API
__all__ = []