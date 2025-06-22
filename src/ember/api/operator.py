"""Operator API (singular import path).

This module provides backward compatibility for code importing from
ember.api.operator (singular). New code should use ember.api.operators.
"""

# Re-export from the new operators module
from ember.api.operators import *
import ember.api.operators

# Maintain backward compatibility
__all__ = ember.api.operators.__all__