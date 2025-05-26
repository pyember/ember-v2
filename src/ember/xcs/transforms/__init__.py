"""Transformation system for XCS operators and functions.

Provides vectorization, parallelization, and other transformations for
enhancing the performance and capabilities of operator functions.
"""

from ember.xcs.transforms.mesh import DeviceMesh, PartitionSpec, mesh_sharded
from ember.xcs.transforms.pmap import pjit, pmap
from ember.xcs.transforms.transform_base import (
    BaseTransformation,
    BatchingOptions,
    ParallelOptions,
    TransformError,
    TransformProtocol,
    compose)
from ember.xcs.transforms.vmap import vmap

__all__ = [
    # Base transformation system
    "BaseTransformation",
    "TransformProtocol",
    "TransformError",
    "BatchingOptions",
    "ParallelOptions",
    "compose",
    # Core transformations
    "vmap",
    "pmap",
    "pjit",
    # Device mesh support
    "DeviceMesh",
    "PartitionSpec",
    "mesh_sharded"]
