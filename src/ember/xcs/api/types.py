import dataclasses
"""
Type definitions for the XCS API.

This module provides type definitions for the XCS API, ensuring type safety and
proper interface contracts. These types are used throughout the XCS system to
maintain consistency and clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable)

from ember.xcs.graph import Graph
from ember.xcs.tracer.xcs_tracing import TraceRecord

# Type variables for generic operators
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

# Precise option value types for configuration systems
# These define the actual value types that flow through execution contexts
# and configuration objects throughout the XCS system
OptionValue = Union[
    # Primitive types
    str,
    int,
    float,
    bool,
    None,
    # Container types (recursive definition, using Any for circularity)
    Dict[str, Any],
    List[Any],
    tuple,
    # Function references
    Callable[..., Any]]

# Strong typing for execution context dictionaries
ContextDict = Dict[str, OptionValue]


@dataclass
class XCSExecutionOptions:
    """Configuration options for XCS execution."""

    max_workers: int = 10
    """Maximum number of concurrent workers for parallel execution."""

    timeout: Optional[float] = None
    """Optional timeout in seconds for execution."""

    cache_results: bool = True
    """Whether to cache execution results."""

    debug_mode: bool = False
    """Whether to enable debug mode with additional logging."""


@dataclass
class ExecutionResult:
    """Result of executing a graph or operation."""

    outputs: Dict[str, Any]
    """The outputs from the execution."""

    execution_time: float
    """Time taken for execution in seconds."""

    node_stats: Optional[Dict[str, Dict[str, Any]]] = None
    """Optional statistics about individual node execution."""


@dataclass
class JITOptions:
    """Configuration options for JIT compilation."""

    sample_input: Optional[Dict[str, Any]] = None
    """Sample input for eager compilation."""

    force_trace: bool = False
    """Whether to force tracing on every execution."""

    recursive: bool = True
    """Whether to trace recursively."""

    cache_key_fn: Optional[Callable[[Dict[str, Any]], str]] = None
    """Optional function to create cache keys from inputs."""


@dataclass
class TransformOptions:
    """Configuration options for transforms like vmap and pmap."""

    in_axes: Optional[Union[int, Dict[str, int]]] = 0
    """Input axes for vectorization/parallelization."""

    out_axes: Optional[int] = 0
    """Output axes for vectorization/parallelization."""

    devices: Optional[List[Any]] = None
    """Devices for parallelization."""


@runtime_checkable
class GraphBuilder(Protocol):
    """Protocol for graph builders."""

    def build_graph(self, records: List[TraceRecord]) -> Graph:
        """
        Build a graph from trace records.

        Args:
            records: List of trace records

        Returns:
            An XCS graph for execution
        """
        ...


__all__ = [
    # Type variables
    "T",
    "U",
    "V",
    # Option and context types
    "OptionValue",
    "ContextDict",
    # Classes
    "XCSExecutionOptions",
    "ExecutionResult",
    "JITOptions",
    "TransformOptions",
    # Protocols
    "GraphBuilder"]


@dataclasses.dataclass
class ExecutionResult:
    """Result of executing a computation graph.

    Contains the outputs of each node in the graph as well as metrics
    about the execution.

    Attributes:
        node_outputs: Dictionary mapping node IDs to their outputs
        metrics: Execution metrics (timing, etc.)
        errors: Dictionary of errors encountered during execution
    """

    node_outputs: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    errors: Dict[str, Exception] = dataclasses.field(default_factory=dict)

    def get_result(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the result for a specific node.

        Args:
            node_id: ID of the node to retrieve results for

        Returns:
            Node's output or None if not found
        """
        return self.node_outputs.get(node_id)

    def get_error(self, node_id: str) -> Optional[Exception]:
        """Get the error for a specific node.

        Args:
            node_id: ID of the node to retrieve error for

        Returns:
            Node's error or None if no error occurred
        """
        return self.errors.get(node_id)

    def has_error(self) -> bool:
        """Check if any errors occurred during execution.

        Returns:
            True if at least one node had an error
        """
        return len(self.errors) > 0

    def is_complete(self) -> bool:
        """Check if execution completed without errors.

        Returns:
            True if execution completed successfully
        """
        return not self.has_error()

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update execution metrics.

        Args:
            metrics: New metrics to add
        """
        self.metrics.update(metrics)

