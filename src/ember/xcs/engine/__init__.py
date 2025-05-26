"""XCS engine module - now just a compatibility layer."""

# Engine functionality has been simplified and moved to other modules.
from ..graph import Graph, execute_graph
from ..execution_options import ExecutionOptions

__all__ = ["Graph", "execute_graph", "ExecutionOptions"]
