"""IR Builder - constructs computation graphs from Python functions.

This is hidden from users. They just write Python, we build the graph.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ember.xcs._internal.ir import IRNode, IRGraph
from ember.xcs._internal.tracer import PythonTracer, Operation, TracingError


@dataclass
class BuilderState:
    """Mutable state during graph construction."""
    nodes: Dict[str, IRNode] = field(default_factory=dict)
    var_counter: int = 0
    node_counter: int = 0
    var_producers: Dict[str, str] = field(default_factory=dict)  # var_name -> node_id
    
    def next_var(self) -> str:
        """Generate next variable name."""
        var_name = f"_var_{self.var_counter}"
        self.var_counter += 1
        return var_name
    
    def next_node_id(self) -> str:
        """Generate next node ID."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        return node_id


class IRBuilder:
    """Builds IR graphs from Python functions.
    
    Users never see this - it's pure implementation detail.
    """
    
    def __init__(self):
        self.state = BuilderState()
        self.tracer = PythonTracer()
    
    def trace_function(self, func: Callable, args: tuple, kwargs: dict) -> IRGraph:
        """Trace function execution to build IR graph.
        
        Uses runtime tracing to capture actual execution, not AST analysis.
        """
        try:
            # Trace the function to get operations
            operations = self.tracer.trace_function(func, args, kwargs)
            
            # Convert operations to IR graph
            return self._build_graph_from_operations(operations)
            
        except TracingError:
            # Can't trace - create single node graph
            return self._create_single_node_graph(func, args, kwargs)
    
    def _build_graph_from_operations(self, operations: List[Operation]) -> IRGraph:
        """Build IR graph from traced operations."""
        # Reset state for new graph
        self.state = BuilderState()
        
        # Map operation IDs to node IDs
        op_to_node: Dict[int, str] = {}
        
        # Track the main function operation
        main_func_op = None
        
        # Process each operation
        for op in operations:
            # Save the main function operation for later
            if op.operation_id == len(operations) - 1:
                main_func_op = op
                continue
                
            node_id = self.state.next_node_id()
            op_to_node[op.operation_id] = node_id
            
            # Create input variables based on dependencies
            inputs = []
            for dep_id in sorted(op.dependencies):
                if dep_id in op_to_node:
                    # Reference output of dependency
                    inputs.append(f"{op_to_node[dep_id]}_out")
            
            # If no dependencies, create argument variables
            if not inputs:
                for i, arg in enumerate(op.args):
                    inputs.append(f"_arg_{i}")
            
            # Create output variable
            output_var = f"{node_id}_out"
            
            # Create IR node
            node = IRNode(
                id=node_id,
                operator=op.func,
                inputs=tuple(inputs),
                outputs=(output_var,),
                metadata={
                    'args': op.args,
                    'kwargs': op.kwargs,
                    'result': op.result
                }
            )
            
            self.state.nodes[node_id] = node
            self.state.var_producers[output_var] = node_id
        
        # Add a return node to capture the actual return value
        if main_func_op:
            return_node_id = "return_node"
            return_output_var = "_return_value"
            
            # Create a simple identity operator that returns the result
            def return_op(x=None):
                return main_func_op.result
            
            return_node = IRNode(
                id=return_node_id,
                operator=return_op,
                inputs=(),  # No inputs needed
                outputs=(return_output_var,),
                metadata={
                    'is_return': True,
                    'result': main_func_op.result
                }
            )
            
            self.state.nodes[return_node_id] = return_node
        
        return self.build()
    
    def _create_single_node_graph(self, func: Callable, args: tuple, kwargs: dict) -> IRGraph:
        """Create graph with single node for opaque function."""
        node_id = self.state.next_node_id()
        output_var = self.state.next_var()
        
        # Create input variables
        input_vars = []
        for i in range(len(args)):
            var = f"_arg_{i}"
            input_vars.append(var)
        
        node = IRNode(
            id=node_id,
            operator=func,
            inputs=tuple(input_vars),
            outputs=(output_var,),
            metadata={'opaque': True}
        )
        
        self.state.nodes[node_id] = node
        self.state.var_producers[output_var] = node_id
        
        return self.build()
    
    def add_operation(self, operator: Any, inputs: List[str]) -> List[str]:
        """Add an operation to the graph."""
        node_id = self.state.next_node_id()
        
        # Generate output variables
        outputs = [self.state.next_var()]
        
        # Create node
        node = IRNode(
            id=node_id,
            operator=operator,
            inputs=tuple(inputs),
            outputs=tuple(outputs)
        )
        
        self.state.nodes[node_id] = node
        
        # Track variable producers
        for output in outputs:
            self.state.var_producers[output] = node_id
        
        return outputs
    
    def build(self) -> IRGraph:
        """Construct the final immutable graph."""
        # Build edges from variable dependencies
        edges = {}
        
        for node_id, node in self.state.nodes.items():
            # Find producers of this node's inputs
            producers = set()
            for input_var in node.inputs:
                producer_id = self.state.var_producers.get(input_var)
                if producer_id and producer_id != node_id:
                    producers.add(producer_id)
            
            # Add edges from producers to this node
            for producer_id in producers:
                if producer_id not in edges:
                    edges[producer_id] = set()
                edges[producer_id].add(node_id)
        
        # Convert to frozen sets for immutability
        frozen_edges = {k: frozenset(v) for k, v in edges.items()}
        
        return IRGraph(
            nodes=dict(self.state.nodes),
            edges=frozen_edges
        )