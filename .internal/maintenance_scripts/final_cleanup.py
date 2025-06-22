#!/usr/bin/env python3
"""Final cleanup after removing xcs_engine.py.

This script removes remaining scheduler references and simplifies execution.
"""

import os
import re
from pathlib import Path

def update_structural_jit():
    """Update structural_jit.py to remove scheduler references."""
    file_path = Path("src/ember/xcs/tracer/structural_jit.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove scheduler imports
    content = re.sub(
        r'from ember\.xcs\.engine\.xcs_engine import.*\n',
        '',
        content
    )
    
    content = re.sub(
        r'from ember\.xcs\.engine\.xcs_noop_scheduler import.*\n',
        '',
        content
    )
    
    # Remove IScheduler references
    content = re.sub(
        r'IScheduler,?\s*\n?',
        '',
        content
    )
    
    content = re.sub(
        r'TopologicalSchedulerWithParallelDispatch,?\s*\n?',
        '',
        content
    )
    
    # Replace get_scheduler function with direct graph execution
    content = re.sub(
        r'def get_scheduler\(.*?\n.*?raise ValueError\([^)]+\)\n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Replace _execute_with_engine to use graph.run() directly
    new_execute = '''def _execute_with_engine(
    graph: Graph,
    inputs: Dict[str, Any],
    config: ExecutionConfig) -> Dict[str, Any]:
    """Execute a graph directly using Graph.run().
    
    Args:
        graph: Graph to execute
        inputs: Input data
        config: Execution configuration
        
    Returns:
        Execution results
    """
    logger = logging.getLogger("ember.xcs.tracer.structural_jit")
    
    try:
        # Determine parallel execution based on config
        if config.strategy == "sequential":
            max_workers = 1
        elif config.strategy == "parallel":
            max_workers = config.max_workers
        else:  # auto
            # Use graph's wave analysis to decide
            max_workers = None if len(graph.nodes) >= config.parallel_threshold else 1
            
        logger.debug(f"Executing graph with {len(graph.nodes)} nodes")
        
        # Execute the graph
        results = graph.run(inputs, max_workers=max_workers)
        
        # Extract appropriate result
        result = _extract_result(graph, results, logger)
        return result
        
    except Exception as e:
        # Handle execution errors
        from ember.core.exceptions import OperatorExecutionError
        
        # Propagate operator errors without recovery attempts
        if isinstance(e, (OperatorExecutionError, ValueError, TypeError, RuntimeError)):
            raise
            
        # For machinery errors, try to recover with cached result if available
        if hasattr(graph, "original_result") and graph.original_result is not None:
            logger.debug(f"Recovering from JIT error: {str(e)}")
            return graph.original_result
            
        # Cannot recover - re-raise the original exception
        raise
'''
    
    # Find and replace the _execute_with_engine function
    content = re.sub(
        r'def _execute_with_engine\(.*?\n.*?raise\n',
        new_execute + '\n',
        content,
        flags=re.DOTALL
    )
    
    # Remove compile_graph references
    content = re.sub(
        r'compile_graph,?\s*\n?',
        '',
        content
    )
    
    # Remove scheduler-related code from _execute_with_engine
    content = re.sub(
        r'# Get appropriate scheduler.*?graph=graph\)',
        '# Direct execution without scheduler abstraction',
        content,
        flags=re.DOTALL
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path}")

def update_tracer_decorator():
    """Update tracer_decorator.py to remove scheduler references."""
    file_path = Path("src/ember/xcs/tracer/tracer_decorator.py")
    
    if not file_path.exists():
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove scheduler imports
    content = re.sub(
        r'TopologicalSchedulerWithParallelDispatch,?\s*\n?',
        '',
        content
    )
    
    # Replace scheduler usage with direct graph execution
    content = re.sub(
        r'scheduler=TopologicalSchedulerWithParallelDispatch\(\)',
        'parallel=True',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path}")

def remove_noop_scheduler():
    """Remove the xcs_noop_scheduler.py file."""
    file_path = Path("src/ember/xcs/engine/xcs_noop_scheduler.py")
    
    if file_path.exists():
        os.remove(file_path)
        print(f"Removed {file_path}")

def update_parallel_pipeline_example():
    """Update the parallel pipeline example to use Graph directly."""
    file_path = Path("src/ember/examples/legacy/advanced/parallel_pipeline_example.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove scheduler imports
    content = re.sub(
        r'    TopologicalSchedulerWithParallelDispatch,\n',
        '',
        content
    )
    
    # Update execute_graph usage
    content = re.sub(
        r'from ember\.xcs\.graph import Graph',
        'from ember.xcs.graph import Graph, execute_graph',
        content
    )
    
    # Replace scheduler usage with parallel parameter
    content = re.sub(
        r'scheduler = TopologicalSchedulerWithParallelDispatch.*?\n',
        '',
        content
    )
    
    content = re.sub(
        r'result = execute_graph\(\s*graph=graph,\s*global_input=.*?,\s*scheduler=scheduler\s*\)',
        'result = execute_graph(graph=graph, inputs={"query": query}, parallel=workers)',
        content
    )
    
    # Fix the imports section
    content = re.sub(
        r'# Import internals only when needed for advanced usage\n.*?\n',
        '# Import internals only when needed for advanced usage\n',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path}")

def check_remaining_references():
    """Check for any remaining references to removed components."""
    patterns = [
        "XCSTask",
        "XCSPlan", 
        "ExecutionMetrics",
        "GraphExecutor",
        "TopologicalScheduler",
        "XCSNoOpScheduler",
        "IScheduler",
        "compile_graph"
    ]
    
    print("\nChecking for remaining references...")
    
    for pattern in patterns:
        result = os.popen(f'grep -r "{pattern}" src/ember/xcs/ --include="*.py" | grep -v "test_" | wc -l').read().strip()
        if result != "0":
            print(f"  Found {result} references to {pattern}")

def main():
    print("Starting final cleanup...")
    
    # Update files
    update_structural_jit()
    update_tracer_decorator()
    update_parallel_pipeline_example()
    
    # Remove remaining files
    remove_noop_scheduler()
    
    # Check for remaining references
    check_remaining_references()
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    main()