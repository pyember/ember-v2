"""Trace decorator for execution analysis and metrics.

Not for optimization - just for understanding what happened.
"""

import functools
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import asdict

from ember.xcs.tracer.xcs_tracing import TracerContext, TraceRecord


class TraceResult:
    """Result of tracing a function execution."""
    
    def __init__(self, result: Any, records: List[TraceRecord]):
        self.result = result
        self.records = records
        self._analyze()
    
    def _analyze(self):
        """Analyze trace records for useful metrics."""
        self.total_duration = sum(r.duration for r in self.records)
        self.operation_count = len(self.records)
        
        # Find slowest operations
        self.slowest_ops = sorted(
            self.records, 
            key=lambda r: r.duration, 
            reverse=True
        )[:5]
        
        # Group by operator name
        self.op_summary = {}
        for record in self.records:
            name = record.operator_name
            if name not in self.op_summary:
                self.op_summary[name] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0
                }
            self.op_summary[name]["count"] += 1
            self.op_summary[name]["total_time"] += record.duration
        
        # Calculate averages
        for name, stats in self.op_summary.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
    
    def print_summary(self):
        """Print a human-readable summary."""
        print(f"\n=== Trace Summary ===")
        print(f"Total operations: {self.operation_count}")
        print(f"Total duration: {self.total_duration*1000:.1f}ms")
        
        print(f"\nSlowest operations:")
        for op in self.slowest_ops[:3]:
            print(f"  {op.operator_name}: {op.duration*1000:.1f}ms")
        
        print(f"\nOperation breakdown:")
        for name, stats in sorted(
            self.op_summary.items(), 
            key=lambda x: x[1]["total_time"], 
            reverse=True
        ):
            print(f"  {name}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total: {stats['total_time']*1000:.1f}ms")
            print(f"    Avg: {stats['avg_time']*1000:.1f}ms")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export trace data for analysis."""
        return {
            "summary": {
                "total_duration_ms": self.total_duration * 1000,
                "operation_count": self.operation_count,
                "operation_breakdown": self.op_summary
            },
            "records": [asdict(r) for r in self.records],
            "slowest_operations": [
                {
                    "name": r.operator_name,
                    "duration_ms": r.duration * 1000,
                    "node_id": r.node_id
                }
                for r in self.slowest_ops
            ]
        }


def trace(func: Optional[Callable] = None, *, 
         save_to: Optional[str] = None,
         print_summary: bool = False) -> Callable:
    """Trace function execution for analysis.
    
    NOT for optimization - use @jit for that.
    This is for understanding execution patterns, finding bottlenecks,
    and gathering metrics.
    
    Args:
        func: Function to trace
        save_to: Optional file path to save trace data
        print_summary: Whether to print summary after execution
        
    Returns:
        Traced function that returns (result, trace_data)
        
    Example:
        @trace(print_summary=True)
        def my_pipeline(inputs):
            # Complex operations
            return result
            
        result = my_pipeline(data)
        # Prints execution summary
    """
    def decorator(fn):
        @functools.wraps(fn)
        def traced_function(*args, **kwargs):
            # Use XCS tracer to record execution
            with TracerContext() as tracer:
                # Execute function
                start_time = time.time()
                result = fn(*args, **kwargs)
                total_time = time.time() - start_time
                
                # Create trace result
                trace_result = TraceResult(result, tracer.records)
                
                # Add total execution time
                trace_result.total_execution_time = total_time
                
                # Print summary if requested
                if print_summary:
                    trace_result.print_summary()
                
                # Save to file if requested
                if save_to:
                    import json
                    with open(save_to, 'w') as f:
                        json.dump(trace_result.to_dict(), f, indent=2)
                
                # Store trace result as attribute for access
                traced_function.last_trace = trace_result
                
                # Return just the result (not the trace)
                return result
        
        # Add method to get last trace
        traced_function.get_trace = lambda: getattr(
            traced_function, 'last_trace', None
        )
        
        return traced_function
    
    # Handle @trace or @trace(...)
    if func is None:
        return decorator
    else:
        return decorator(func)


# Convenience function for one-off tracing
def trace_execution(func: Callable, *args, **kwargs) -> TraceResult:
    """Trace a single execution without decorator.
    
    Example:
        trace_result = trace_execution(my_function, input_data)
        trace_result.print_summary()
    """
    traced = trace(func)
    result = traced(*args, **kwargs)
    return traced.get_trace()