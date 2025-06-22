"""Experimental tracing functionality for EmberModule.

This is experimental code exploring tracing capabilities.
It's not part of the core API and may change or be removed.

Design notes:
- Completely separate from core module functionality
- Opt-in via decorator
- Simple and explicit
- May add overhead
"""

from typing import Any, Callable, Dict, List


class Trace:
    """Simple execution tracer. Opt-in and explicit."""
    
    def __init__(self):
        self.events = []
        self._depth = 0
    
    def __call__(self, fn: Callable) -> Callable:
        """Decorator to trace function execution."""
        def traced(*args, **kwargs):
            self._depth += 1
            self.events.append({
                'type': 'call',
                'name': fn.__name__,
                'depth': self._depth,
                'args_types': [type(a).__name__ for a in args],
            })
            
            try:
                result = fn(*args, **kwargs)
                self.events.append({
                    'type': 'return',
                    'name': fn.__name__,
                    'depth': self._depth,
                    'result_type': type(result).__name__,
                })
                return result
            except Exception as e:
                self.events.append({
                    'type': 'error',
                    'name': fn.__name__,
                    'depth': self._depth,
                    'error': str(e),
                })
                raise
            finally:
                self._depth -= 1
        
        traced.__name__ = fn.__name__
        traced.__wrapped__ = fn
        return traced
    
    def clear(self):
        """Clear collected events."""
        self.events.clear()
        self._depth = 0
    
    def summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        calls = [e for e in self.events if e['type'] == 'call']
        errors = [e for e in self.events if e['type'] == 'error']
        
        return {
            'total_calls': len(calls),
            'total_errors': len(errors),
            'call_counts': self._count_by_name(calls),
            'error_counts': self._count_by_name(errors),
        }
    
    def _count_by_name(self, events: List[Dict]) -> Dict[str, int]:
        counts = {}
        for event in events:
            name = event['name']
            counts[name] = counts.get(name, 0) + 1
        return counts


# Global tracer instance for convenience
trace = Trace()


# Example usage:
if __name__ == "__main__":
    @trace
    def add(x, y):
        return x + y
    
    @trace
    def multiply(x, y):
        return x * y
    
    @trace
    def compute(a, b):
        return multiply(add(a, b), 2)
    
    result = compute(3, 4)
    print(f"Result: {result}")
    print(f"Summary: {trace.summary()}")
    
    # Output:
    # Result: 14
    # Summary: {'total_calls': 3, 'total_errors': 0, 
    #           'call_counts': {'compute': 1, 'add': 1, 'multiply': 1},
    #           'error_counts': {}}