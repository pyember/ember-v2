"""Simplified operator API with progressive disclosure.

This module provides three levels of operator definition:
1. Simple functions (90% of use cases)
2. Validated functions with optional type checking (9% of use cases)  
3. Full specifications for complex requirements (1% of use cases)

Example:
    # Level 1: Just a function
    def process_text(text):
        return models("gpt-4", f"Summarize: {text}")
    
    # Level 2: With validation
    @validate(input=str, output=str)
    def process_validated(text: str) -> str:
        return models("gpt-4", f"Summarize: {text}")
    
    # Level 3: Full specification (when needed)
    class ComplexOperator:
        spec = Specification(...)
        def __call__(self, inputs):
            # Complex logic
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps
import inspect
from dataclasses import dataclass

# Type variables for generic functions
T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# Level 1: Simple operators - just functions
# No imports needed, no base classes, just write Python


# Level 2: Optional validation decorator
def validate(
    input: Optional[type] = None,
    output: Optional[type] = None,
    examples: Optional[List[tuple]] = None,
    description: Optional[str] = None
) -> Callable[[Callable], Callable]:
    """Decorator for adding optional validation to operators.
    
    Progressive enhancement - only validates if types are specified.
    
    Args:
        input: Expected input type (optional)
        output: Expected output type (optional)
        examples: Input/output examples for testing (optional)
        description: Human-readable description (optional)
        
    Returns:
        Decorated function with optional validation
        
    Example:
        @validate(input=str, output=str, examples=[("hello", "HELLO")])
        def uppercase(text: str) -> str:
            return text.upper()
    """
    def decorator(func: Callable) -> Callable:
        # Store validation metadata
        func._validation = {
            "input": input,
            "output": output,
            "examples": examples,
            "description": description or func.__doc__
        }
        
        if not (input or output):
            # No validation requested, return function as-is
            return func
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Only validate if types were specified
            if input and args:
                arg_value = args[0]
                if not isinstance(arg_value, input):
                    raise TypeError(
                        f"{func.__name__} expected {input.__name__}, "
                        f"got {type(arg_value).__name__}"
                    )
            
            result = func(*args, **kwargs)
            
            if output and result is not None:
                if not isinstance(result, output):
                    raise TypeError(
                        f"{func.__name__} expected to return {output.__name__}, "
                        f"got {type(result).__name__}"
                    )
                    
            return result
        
        # Preserve validation metadata
        wrapper._validation = func._validation
        return wrapper
        
    return decorator


# Level 3: Specification for complex cases
@dataclass
class Specification:
    """Full specification for complex operators.
    
    Only use this when you need:
    - Complex input/output schemas
    - Detailed validation rules
    - Custom prompt templates
    - Advanced error handling
    
    For simple cases, just use functions or @validate decorator.
    """
    input_schema: Optional[Dict[str, type]] = None
    output_schema: Optional[Dict[str, type]] = None
    examples: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None
    prompt_template: Optional[str] = None
    
    def validate_input(self, data: Any) -> None:
        """Validate input against schema if defined."""
        if not self.input_schema:
            return
            
        if not isinstance(data, dict):
            raise TypeError("Specification expects dict input")
            
        for key, expected_type in self.input_schema.items():
            if key not in data:
                raise ValueError(f"Missing required input field: {key}")
            if not isinstance(data[key], expected_type):
                raise TypeError(
                    f"Field {key} expected {expected_type.__name__}, "
                    f"got {type(data[key]).__name__}"
                )
    
    def validate_output(self, data: Any) -> None:
        """Validate output against schema if defined."""
        if not self.output_schema:
            return
            
        if not isinstance(data, dict):
            raise TypeError("Specification expects dict output")
            
        for key, expected_type in self.output_schema.items():
            if key not in data:
                raise ValueError(f"Missing required output field: {key}")
            if not isinstance(data[key], expected_type):
                raise TypeError(
                    f"Field {key} expected {expected_type.__name__}, "
                    f"got {type(data[key]).__name__}"
                )


# Composition utilities
def chain(*operators: Callable) -> Callable:
    """Chain operators sequentially.
    
    Args:
        *operators: Functions to chain together
        
    Returns:
        Composed function that applies operators in sequence
        
    Example:
        process = chain(tokenize, embed, classify)
        result = process(text)
    """
    def chained(x):
        for op in operators:
            x = op(x)
        return x
    
    # Preserve metadata from first operator
    if hasattr(operators[0], "_validation"):
        chained._validation = operators[0]._validation
        
    return chained


def parallel(*operators: Callable) -> Callable:
    """Run operators in parallel.
    
    Args:
        *operators: Functions to run in parallel
        
    Returns:
        Function that returns list of results
        
    Example:
        analyze = parallel(sentiment, entities, summary)
        results = analyze(text)  # [sentiment, entities, summary]
    """
    def paralleled(x):
        # Simple synchronous version for now
        # Can be enhanced with async/threading later
        return [op(x) for op in operators]
    
    return paralleled


def ensemble(
    *operators: Callable,
    reducer: Optional[Callable[[List[Any]], Any]] = None
) -> Callable:
    """Ensemble of operators with optional reduction.
    
    Args:
        *operators: Functions to ensemble
        reducer: Function to reduce results (default: return list)
        
    Returns:
        Ensemble function
        
    Example:
        classify = ensemble(
            model1_classify,
            model2_classify,
            model3_classify,
            reducer=majority_vote
        )
    """
    def ensembled(x):
        results = [op(x) for op in operators]
        if reducer:
            return reducer(results)
        return results
        
    return ensembled


# Helper to check if something is an operator
def is_operator(obj: Any) -> bool:
    """Check if object can be used as an operator.
    
    In our simplified design, any callable is an operator.
    """
    return callable(obj)


# Convert class-based operators to functions
def to_function(operator: Any) -> Callable:
    """Convert any operator to a simple function.
    
    Handles:
    - Functions (returned as-is)
    - Objects with __call__ method
    - Legacy Operator subclasses
    """
    if inspect.isfunction(operator):
        return operator
    elif hasattr(operator, "__call__"):
        return operator
    else:
        raise TypeError(f"Cannot convert {type(operator).__name__} to function")


# Backward compatibility helpers
def create_operator(func: Callable, **kwargs) -> Callable:
    """Create an operator from a function.
    
    In the new design, functions ARE operators, so this just
    returns the function, optionally adding validation.
    """
    if "input" in kwargs or "output" in kwargs:
        return validate(**kwargs)(func)
    return func


# Export the simple, clean API
__all__ = [
    # Core decorator
    "validate",
    
    # Specification for complex cases
    "Specification",
    
    # Composition utilities
    "chain",
    "parallel", 
    "ensemble",
    
    # Helpers
    "is_operator",
    "to_function",
    "create_operator",
]