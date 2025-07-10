"""Decorators for creating Ember operators.

This module provides decorators that enable the simplest form of operator
creation in Ember. The @op decorator allows developers to turn plain Python
functions into full Ember operators, supporting the principle of progressive
disclosure where simple tasks should be simple.

The decorators in this module are designed to:
- Minimize boilerplate for common use cases
- Preserve function signatures and documentation
- Provide full compatibility with JAX transformations
- Enable gradual migration from functions to complex operators

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

from typing import Callable

from ember.operators.base import Operator


def op(fn: Callable) -> Operator:
    """Transform a function into an Ember operator.
    
    This decorator provides the simplest way to create operators in Ember,
    perfect for prototyping and simple use cases. The decorated function
    becomes a full Ember operator that can be composed, transformed with
    JAX, and integrated into larger systems.
    
    The function's signature is preserved, making it callable exactly as
    before decoration. Behind the scenes, it's wrapped in an Operator
    subclass with the function as its forward() method.
    
    Args:
        fn: A callable to convert to an operator. The function can:
            - Accept any arguments
            - Return any type
            - Use Ember models, tools, or any Python code
            - Contain JAX operations (though typically used for simpler cases)
        
    Returns:
        An Operator instance that wraps the function. The operator:
            - Can be called with the same signature as the original function
            - Works with JAX transformations (jit, vmap, grad, etc.)
            - Can be composed with other operators
            - Preserves the function's name and docstring
        
    Examples:
        Basic sentiment classification:
        
        >>> import ember
        >>> 
        >>> @ember.op
        ... def classify_sentiment(text: str) -> str:
        ...     '''Classify text sentiment as positive, negative, or neutral.'''
        ...     model = ember.model("gpt-4")
        ...     response = model(f"Classify sentiment: {text}")
        ...     return response.text
        >>> 
        >>> # Use like a regular function
        >>> result = classify_sentiment("This product is amazing!")
        >>> print(result)  # "positive"
        
        Function with multiple parameters:
        
        >>> @ember.op
        ... def summarize(text: str, max_words: int = 50) -> str:
        ...     '''Summarize text in specified number of words.'''
        ...     model = ember.model("gpt-4", temperature=0.3)
        ...     prompt = f"Summarize in {max_words} words: {text}"
        ...     return model(prompt).text
        >>> 
        >>> summary = summarize(article_text, max_words=100)
        
        Composing decorated functions:
        
        >>> @ember.op
        ... def extract_topics(text: str) -> List[str]:
        ...     # Extract main topics from text
        ...     ...
        >>> 
        >>> @ember.op 
        ... def analyze_document(text: str) -> dict:
        ...     return {
        ...         "sentiment": classify_sentiment(text),
        ...         "summary": summarize(text, max_words=50),
        ...         "topics": extract_topics(text)
        ...     }
        
        Integration with JAX (though typically for more complex cases):
        
        >>> import jax
        >>> 
        >>> # Batch processing
        >>> texts = ["text1", "text2", "text3"]
        >>> results = jax.vmap(classify_sentiment)(texts)
        >>> 
        >>> # Note: Most JAX transformations are more useful with
        >>> # operators containing learnable parameters (JAX arrays)
    
    See Also:
        ember.core.operators.base.Operator: The base operator class
        ember.api.model: For creating model bindings used in operators
    """
    class FunctionOperator(Operator):
        """Operator wrapper for the decorated function.
        
        This internal class wraps the function in an Operator interface,
        delegating all calls to the original function.
        """
        
        def forward(self, *args, **kwargs):
            """Execute the wrapped function.
            
            All arguments are passed through unchanged to the original function.
            """
            return fn(*args, **kwargs)
    
    # Preserve function metadata
    FunctionOperator.__name__ = f"FunctionOperator({fn.__name__})"
    FunctionOperator.__doc__ = fn.__doc__
    
    # Create and return instance
    return FunctionOperator()


__all__ = ["op"]