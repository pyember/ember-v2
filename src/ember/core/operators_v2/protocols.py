"""Core protocols for the Ember v2 operator system.

Following the principle of "protocols over base classes", these define
the minimal contracts for operators without forcing inheritance.
"""

from typing import Protocol, TypeVar, runtime_checkable, Callable, Any, Dict, List

# Type variables with minimal constraints
T = TypeVar('T')
S = TypeVar('S') 
F = TypeVar('F', bound=Callable[..., Any])


@runtime_checkable
class Operator(Protocol[T, S]):
    """Something that transforms T to S. That's all.
    
    This protocol defines the minimal contract for an operator.
    Any callable that takes T and returns S is an operator.
    
    Example:
        def double(x: int) -> int:
            return x * 2
        
        # double is now an Operator[int, int]
        assert isinstance(double, Operator)
        
        class Tokenizer:
            def __init__(self, vocab: Dict[str, int]):
                self.vocab = vocab
            
            def __call__(self, text: str) -> List[int]:
                return [self.vocab.get(word, 0) for word in text.split()]
        
        # Tokenizer instances are Operator[str, List[int]]
        tokenizer = Tokenizer(vocab)
        assert isinstance(tokenizer, Operator)
    """
    
    def __call__(self, input: T) -> S:
        """Transform input of type T to output of type S."""
        ...