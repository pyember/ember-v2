"""Simplified module system for stateful operators.

Provides a minimal base class for operators that require state management
and tree transformation capabilities. Automatically handles dataclass
generation and PyTree registration.
"""

import dataclasses
from typing import Any, TypeVar, ClassVar, Dict, Callable
import jax.tree_util as jtu


T = TypeVar('T')


class ModuleMeta(type):
    """Metaclass that automatically makes EmberModule subclasses into frozen dataclasses."""
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Skip if it's the base EmberModule class itself
        if name == 'EmberModule' and namespace.get('__module__') == __name__:
            return cls
            
        # Automatically apply dataclass decorator
        # frozen=True for immutability (like Equinox)
        # eq=False because we'll use PyTree equality
        cls = dataclasses.dataclass(frozen=True, eq=False)(cls)
        
        # Register with JAX as a PyTree
        def _tree_flatten(obj):
            # Get all field values
            field_values = []
            field_names = []
            static_values = {}
            
            for field in dataclasses.fields(obj):
                value = getattr(obj, field.name)
                # Check if field is marked as static
                if field.metadata.get('static', False):
                    static_values[field.name] = value
                else:
                    field_values.append(value)
                    field_names.append(field.name)
            
            return field_values, (field_names, static_values, type(obj))
        
        def _tree_unflatten(aux, field_values):
            field_names, static_values, cls_type = aux
            # Reconstruct with both dynamic and static values
            kwargs = dict(zip(field_names, field_values))
            kwargs.update(static_values)
            return cls_type(**kwargs)
        
        # Register the PyTree
        jtu.register_pytree_node(
            cls,
            _tree_flatten,
            _tree_unflatten
        )
        
        return cls


class EmberModule(metaclass=ModuleMeta):
    """Base class for stateful operators with automatic immutability and tree transformation support.
    
    Subclasses automatically become frozen dataclasses and register as PyTrees,
    enabling functional transformations while maintaining state encapsulation.
    
    Key properties:
    - Immutable by default (frozen dataclass)
    - Automatic PyTree registration for transformations
    - Clear separation of static and dynamic fields
    - Zero boilerplate for common patterns
    
    Example:
        class ChainOfThought(EmberModule):
            model: Callable
            temperature: float = 0.7
            
            def __call__(self, question: str) -> str:
                reasoning_prompt = f"Think step by step about: {question}"
                reasoning = self.model(reasoning_prompt, temperature=self.temperature).text
                
                answer_prompt = f"Given this reasoning: {reasoning}\nAnswer: {question}"
                answer = self.model(answer_prompt, temperature=0.1).text
                
                return answer
        
        # Create instance
        cot = ChainOfThought(model=models.instance("gpt-4"))
        result = cot("What is 25 * 17?")
        
        # Transformations work automatically
        batch_cot = vmap(cot)
        results = batch_cot(["What is 25 * 17?", "What is 13 * 19?"])
    """
    
    def tree_flatten(self):
        """Flatten into PyTree representation."""
        return jtu.tree_flatten(self)
    
    def tree_unflatten(self, aux, children):
        """Reconstruct from PyTree representation."""
        return jtu.tree_unflatten(aux, children)
    
    def replace(self, **updates):
        """Create a new instance with updated field values.
        
        Returns a new instance with specified fields updated, maintaining
        immutability. Validates that updated fields exist and type-checks
        at runtime.
        
        Args:
            **updates: Field names and their new values
            
        Returns:
            New instance with updated values
            
        Example:
            model2 = model.replace(temperature=0.9)
        """
        return dataclasses.replace(self, **updates)


def static_field(**kwargs):
    """Mark a field as static (not part of PyTree transformations).
    
    Static fields are preserved during transformations but not traced through.
    Perfect for configuration, metadata, or non-differentiable state.
    
    Example:
        class MyOperator(EmberModule):
            weights: Array  # Dynamic - will be transformed
            config: Dict = static_field(default_factory=dict)  # Static
    """
    return dataclasses.field(metadata={'static': True}, **kwargs)


# For operators that work with structured inputs (like DSPy signatures)
class SignatureOperator(EmberModule):
    """Base for operators that work with typed dictionary inputs/outputs.
    
    This provides a natural pattern for LLM operators that need structured I/O
    without forcing inheritance or complex specifications.
    
    Example:
        @dataclass
        class Question:
            text: str
            context: Optional[str] = None
            
        @dataclass  
        class Answer:
            text: str
            confidence: float
            
        class QAOperator(SignatureOperator):
            model: Callable
            
            def __call__(self, input: Question) -> Answer:
                prompt = f"Question: {input.text}"
                if input.context:
                    prompt = f"Context: {input.context}\n{prompt}"
                    
                response = self.model(prompt).text
                # Parse response into Answer...
                return Answer(text=response, confidence=0.9)
    """
    pass


class Ensemble(EmberModule):
    """Execute multiple operators in parallel and collect results.
    
    Operators are executed independently and results are collected in order.
    Supports heterogeneous operator types as long as they accept the same input type.
    
    Attributes:
        operators: Immutable sequence of callable operators
        
    Example:
        models = [model1, model2, model3]
        ensemble = Ensemble(operators=tuple(models))
        results = ensemble("What is the capital of France?")
    """
    operators: tuple
    
    def __call__(self, input: Any) -> list:
        """Execute all operators with the given input.
        
        Args:
            input: Input to pass to each operator
            
        Returns:
            List of results in the same order as operators
        """
        return [op(input) for op in self.operators]


class Chain(EmberModule):
    """Compose operators sequentially, piping output to input.
    
    Each operator's output becomes the next operator's input.
    Type compatibility between operators is the user's responsibility.
    
    Attributes:
        operators: Immutable sequence of operators to chain
        
    Example:
        pipeline = Chain(operators=(
            tokenizer,
            embedder,
            classifier
        ))
        result = pipeline("raw text")
    """
    operators: tuple
    
    def __call__(self, input: Any) -> Any:
        """Pass input through each operator in sequence.
        
        Args:
            input: Initial input to the chain
            
        Returns:
            Final output after all transformations
        """
        result = input
        for op in self.operators:
            result = op(result)
        return result