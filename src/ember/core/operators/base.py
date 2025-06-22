"""Base operator class for Ember.

This module provides the foundational Operator class that serves as the basis
for all AI components in Ember. Operators enable composition, transformation,
and optimization of AI systems through a clean, unified interface.

The design follows key principles:
- Progressive disclosure: Start simple, add complexity only when needed
- Zero configuration: Works out of the box with sensible defaults
- Full JAX compatibility: Seamless integration with automatic differentiation
- Type safety: Optional validation through Pydantic models

Key classes:
    Operator: Base class for all Ember operators with optional validation.

Typical usage example:
    from ember.core.operators import Operator
    from ember.api import model
    
    class Summarizer(Operator):
        def __init__(self):
            self.model = model("gpt-4")
        
        def forward(self, text: str) -> str:
            return self.model(f"Summarize: {text}")
    
    summarizer = Summarizer()
    summary = summarizer("Long document text...")

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

from typing import Optional, Type, get_type_hints

from ember.core.module import Module


class Operator(Module):
    """Base operator class for building composable AI systems.
    
    Operators are the fundamental building blocks in Ember, providing a clean
    abstraction for AI components that can be composed, transformed, and
    optimized. They support both static components (model API calls, tools)
    and dynamic components (learnable JAX arrays).
    
    The design follows the principle of progressive disclosure:
    - Simple operators require only a forward() method
    - Validation can be added via optional input/output specifications
    - JAX arrays are automatically detected as learnable parameters
    
    Attributes:
        input_spec: Optional type specification for input validation.
            When provided, inputs are automatically validated before forward().
        output_spec: Optional type specification for output validation.
            When provided, outputs are automatically validated after forward().
    
    Examples:
        Basic operator for text classification:
        
        >>> class Classifier(Operator):
        ...     def __init__(self, model_name: str = "gpt-4"):
        ...         self.model = ember.model(model_name)
        ...     
        ...     def forward(self, text: str) -> str:
        ...         response = self.model(f"Classify sentiment: {text}")
        ...         return response.text
        >>> 
        >>> classifier = Classifier()
        >>> result = classifier("This product is excellent!")
        >>> print(result)  # "positive"
        
        Operator with input/output validation:
        
        >>> from ember.core.types import EmberModel
        >>> 
        >>> class TextInput(EmberModel):
        ...     text: str
        ...     max_length: int = 1000
        >>> 
        >>> class SentimentOutput(EmberModel):
        ...     label: str
        ...     confidence: float
        >>> 
        >>> class ValidatedClassifier(Operator):
        ...     input_spec = TextInput
        ...     output_spec = SentimentOutput
        ...     
        ...     def __init__(self):
        ...         self.model = ember.model("gpt-4", temperature=0)
        ...     
        ...     def forward(self, input: TextInput) -> SentimentOutput:
        ...         # Input is pre-validated as TextInput
        ...         prompt = f"Analyze sentiment: {input.text[:input.max_length]}"
        ...         response = self.model(prompt)
        ...         
        ...         # Parse response and return validated output
        ...         return SentimentOutput(
        ...             label="positive",
        ...             confidence=0.95
        ...         )
        >>> 
        >>> # Accepts dict input, automatically validates to TextInput
        >>> classifier = ValidatedClassifier()
        >>> result = classifier({"text": "Great product!"})
        >>> print(result.label)  # "positive"
        
        Operator with learnable parameters (JAX integration):
        
        >>> import jax.numpy as jnp
        >>> 
        >>> class LearnableClassifier(Operator):
        ...     def __init__(self, num_classes: int, key: jax.Array):
        ...         self.model = ember.model("gpt-4")
        ...         # JAX arrays are automatically detected as dynamic
        ...         self.class_weights = jax.random.normal(key, (num_classes,))
        ...         self.threshold = jnp.array(0.5)
        ...     
        ...     def forward(self, text: str) -> int:
        ...         # Get model prediction
        ...         response = self.model(f"Classify: {text}")
        ...         
        ...         # Apply learnable weights (differentiable)
        ...         scores = self.compute_scores(response.text)
        ...         weighted_scores = scores * self.class_weights
        ...         
        ...         # Return highest scoring class
        ...         return jnp.argmax(weighted_scores)
        >>> 
        >>> # Works with JAX transformations
        >>> classifier = LearnableClassifier(3, jax.random.PRNGKey(0))
        >>> grads = jax.grad(loss_fn)(classifier)
        >>> # grads.class_weights has gradients
        >>> # grads.model is None (static field)
    
    See Also:
        ember.api.decorators.op: Simple decorator for function-style operators
        ember.core.module.Module: Base module class providing JAX compatibility
    """
    
    # Optional specifications only
    input_spec: Optional[Type] = None
    output_spec: Optional[Type] = None
    
    def forward(self, input):
        """Process input to produce output.
        
        This method contains the core logic of the operator and must be
        implemented by subclasses. It receives validated input (if input_spec
        is provided) and should return output that matches output_spec (if
        provided).
        
        Args:
            input: The input to process. If input_spec is defined, this will
                be a validated instance of that type. Otherwise, it's the raw
                input passed to __call__.
                
        Returns:
            The processed output. Should match output_spec if defined.
            
        Raises:
            NotImplementedError: Always raised by base class. Subclasses must
                override this method.
                
        Note:
            This method should be pure in terms of side effects on the operator
            state. Any stateful operations should be explicit (e.g., updating
            JAX arrays through optimization).
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, input):
        """Process input through the operator with optional validation.
        
        This method handles the complete operator execution pipeline:
        1. Validates input against input_spec (if provided)
        2. Calls forward() with validated input
        3. Validates output against output_spec (if provided)
        4. Returns the final output
        
        The validation is transparent - if no specs are provided, input and
        output pass through unchanged. This supports progressive disclosure
        from simple to validated operators.
        
        Args:
            input: The input to process. Can be:
                - Raw input of any type (if no input_spec)
                - Dict that can be validated to input_spec
                - Instance of input_spec type
                
        Returns:
            The processed output. Will be:
                - Raw output from forward() (if no output_spec)
                - Validated instance of output_spec type
                
        Raises:
            ValidationError: If input doesn't match input_spec or output
                doesn't match output_spec (when specs are provided).
                
        Examples:
            >>> # Without validation specs
            >>> op = SimpleOperator()
            >>> result = op("raw string input")
            >>> 
            >>> # With validation specs
            >>> op = ValidatedOperator()  # Has input_spec and output_spec
            >>> result = op({"field": "value"})  # Dict validated to input_spec
            >>> isinstance(result, op.output_spec)  # True
        """
        # Get validation spec (explicit or from type hints)
        input_validator = self.input_spec
        output_validator = self.output_spec
        
        # If no explicit spec, try type hints
        if input_validator is None or output_validator is None:
            try:
                hints = get_type_hints(self.forward)
                if input_validator is None and 'input' in hints:
                    hint = hints['input']
                    # Check if it's a Pydantic model
                    if hasattr(hint, 'model_validate'):
                        input_validator = hint
                if output_validator is None and 'return' in hints:
                    hint = hints['return']
                    if hasattr(hint, 'model_validate'):
                        output_validator = hint
            except (NameError, AttributeError):
                # Type hints might reference undefined types
                pass
        
        # Apply input validation
        if input_validator and hasattr(input_validator, 'model_validate'):
            if isinstance(input, dict):
                input = input_validator.model_validate(input)
            elif not isinstance(input, input_validator):
                input = input_validator.model_validate(input)
        
        output = self.forward(input)
        
        # Apply output validation
        if output_validator and hasattr(output_validator, 'model_validate'):
            if isinstance(output, dict):
                output = output_validator.model_validate(output)
            elif not isinstance(output, output_validator):
                output = output_validator.model_validate(output)
            
        return output


__all__ = ["Operator"]