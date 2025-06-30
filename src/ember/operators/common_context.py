"""Context-aware operators for preserving metadata through operator chains.

This module provides operators and models for context management - preserving
arbitrary metadata (like original queries, user IDs, processing history) as
data flows through operator chains.

Core pattern: InitialContext() -> ContextAgnostic(operators) -> ContextAware(operators)
"""

from typing import Dict, Any, Callable, Optional

from ember.operators.base import Operator
from ember._internal.types import EmberModel


class ContextualInput(EmberModel):
    """Input wrapper that carries arbitrary context through operator chains.
    
    This model allows operators to preserve any kind of contextual information
    while processing data. Unlike rigid schemas, the context is a flexible
    dictionary that can hold any metadata needed by downstream operators.
    
    Attributes:
        context: Dictionary containing arbitrary contextual information.
        data: The actual input data for processing.
    
    Examples:       
        Multi-step processing with context:
        
        >>> # Context flows through processing pipeline
        >>> preprocessing_result = ContextualInput(
        ...     context={
        ...         "original_query": "Analyze this sentiment",
        ...         "preprocessing_steps": ["tokenize", "normalize"],
        ...         "confidence_threshold": 0.8
        ...     },
        ...     data="This product is amazing!"
        ... )
        
        Ensemble verification with rich context:
        
        >>> # Context helps verifier understand the full picture
        >>> verification_input = ContextualInput(
        ...     context={
        ...         "original_query": "Is this review positive?",
        ...         "models_used": ["gpt-4", "claude-3", "gemini"],
        ...         "ensemble_agreement": 0.95,
        ...         "processing_time": 1.2
        ...     },
        ...     data=["positive", "positive", "positive"]
        ... )
    """
    context: Dict[str, Any]
    data: Any


class ContextualOutput(EmberModel):
    """Output wrapper that preserves context through operator chains.
    
    This model maintains contextual information alongside operator outputs,
    enabling sophisticated downstream processing, debugging, and analysis.
    The context can be augmented by each operator in the chain.
    
    Attributes:
        context: Dictionary containing contextual information, potentially
            augmented by the operator that produced this output.
        data: The actual output data from the operator.
    
    Examples:
        Basic context preservation:
        
        >>> @operator.op
        >>> def my_operator_with_context(input: ContextualInput) -> ContextualOutput:
        ...     processed = process_data(input.data)
        ...     
        ...     # Augment context with processing metadata
        ...     new_context = input.context.copy()
        ...     new_context.update({
        ...         "processing_time": 0.5,
        ...         "confidence": 0.92,
        ...         "model_version": "v2.1"
        ...     })
        ...     
        ...     return ContextualOutput(
        ...         context=new_context,
        ...         data=processed
        ...     )
        
        Context-aware result analysis:
        
        >>> # Downstream operator can use rich context
        >>> @operator.op
        >>> def context_aware_analyzer(output: ContextualOutput):
        ...     if output.context.get("confidence", 0) < 0.8:
        ...         return "Low confidence result needs review"
        ...     
        ...     if "original_query" in output.context:
        ...         # Verify result against original intent
        ...         return verify_against_query(
        ...             output.context["original_query"], 
        ...             output.data
        ...         )
        ...     
        ...     return output.data
    """
    context: Dict[str, Any]
    data: Any


class ContextAgnostic(Operator):
    """Wrapper that lets context-unaware operators work in contextual chains.
    
    This operator extracts data from ContextualInput, passes it to a wrapped
    operator that doesn't care about context, then wraps the result back in
    ContextualOutput with preserved/transformed context.
    
    Attributes:
        operator: The context-unaware operator to wrap.
        context_transform: Optional function to transform context.
    
    Examples:
        Basic context preservation around unaware operator:
        
        >>> # ModelCall doesn't know about context, but we want to preserve it
        >>> contextual_model = ContextAgnostic(ModelCall("gpt-4"))
        >>> 
        >>> # Context flows through automatically
        >>> input_with_context = ContextualInput(
        ...     context={"user_id": "123", "original_query": "Classify this"},
        ...     data="This movie is great!"
        ... )
        >>> result = contextual_model(input_with_context)
        >>> # result.context == {"user_id": "123", "original_query": "Classify this"}
        >>> # result.data.text == "positive"
        
        Transform context during processing:
        
        >>> def add_metadata(context, input_data, output_data):
        ...     return {
        ...         **context,
        ...         "processing_time": 0.5,
        ...         "input_length": len(str(input_data)),
        ...         "model_used": "gpt-4"
        ...     }
        >>> 
        >>> enhanced_model = ContextAgnostic(
        ...     ModelCall("gpt-4"),
        ...     context_transform=add_metadata
        ... )
        
        Chain with mix of context-aware and context-agnostic:
        
        >>> chain = Chain([
        ...     InitialContext(user_id="123"),
        ...     ContextAgnostic(ModelCall("gpt-4")),  # Ignores context
        ...     ContextAware(VerifierOperator())  # Uses context directly
        ... ])
    """
    
    input_spec = ContextualInput
    output_spec = ContextualOutput
    
    operator: Operator
    context_transform: Optional[Callable[[Dict[str, Any], Any, Any], Dict[str, Any]]]
    
    def __init__(self, 
                 operator: Operator,
                 context_transform: Optional[Callable[[Dict[str, Any], Any, Any], Dict[str, Any]]] = None):
        """Initialize context-agnostic wrapper.
        
        Args:
            operator: The context-unaware operator to wrap.
            context_transform: Optional function to transform context. Takes
                (input_context, input_data, output_data) and returns new context.
        """
        self.operator = operator
        self.context_transform = context_transform
    
    def forward(self, input: ContextualInput) -> ContextualOutput:
        """Execute wrapped operator while managing context flow.
        
        Args:
            input: Input with context and data.
            
        Returns:
            Output with preserved/transformed context and operator result.
        """
        # Execute wrapped operator with just the data
        result = self.operator(input.data)
        
        # Transform context if function provided
        if self.context_transform:
            final_context = self.context_transform(input.context, input.data, result)
        else:
            final_context = input.context.copy()
        
        return ContextualOutput(
            context=final_context,
            data=result
        )


class ContextAware(Operator):
    """Wrapper that ensures operators receive full contextual input.
    
    This operator passes the complete ContextualInput to the wrapped operator,
    which must be designed to handle context. Useful for making the context
    requirement explicit and providing input validation.
    
    Attributes:
        operator: The context-aware operator to wrap.
        context_transform: Optional function to transform output context.
    
    Examples:
        Explicitly mark operator as context-aware:
        
        >>> # VerifierOperator expects ContextualInput
        >>> verifier = ContextAware(VerifierOperator())
        >>> 
        >>> # Input validation ensures we have proper context
        >>> result = verifier(ContextualInput(
        ...     context={"original_query": "What is AI?"},
        ...     data=["Response 1", "Response 2"]
        ... ))
        
        Transform context on output:
        
        >>> def add_verification_metadata(context, input_data, output_data):
        ...     return {
        ...         **context,
        ...         "verified_at": datetime.now().isoformat(),
        ...         "verification_score": 0.95
        ...     }
        >>> 
        >>> enhanced_verifier = ContextAware(
        ...     VerifierOperator(),
        ...     context_transform=add_verification_metadata
        ... )
    """
    
    input_spec = ContextualInput
    output_spec = ContextualOutput
    
    operator: Operator
    context_transform: Optional[Callable[[Dict[str, Any], Any, Any], Dict[str, Any]]]
    
    def __init__(self, 
                 operator: Operator,
                 context_transform: Optional[Callable[[Dict[str, Any], Any, Any], Dict[str, Any]]] = None):
        """Initialize context-aware wrapper.
        
        Args:
            operator: The context-aware operator to wrap.
            context_transform: Optional function to transform output context.
        """
        self.operator = operator
        self.context_transform = context_transform
    
    def forward(self, input: ContextualInput) -> ContextualOutput:
        """Execute wrapped operator with full contextual input.
        
        Args:
            input: Input with context and data.
            
        Returns:
            Output from operator, optionally with transformed context.
        """
        # Execute wrapped operator with full contextual input
        result = self.operator(input)
        
        # Transform context if function provided
        if self.context_transform:
            final_context = self.context_transform(input.context, input.data, result.data)
            return ContextualOutput(
                context=final_context,
                data=result.data
            )
        
        return result


class InitialContext(Operator):
    """Operator that bootstraps context for chains that start with regular input.
    
    This operator converts regular input into ContextualOutput, automatically
    storing the input as the original_query and allowing additional context
    to be initialized. It's designed to be the first operator in chains that
    need context flow but start with simple string inputs.
    
    Attributes:
        context_defaults: Default context values to initialize with the input.
    
    Examples:
        Basic context initialization:
        
        >>> # Start a chain with context from regular string input
        >>> chain = Chain([
        ...     InitialContext(user_id="123", session_id="abc"),
        ...     ContextAgnostic(ModelCall("gpt-4")),
        ...     ContextAware(SomeContextAwareOperator())
        ... ])
        >>> result = chain("What is the capital of France?")
        >>> # Context flows through entire chain with original_query preserved
        
        Ensemble-verifier with context initialization:
        
        >>> # Bootstrap context for ensemble-verifier pattern
        >>> ensemble_chain = Chain([
        ...     InitialContext(task_type="qa", priority="high"),
        ...     ContextAgnostic(Ensemble([
        ...         ModelCall("gpt-4"),
        ...         ModelCall("claude-3")
        ...     ])),
        ...     ContextAware(VerifierOperator())
        ... ])
        >>> result = ensemble_chain("Explain quantum computing")
        >>> # Verifier can access original_query and metadata
    """
    
    output_spec = ContextualOutput
    
    def __init__(self, **context_defaults: Any):
        """Initialize context bootstrapper.
        
        Args:
            **context_defaults: Default context values to set alongside original_query.
        """
        self.context_defaults = context_defaults

    def forward(self, input: Any) -> ContextualOutput:
        """Convert regular input to ContextualOutput with initialized context.
        
        Args:
            input: Regular input (typically string) to process.
            
        Returns:
            ContextualOutput with input stored as both original_query and data.
        """
        context = self.context_defaults.copy()
        context["original_query"] = input

        return ContextualOutput(context=context, data=input)


# Convenience functions for creating context operators

def context_agnostic(operator: Operator, **kwargs) -> ContextAgnostic:
    """Create a context-agnostic wrapper for an operator.
    
    Args:
        operator: The context-unaware operator to wrap.
        **kwargs: Additional arguments for ContextAgnostic constructor.
        
    Returns:
        ContextAgnostic operator.
    """
    return ContextAgnostic(operator, **kwargs)


def context_aware(operator: Operator, **kwargs) -> ContextAware:
    """Create a context-aware wrapper for an operator.
    
    Args:
        operator: The context-aware operator to wrap.
        **kwargs: Additional arguments for ContextAware constructor.
        
    Returns:
        ContextAware operator.
    """
    return ContextAware(operator, **kwargs) 