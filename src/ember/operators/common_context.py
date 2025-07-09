"""Context-aware operators for preserving metadata through operator chains.

This module provides operators and models for context management - preserving
arbitrary metadata (like original queries, user IDs, processing history) as
data flows through operator chains.
"""

from typing import Dict, Any, Optional, List

from ember.operators.base import Operator
from ember._internal.types import EmberModel


class ContextualData(EmberModel):
    """Data wrapper that carries arbitrary context through operator chains.
    
    This model allows operators to preserve any kind of contextual information
    while processing data. Unlike rigid schemas, the context is a flexible
    dictionary that can hold any metadata needed by downstream operators.
    
    The same type is used for both input and output, enabling seamless chaining
    of context-aware operators without validation issues.
    
    Attributes:
        context: Dictionary containing arbitrary contextual information.
        data: The actual data being processed.
    
    Examples:       
        Multi-step processing with context:
        
        >>> # Context flows through processing pipeline
        >>> preprocessing_result = ContextualData(
        ...     context={
        ...         "original_query": "Analyze this sentiment",
        ...         "preprocessing_steps": ["tokenize", "normalize"],
        ...         "confidence_threshold": 0.8
        ...     },
        ...     data="This product is amazing!"
        ... )
        
        Context augmentation during processing:
        
        >>> @operator.op
        >>> def my_contextual_operator(input: ContextualData) -> ContextualData:
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
        ...     return ContextualData(
        ...         context=new_context,
        ...         data=processed
        ...     )
        
        Context-aware analysis:
        
        >>> # Downstream operator can use rich context
        >>> @operator.op
        >>> def context_aware_analyzer(input: ContextualData) -> ContextualData:
        ...     if input.context.get("confidence", 0) < 0.8:
        ...         result = "Low confidence result needs review"
        ...     elif "original_query" in input.context:
        ...         # Verify result against original intent
        ...         result = verify_against_query(
        ...             input.context["original_query"], 
        ...             input.data
        ...         )
        ...     else:
        ...         result = input.data
        ...     
        ...     return ContextualData(
        ...         context=input.context,
        ...         data=result
        ...     )
    """
    context: Dict[str, Any]
    data: Any

class ContextualChain(Operator):
    """Chain that automatically handles context propagation based on operator needs.
    
    This operator provides a simplified API for building chains with context flow.
    Operators can declare `needs_context = True` to receive full ContextualData,
    while others automatically get just the data portion. The chain handles
    context initialization, usage tracking, and proper wrapping automatically.
    
    Attributes:
        operators: List of operators to chain together.
        track_usage: Whether to automatically track usage metrics.
        initial_context: Initial context values to set up.
    
    Examples:
        Basic usage tracking chain:
        
        >>> chain = ContextualChain([
        ...     ModelCall("gpt-4", system="Think step by step"),
        ...     ModelCall("gpt-4", system="Summarize the above")
        ... ], track_usage=True)
        >>> result = chain("What is 2+2?")
        >>> # result.context["usage_metrics"] contains accumulated usage
        
        Mixed context-aware and context-agnostic operators:
        
        >>> # VerifierOperator has needs_context = True
        >>> chain = ContextualChain([
        ...     ModelCall("gpt-4"),          # Gets just data
        ...     VerifierOperator()           # Gets full ContextualData
        ... ], track_usage=True)
        
        Custom initial context:
        
        >>> chain = ContextualChain([
        ...     ModelCall("gpt-4")
        ... ], track_usage=True, initial_context={"user_id": "123"})
    """
    
    output_spec = ContextualData
    
    operators: List[Operator]
    track_usage: bool
    initial_context: Dict[str, Any]
    
    def __init__(self, 
                 operators: List[Operator],
                 track_usage: bool = False,
                 initial_context: Optional[Dict[str, Any]] = None):
        """Initialize contextual chain.
        
        Args:
            operators: List of operators to chain together.
            track_usage: Whether to automatically track usage metrics.
            initial_context: Initial context values to set up.
        """
        self.operators = operators
        self.track_usage = track_usage
        self.initial_context = initial_context or {}
    
    def forward(self, input: Any) -> ContextualData:
        """Execute operators in sequence with automatic context handling.
        
        Args:
            input: Input data (can be regular data or ContextualData).
            
        Returns:
            ContextualData with final result and accumulated context.
        """
        # Initialize context if needed
        if not isinstance(input, ContextualData):
            context = self.initial_context.copy()
            context["original_query"] = input
            
            # Set up usage tracking if enabled
            if self.track_usage:
                context["usage_metrics"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0
                }
            
            current = ContextualData(context=context, data=input)
        else:
            current = input
        
        # Process through operators
        for operator in self.operators:
            needs_context = getattr(operator, 'needs_context', False)
            
            if needs_context:
                # Pass full ContextualData to context-aware operator
                current = operator(current)
            else:
                # Extract data, pass to operator, re-wrap with context
                result_data = operator(current.data)
                
                # Accumulate usage metrics if tracking is enabled
                new_context = current.context.copy()
                if self.track_usage and hasattr(result_data, 'usage') and result_data.usage:
                    usage = result_data.usage
                    new_context['usage_metrics']['prompt_tokens'] += usage.get('prompt_tokens', 0)
                    new_context['usage_metrics']['completion_tokens'] += usage.get('completion_tokens', 0)
                    new_context['usage_metrics']['total_tokens'] += usage.get('total_tokens', 0)
                    new_context['usage_metrics']['total_cost'] += usage.get('cost', 0.0)
                
                current = ContextualData(context=new_context, data=result_data)
        
        return current


# Convenience function for creating contextual chains
def contextual_chain(operators: List[Operator], **kwargs) -> ContextualChain:
    """Create a contextual chain with automatic context handling.
    
    Args:
        operators: List of operators to chain together.
        **kwargs: Additional arguments for ContextualChain constructor.
        
    Returns:
        ContextualChain operator.
    """
    return ContextualChain(operators, **kwargs) 