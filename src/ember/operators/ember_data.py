"""EmberData class for threading context through operators.

This module provides EmberData, which is like PyTorch tensors but for LLM 
operations. EmberData carries both the actual data and metadata (usage metrics,
original prompt) through the computation graph.

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class UsageMetrics:
    """Track usage metrics across operations."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    
    def accumulate(self, other_usage: Dict[str, Any]) -> 'UsageMetrics':
        """Accumulate usage from another usage dict."""
        if not other_usage:
            return self
        
        return UsageMetrics(
            prompt_tokens=self.prompt_tokens + other_usage.get('prompt_tokens', 0),
            completion_tokens=self.completion_tokens + other_usage.get('completion_tokens', 0),
            total_tokens=self.total_tokens + other_usage.get('total_tokens', 0),
            total_cost=self.total_cost + other_usage.get('cost', 0.0)
        )


@dataclass
class Context:
    """Core context that every EmberData should have."""
    original_input: Any = None
    usage_metrics: UsageMetrics = field(default_factory=UsageMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def accumulate_usage(self, usage: Dict[str, Any]) -> 'Context':
        """Return new context with accumulated usage."""
        return self.__class__(
            original_input=self.original_input,
            usage_metrics=self.usage_metrics.accumulate(usage),
            metadata=self.metadata.copy()
        )


class EmberData:
    """Data embedded with context metadata (like PyTorch tensor).
    
    EmberData carries both the actual data and metadata (usage metrics,
    original input) through the computation graph. This allows
    operators to access context when needed while maintaining clean interfaces.
    
    Examples:
        >>> # Create EmberData (usually done by EmberEmbedding)
        >>> context = Context(original_input="What is 2+2?")
        >>> data = EmberData("What is 2+2?", context)
        >>> 
        >>> # Access data directly
        >>> print(data.data)  # "What is 2+2?"
        >>> 
        >>> # Access context metadata (PyTorch-style)
        >>> print(data.original_input)  # "What is 2+2?"
        >>> print(data.total_usage.total_cost)  # 0.0
    """
    
    def __init__(self, data: Any, context: Optional[Context] = None):
        """Initialize EmberData with data and optional context.
        
        Args:
            data: The actual data (string, dict, etc.)
            context: Optional context metadata
        """
        self.data = data
        self._context = context or Context()
    
    # PyTorch-style direct access to metadata
    @property
    def original_input(self) -> Any:
        """Get the original input that started this computation."""
        return self._context.original_input
    
    @property
    def total_usage(self) -> UsageMetrics:
        """Get accumulated usage metrics."""
        return self._context.usage_metrics
    
    def accumulate_usage(self, usage: Dict[str, Any]) -> 'EmberData':
        """Return new EmberData with accumulated usage."""
        new_context = self._context.accumulate_usage(usage)
        return EmberData(self.data, new_context)
    
    def with_data(self, new_data: Any) -> 'EmberData':
        """Return new EmberData with different data but same context."""
        return EmberData(new_data, self._context)
    
    def __repr__(self) -> str:
        return f"EmberData(data={repr(self.data)}, context={repr(self._context)})"


# Helper functions for working with EmberData
def create_ember_data(data: Any, original_input: Optional[Any] = None) -> EmberData:
    """Create EmberData with optional original input.
    
    Args:
        data: The actual data
        original_input: The original input that started this computation
        
    Returns:
        EmberData instance
    """
    context = Context(original_input=original_input or data)
    return EmberData(data, context)


def extract_usage(ember_data: EmberData) -> UsageMetrics:
    """Extract usage metrics from EmberData.
    
    Args:
        ember_data: EmberData instance
        
    Returns:
        UsageMetrics instance
    """
    return ember_data.total_usage


__all__ = ["EmberData", "Context", "UsageMetrics", "create_ember_data", "extract_usage"]