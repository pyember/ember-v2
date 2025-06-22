"""Type Safety - Building robust AI applications with type hints.

Learn how to use Python's type system to catch errors early and
make your AI applications more maintainable and reliable.

Example:
    >>> from typing import List, Dict
    >>> def process_responses(responses: List[str]) -> Dict[str, int]:
    ...     return {"count": len(responses)}
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def example_basic_type_hints():
    """Show basic type hints in action."""
    print("\n=== Basic Type Hints ===\n")
    
    # Without type hints - unclear what's expected
    def process_data_unclear(data):
        return len(data)
    
    # With type hints - clear expectations
    def process_data_typed(data: List[str]) -> int:
        """Process a list of strings and return count."""
        return len(data)
    
    print("Without type hints:")
    print("  def process_data(data):")
    print("  → Unclear what 'data' should be\n")
    
    print("With type hints:")
    print("  def process_data(data: List[str]) -> int:")
    print("  → Clear: expects list of strings, returns int\n")
    
    # Using the typed function
    messages = ["Hello", "World", "From", "Ember"]
    count = process_data_typed(messages)
    print(f"Processing {messages}")
    print(f"Result: {count} messages")


def example_complex_types():
    """Demonstrate more complex type annotations."""
    print("\n\n=== Complex Type Annotations ===\n")
    
    # Function with multiple parameter types
    def analyze_responses(
        responses: List[str],
        weights: Optional[List[float]] = None,
        threshold: float = 0.5
    ) -> Dict[str, Union[float, int, List[str]]]:
        """Analyze responses with optional weights."""
        result: Dict[str, Union[float, int, List[str]]] = {
            "count": len(responses),
            "threshold": threshold,
            "items": responses[:3]  # First 3 items
        }
        
        if weights:
            result["avg_weight"] = sum(weights) / len(weights)
        
        return result
    
    print("Complex function signature:")
    print("  analyze_responses(")
    print("      responses: List[str],")
    print("      weights: Optional[List[float]] = None,")
    print("      threshold: float = 0.5")
    print("  ) -> Dict[str, Union[float, int, List[str]]]")
    
    # Use the function
    responses = ["Good", "Excellent", "Fair", "Poor"]
    weights = [0.9, 1.0, 0.6, 0.3]
    
    result = analyze_responses(responses, weights)
    print(f"\nResult: {result}")


def example_dataclasses():
    """Show type safety with dataclasses."""
    print("\n\n=== Type Safety with Dataclasses ===\n")
    
    @dataclass
    class ModelResponse:
        """Structured response from a language model."""
        text: str
        confidence: float
        tokens_used: int
        metadata: Optional[Dict[str, Any]] = None
        
        def is_high_confidence(self) -> bool:
            """Check if response has high confidence."""
            return self.confidence > 0.8
    
    @dataclass
    class AnalysisResult:
        """Result of analyzing multiple responses."""
        responses: List[ModelResponse]
        summary: str
        avg_confidence: float
        
        @classmethod
        def from_responses(cls, responses: List[ModelResponse]) -> 'AnalysisResult':
            """Create analysis from responses."""
            avg_conf = sum(r.confidence for r in responses) / len(responses)
            summary = f"Analyzed {len(responses)} responses"
            return cls(responses, summary, avg_conf)
    
    print("Using dataclasses for type safety:")
    print("  @dataclass")
    print("  class ModelResponse:")
    print("      text: str")
    print("      confidence: float")
    print("      tokens_used: int\n")
    
    # Create typed instances
    responses = [
        ModelResponse("Positive sentiment", 0.95, 42),
        ModelResponse("Neutral sentiment", 0.7, 38),
        ModelResponse("Negative sentiment", 0.88, 40)
    ]
    
    analysis = AnalysisResult.from_responses(responses)
    print(f"Analysis: {analysis.summary}")
    print(f"Average confidence: {analysis.avg_confidence:.2f}")
    
    # Type checking helps catch errors
    high_conf = [r for r in responses if r.is_high_confidence()]
    print(f"High confidence responses: {len(high_conf)}")


def example_function_protocols():
    """Demonstrate function type protocols."""
    print("\n\n=== Function Type Protocols ===\n")
    
    from typing import Protocol, Callable
    
    # Define a protocol for processors
    class TextProcessor(Protocol):
        """Protocol for text processing functions."""
        def __call__(self, text: str) -> str: ...
    
    # Functions that match the protocol
    def uppercase_processor(text: str) -> str:
        return text.upper()
    
    def reverse_processor(text: str) -> str:
        return text[::-1]
    
    def clean_processor(text: str) -> str:
        return ' '.join(text.split())
    
    # Function that uses the protocol
    def apply_processors(
        text: str,
        processors: List[TextProcessor]
    ) -> List[str]:
        """Apply multiple processors to text."""
        return [proc(text) for proc in processors]
    
    print("Using function protocols:")
    print("  class TextProcessor(Protocol):")
    print("      def __call__(self, text: str) -> str: ...\n")
    
    # Use the typed functions
    text = "  Hello   World  "
    processors = [clean_processor, uppercase_processor, reverse_processor]
    results = apply_processors(text, processors)
    
    print(f"Original: '{text}'")
    for proc, result in zip(processors, results):
        print(f"  {proc.__name__}: '{result}'")


def example_generic_types():
    """Show generic type usage."""
    print("\n\n=== Generic Types ===\n")
    
    from typing import TypeVar, Generic, List
    
    T = TypeVar('T')
    
    @dataclass
    class BatchResult(Generic[T]):
        """Generic batch processing result."""
        items: List[T]
        successful: List[T]
        failed: List[Tuple[int, str]]  # (index, error)
        
        @property
        def success_rate(self) -> float:
            """Calculate success rate."""
            total = len(self.items)
            return len(self.successful) / total if total > 0 else 0.0
    
    # Processor that returns specific type
    def process_batch_typed(items: List[str]) -> BatchResult[str]:
        """Process a batch with type safety."""
        successful = []
        failed = []
        
        for i, item in enumerate(items):
            if item.strip():  # Non-empty
                successful.append(item.upper())
            else:
                failed.append((i, "Empty string"))
        
        return BatchResult(items, successful, failed)
    
    print("Generic batch result:")
    print("  class BatchResult(Generic[T]):")
    print("      items: List[T]")
    print("      successful: List[T]")
    print("      failed: List[Tuple[int, str]]\n")
    
    # Use the generic type
    items = ["hello", "world", "", "ember", ""]
    result = process_batch_typed(items)
    
    print(f"Processed {len(items)} items")
    print(f"Success rate: {result.success_rate:.1%}")
    print(f"Successful: {result.successful}")
    print(f"Failed indices: {[idx for idx, _ in result.failed]}")


def example_runtime_validation():
    """Show runtime type validation."""
    print("\n\n=== Runtime Type Validation ===\n")
    
    def validate_input(
        data: Any,
        expected_type: type,
        field_name: str = "input"
    ) -> None:
        """Validate input matches expected type."""
        if not isinstance(data, expected_type):
            raise TypeError(
                f"{field_name} must be {expected_type.__name__}, "
                f"got {type(data).__name__}"
            )
    
    def safe_process(data: List[str]) -> Dict[str, Any]:
        """Process data with runtime validation."""
        # Validate input
        validate_input(data, list, "data")
        
        # Validate list contents
        for i, item in enumerate(data):
            validate_input(item, str, f"data[{i}]")
        
        # Now safe to process
        return {
            "count": len(data),
            "first": data[0] if data else None,
            "types_valid": True
        }
    
    print("Runtime validation example:")
    
    # Valid input
    try:
        result = safe_process(["a", "b", "c"])
        print(f"✅ Valid input processed: {result}")
    except TypeError as e:
        print(f"❌ Error: {e}")
    
    # Invalid input
    try:
        result = safe_process(["a", 2, "c"])  # Mixed types
        print(f"✅ Processed: {result}")
    except TypeError as e:
        print(f"❌ Type error caught: {e}")


def main():
    """Run all type safety examples."""
    print_section_header("Type Safety in AI Applications")
    
    print("🎯 Why Type Safety Matters:\n")
    print("• Catch errors before runtime")
    print("• Better IDE support and autocomplete")
    print("• Self-documenting code")
    print("• Easier refactoring")
    print("• Clearer interfaces between components")
    
    example_basic_type_hints()
    example_complex_types()
    example_dataclasses()
    example_function_protocols()
    example_generic_types()
    example_runtime_validation()
    
    print("\n" + "="*50)
    print("✅ Type Safety Best Practices")
    print("="*50)
    print("\n1. Start simple - add types gradually")
    print("2. Use dataclasses for structured data")
    print("3. Leverage Optional for nullable values")
    print("4. Create type aliases for complex types")
    print("5. Use protocols for duck typing")
    print("6. Add runtime validation for external data")
    print("7. Run mypy for static type checking")
    
    print("\n🔧 Tools for Type Safety:")
    print("• mypy - Static type checker")
    print("• pydantic - Runtime validation")
    print("• dataclasses - Structured data")
    print("• typing - Type hints library")
    
    print("\nNext: Learn about context management in 'context_management.py'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())