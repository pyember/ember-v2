"""Robust Error Handling Patterns - Build reliable AI applications.

Learn essential error handling patterns:
- Graceful degradation
- Retry mechanisms
- Fallback strategies
- Input validation
- Error logging and monitoring
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from _shared.conditional_execution import conditional_llm, SimulatedResponse
from _shared.conditional_llm_template import simulated_models
from ember.api import models
from ember.api.xcs import jit


@conditional_llm()
def example_basic_error_handling(_simulated_mode=False):
    """Basic error handling for LLM calls."""
    print("=" * 50)
    print("Example 1: Basic Error Handling")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    def safe_classify(text: str) -> dict:
        """Classify text with error handling."""
        try:
            # Validate input
            if not text or not isinstance(text, str):
                return {
                    "success": False,
                    "error": "Invalid input: text must be a non-empty string",
                    "result": None,
                }

            # Make LLM call
            prompt = f"Classify sentiment (positive/negative/neutral): {text}"
            response = model_fn("gpt-3.5-turbo", prompt)

            # Parse response
            sentiment = response.text.strip().lower()
            if sentiment not in ["positive", "negative", "neutral"]:
                sentiment = "neutral"  # Default fallback

            return {"success": True, "error": None, "result": sentiment}

        except Exception as e:
            return {
                "success": False,
                "error": f"Classification failed: {str(e)}",
                "result": None,
            }

    # Test with various inputs
    test_cases = [
        "I love this product!",
        "",  # Empty string
        None,  # Invalid type
        "This is okay I guess",
    ]

    print("Testing error handling:")
    for test in test_cases:
        result = safe_classify(test)
        print(f"\nInput: {repr(test)}")
        if result["success"]:
            print(f"✅ Result: {result['result']}")
        else:
            print(f"❌ Error: {result['error']}")


@conditional_llm()
def example_retry_mechanism(_simulated_mode=False):
    """Implement retry logic for transient failures."""
    print("\n" + "=" * 50)
    print("Example 2: Retry Mechanism")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
        """Decorator to retry failed operations."""

        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"  Attempt {attempt + 1} failed: {e}")
                        print(f"  Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"  All {max_retries} attempts failed")

            raise last_error

        return wrapper

    # Simulate a flaky operation
    call_count = 0

    @retry_on_failure
    def flaky_translation(text: str) -> str:
        """Simulates a translation that sometimes fails."""
        nonlocal call_count
        call_count += 1

        # In simulation mode, always succeed on second attempt
        if _simulated_mode:
            if call_count == 1:
                raise ConnectionError("Network timeout")
            return SimulatedResponse("Bonjour le monde", "gpt-3.5-turbo").text

        # Real mode: simulate 50% failure rate
        if call_count % 2 == 1:
            raise ConnectionError("Network timeout")

        # Successful call
        return model_fn("gpt-3.5-turbo", f"Translate to French: {text}").text

    # Test retry mechanism
    print("Testing retry mechanism:")
    try:
        result = flaky_translation("Hello world")
        print(f"✅ Success after {call_count} attempts: {result}")
    except Exception as e:
        print(f"❌ Failed after all retries: {e}")


@conditional_llm()
def example_fallback_strategies(_simulated_mode=False):
    """Implement fallback strategies for robustness."""
    print("\n" + "=" * 50)
    print("Example 3: Fallback Strategies")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    def translate_with_fallback(
        text: str,
        target_lang: str,
        primary_model: str = "gpt-4",
        fallback_model: str = "gpt-3.5-turbo",
    ) -> dict:
        """Translate with model fallback."""
        result = {"translation": None, "model_used": None, "fallback_used": False}

        try:
            # Try primary model
            print(f"Trying primary model: {primary_model}")
            # In simulation mode, always fail primary model to demo fallback
            if _simulated_mode:
                raise Exception("Simulated primary model failure")
            response = model_fn(primary_model, f"Translate to {target_lang}: {text}")
            result["translation"] = response.text.strip()
            result["model_used"] = primary_model

        except Exception as e:
            print(f"Primary model failed: {e}")
            print(f"Falling back to: {fallback_model}")

            try:
                # Fallback to secondary model
                response = model_fn(
                    fallback_model, f"Translate to {target_lang}: {text}"
                )
                result["translation"] = response.text.strip()
                result["model_used"] = fallback_model
                result["fallback_used"] = True

            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                # Last resort: return original text
                result["translation"] = f"[Translation failed: {text}]"
                result["model_used"] = "none"
                result["fallback_used"] = True

        return result

    # Test fallback
    result = translate_with_fallback("Good morning", "Spanish")
    print(f"\nResult: {result['translation']}")
    print(f"Model used: {result['model_used']}")
    print(f"Fallback used: {result['fallback_used']}")


@conditional_llm()
def example_input_validation(_simulated_mode=False):
    """Comprehensive input validation patterns."""
    print("\n" + "=" * 50)
    print("Example 4: Input Validation")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    def validate_and_process(data: Dict[str, Any]) -> dict:
        """Process data with comprehensive validation."""
        errors = []
        warnings = []

        # Validate required fields
        required_fields = ["text", "language"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Validate data types
        if "text" in data:
            if not isinstance(data["text"], str):
                errors.append("Field 'text' must be a string")
            elif len(data["text"].strip()) == 0:
                errors.append("Field 'text' cannot be empty")
            elif len(data["text"]) > 1000:
                warnings.append("Text truncated to 1000 characters")
                data["text"] = data["text"][:1000]

        # Validate language
        if "language" in data:
            supported_languages = ["english", "spanish", "french", "german"]
            if data["language"].lower() not in supported_languages:
                errors.append(f"Unsupported language: {data['language']}")

        # Return early if errors
        if errors:
            return {
                "success": False,
                "errors": errors,
                "warnings": warnings,
                "result": None,
            }

        # Process validated data
        try:
            text = data["text"].strip()
            language = data["language"].lower()

            # Simulate processing
            prompt = f"Analyze this {language} text: {text}"
            result = model_fn("gpt-3.5-turbo", prompt).text

            return {
                "success": True,
                "errors": [],
                "warnings": warnings,
                "result": {
                    "analysis": result,
                    "text_length": len(text),
                    "language": language,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Processing failed: {str(e)}"],
                "warnings": warnings,
                "result": None,
            }

    # Test various inputs
    test_inputs = [
        {"text": "Hello world", "language": "English"},
        {"text": "", "language": "Spanish"},
        {"language": "French"},  # Missing text
        {"text": 123, "language": "German"},  # Wrong type
        {"text": "Bonjour", "language": "Klingon"},  # Unsupported language
    ]

    print("Testing input validation:")
    for i, test_data in enumerate(test_inputs):
        print(f"\nTest {i+1}: {test_data}")
        result = validate_and_process(test_data)

        if result["success"]:
            print("✅ Success")
            print(f"   Result: {result['result']}")
        else:
            print("❌ Failed")
            print(f"   Errors: {result['errors']}")

        if result["warnings"]:
            print(f"⚠️  Warnings: {result['warnings']}")


@conditional_llm()
def example_circuit_breaker(_simulated_mode=False):
    """Implement circuit breaker pattern."""
    print("\n" + "=" * 50)
    print("Example 5: Circuit Breaker Pattern")
    print("=" * 50 + "\n")

    class CircuitBreaker:
        """Simple circuit breaker implementation."""

        def __init__(self, failure_threshold: int = 3, recovery_time: float = 5.0):
            self.failure_threshold = failure_threshold
            self.recovery_time = recovery_time
            self.failure_count = 0
            self.last_failure_time = None
            self.is_open = False

        def call(self, func, *args, **kwargs):
            """Execute function with circuit breaker protection."""
            # Check if circuit is open
            if self.is_open:
                if time.time() - self.last_failure_time > self.recovery_time:
                    print("  Circuit breaker: Attempting recovery...")
                    self.is_open = False
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")

            try:
                result = func(*args, **kwargs)
                self.failure_count = 0  # Reset on success
                return result

            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.is_open = True
                    print(
                        f"  Circuit breaker: OPENED after {self.failure_count} failures"
                    )

                raise e

    # Example usage
    breaker = CircuitBreaker(failure_threshold=2, recovery_time=3.0)

    def unreliable_service(text: str) -> str:
        """Simulates an unreliable service."""
        # Simulate 70% failure rate
        import random

        if random.random() < 0.7:
            raise Exception("Service temporarily unavailable")

        return f"Processed: {text}"

    print("Testing circuit breaker:")
    for i in range(6):
        try:
            result = breaker.call(unreliable_service, f"Request {i+1}")
            print(f"✅ Request {i+1}: {result}")
        except Exception as e:
            print(f"❌ Request {i+1}: {e}")

        time.sleep(1)  # Small delay between requests


@conditional_llm()
def example_error_aggregation(_simulated_mode=False):
    """Aggregate and report errors effectively."""
    print("\n" + "=" * 50)
    print("Example 6: Error Aggregation and Reporting")
    print("=" * 50 + "\n")

    class ErrorCollector:
        """Collects and aggregates errors for reporting."""

        def __init__(self):
            self.errors = []
            self.warnings = []
            self.stats = {"total_operations": 0, "successful": 0, "failed": 0}

        def record_operation(self, success: bool, error: Optional[str] = None):
            """Record operation result."""
            self.stats["total_operations"] += 1

            if success:
                self.stats["successful"] += 1
            else:
                self.stats["failed"] += 1
                if error:
                    self.errors.append({"timestamp": time.time(), "error": error})

        def add_warning(self, warning: str):
            """Add a warning."""
            self.warnings.append({"timestamp": time.time(), "warning": warning})

        def get_report(self) -> dict:
            """Generate error report."""
            success_rate = (
                self.stats["successful"] / self.stats["total_operations"]
                if self.stats["total_operations"] > 0
                else 0
            )

            return {
                "summary": {
                    "total_operations": self.stats["total_operations"],
                    "successful": self.stats["successful"],
                    "failed": self.stats["failed"],
                    "success_rate": f"{success_rate * 100:.1f}%",
                },
                "recent_errors": self.errors[-5:],  # Last 5 errors
                "warnings": self.warnings,
                "health_status": "healthy" if success_rate > 0.8 else "degraded",
            }

    # Example usage
    collector = ErrorCollector()

    # Simulate batch processing with error collection
    @jit
    def process_item(item: str) -> Optional[str]:
        """Process item with error tracking."""
        try:
            # Simulate processing with 20% failure rate
            import random

            if random.random() < 0.2:
                raise Exception(f"Failed to process: {item}")

            result = f"Processed: {item}"
            collector.record_operation(True)
            return result

        except Exception as e:
            collector.record_operation(False, str(e))
            collector.add_warning(f"Item skipped: {item}")
            return None

    # Process batch
    items = [f"Item_{i}" for i in range(10)]
    results = []

    print("Processing items:")
    for item in items:
        result = process_item(item)
        if result:
            print(f"  ✅ {result}")
        else:
            print(f"  ❌ Failed: {item}")
        results.append(result)

    # Generate report
    report = collector.get_report()
    print("\nError Report:")
    print(f"  Summary: {report['summary']}")
    print(f"  Health Status: {report['health_status']}")
    print(f"  Recent Errors: {len(report['recent_errors'])}")
    print(f"  Warnings: {len(report['warnings'])}")


def main():
    """Run all error handling examples."""
    print_section_header("Robust Error Handling Patterns")

    try:
        example_basic_error_handling()
        example_retry_mechanism()
        example_fallback_strategies()
        example_input_validation()
        example_circuit_breaker()
        example_error_aggregation()

        print("\n" + "=" * 50)
        print("✅ Best Practices Summary")
        print("=" * 50)
        print("\n1. Always validate inputs before processing")
        print("2. Use try-catch blocks around external calls")
        print("3. Implement retry logic for transient failures")
        print("4. Have fallback strategies for critical paths")
        print("5. Use circuit breakers to prevent cascading failures")
        print("6. Aggregate and monitor errors for insights")
        print("7. Return structured error responses")
        print("8. Log errors for debugging and monitoring")

    except Exception as e:
        print(f"\n❌ Example error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
