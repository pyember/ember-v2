"""Error Handling - Building resilient AI applications.

Learn how to handle errors gracefully in AI applications, including
API failures, rate limits, timeouts, and invalid responses.

Example:
    >>> from ember.core.exceptions import ModelError, RateLimitError
    >>> try:
    ...     response = model.generate(prompt)
    ... except RateLimitError:
    ...     # Handle rate limiting
"""

import sys
from pathlib import Path
from typing import Optional, Any, List
import time

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def example_basic_error_handling():
    """Show basic error handling patterns."""
    print("\n=== Basic Error Handling ===\n")
    
    print("Common AI application errors:")
    print("  • API errors (connection, authentication)")
    print("  • Rate limiting")
    print("  • Token limits exceeded")
    print("  • Invalid responses")
    print("  • Timeout errors\n")
    
    # Basic try-except pattern
    print("Basic error handling:")
    print("""
    try:
        response = model.generate(prompt)
        result = response.text
    except Exception as e:
        print(f"Error: {e}")
        result = "Failed to generate response"
    """)
    
    # Simulated error handling
    def safe_api_call(prompt: str) -> str:
        """Simulate safe API call with error handling."""
        try:
            # Simulate API call
            if "error" in prompt.lower():
                raise Exception("API Error: Invalid prompt")
            return f"Response to: {prompt}"
        except Exception as e:
            return f"Error handled: {str(e)}"
    
    result = safe_api_call("Tell me about Python")
    print(f"\nSuccess case: {result}")
    
    result = safe_api_call("Trigger error please")
    print(f"Error case: {result}")


def example_specific_exceptions():
    """Demonstrate handling specific exception types."""
    print("\n\n=== Specific Exception Handling ===\n")
    
    # Define custom exceptions
    class ModelError(Exception):
        """Base exception for model errors."""
        pass
    
    class RateLimitError(ModelError):
        """Rate limit exceeded."""
        def __init__(self, retry_after: int):
            self.retry_after = retry_after
            super().__init__(f"Rate limited. Retry after {retry_after}s")
    
    class TokenLimitError(ModelError):
        """Token limit exceeded."""
        def __init__(self, used: int, limit: int):
            self.used = used
            self.limit = limit
            super().__init__(f"Token limit exceeded: {used}/{limit}")
    
    print("Ember exception hierarchy:")
    print("  ModelError")
    print("  ├── RateLimitError")
    print("  ├── TokenLimitError")
    print("  ├── AuthenticationError")
    print("  └── InvalidResponseError\n")
    
    print("Handling specific exceptions:")
    print("""
    try:
        response = model.generate(prompt)
    except RateLimitError as e:
        print(f"Rate limited. Waiting {e.retry_after}s...")
        time.sleep(e.retry_after)
        response = model.generate(prompt)  # Retry
    except TokenLimitError as e:
        print(f"Too many tokens: {e.used}/{e.limit}")
        prompt = truncate_prompt(prompt)
        response = model.generate(prompt)
    except ModelError as e:
        print(f"Model error: {e}")
        response = fallback_response()
    """)


def example_retry_strategies():
    """Show retry strategies for transient errors."""
    print("\n\n=== Retry Strategies ===\n")
    
    def exponential_backoff_retry(
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """Retry with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
    
    print("1. Exponential Backoff:")
    print("   Delays: 1s → 2s → 4s → 8s → ...")
    print("   Good for: Rate limits, temporary failures\n")
    
    print("2. Linear Backoff:")
    print("   Delays: 1s → 2s → 3s → 4s → ...")
    print("   Good for: Predictable recovery times\n")
    
    print("3. Immediate Retry:")
    print("   Delays: 0s → 0s → 0s")
    print("   Good for: Network blips\n")
    
    # Example usage
    print("Example retry implementation:")
    attempts = 0
    def flaky_operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception(f"Transient error on attempt {attempts}")
        return "Success!"
    
    try:
        result = exponential_backoff_retry(flaky_operation)
        print(f"\n{result} after {attempts} attempts")
    except Exception as e:
        print(f"\nFailed after all retries: {e}")


def example_fallback_patterns():
    """Demonstrate fallback strategies."""
    print("\n\n=== Fallback Patterns ===\n")
    
    print("Fallback strategies for AI applications:\n")
    
    print("1. Model Fallback Chain:")
    print("""
    models = ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-instruct']
    
    for model_name in models:
        try:
            response = use_model(model_name, prompt)
            break
        except ModelError:
            continue
    else:
        response = "All models failed"
    """)
    
    print("\n2. Cached Response Fallback:")
    print("""
    try:
        response = model.generate(prompt)
        cache.store(prompt, response)
    except ModelError:
        response = cache.get(prompt)
        if not response:
            response = default_response
    """)
    
    print("\n3. Degraded Functionality:")
    print("""
    try:
        # Full analysis with GPT-4
        analysis = advanced_analysis(text)
    except (ModelError, TimeoutError):
        # Fallback to simpler analysis
        analysis = basic_analysis(text)
    """)


def example_circuit_breaker():
    """Show circuit breaker pattern."""
    print("\n\n=== Circuit Breaker Pattern ===\n")
    
    class CircuitBreaker:
        def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        def call(self, func, *args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                raise
    
    print("Circuit Breaker States:")
    print("  • CLOSED: Normal operation")
    print("  • OPEN: Failing fast, rejecting calls")
    print("  • HALF_OPEN: Testing if service recovered\n")
    
    print("Benefits:")
    print("  • Prevents cascading failures")
    print("  • Gives failing services time to recover")
    print("  • Fails fast when service is down")
    print("  • Automatic recovery detection")


def example_error_context():
    """Show error context and debugging."""
    print("\n\n=== Error Context and Debugging ===\n")
    
    class DetailedError(Exception):
        """Error with additional context."""
        def __init__(self, message: str, **context):
            super().__init__(message)
            self.context = context
        
        def __str__(self):
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{super().__str__()} [{ctx_str}]"
    
    print("Rich error information:")
    print("""
    try:
        response = model.generate(prompt)
    except ModelError as e:
        raise DetailedError(
            "Model generation failed",
            model=model_name,
            prompt_length=len(prompt),
            timestamp=datetime.now(),
            attempt=retry_count,
            original_error=str(e)
        )
    """)
    
    # Example
    try:
        raise DetailedError(
            "API call failed",
            endpoint="https://api.example.com/v1/generate",
            status_code=429,
            retry_after=60
        )
    except DetailedError as e:
        print(f"\nExample error: {e}")


def example_validation_errors():
    """Handle validation and input errors."""
    print("\n\n=== Validation and Input Errors ===\n")
    
    def validate_prompt(prompt: str) -> None:
        """Validate prompt before sending to API."""
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        if len(prompt) > 10000:
            raise ValueError(f"Prompt too long: {len(prompt)} chars (max: 10000)")
        
        forbidden_patterns = ["<script>", "javascript:", "onclick="]
        for pattern in forbidden_patterns:
            if pattern in prompt.lower():
                raise ValueError(f"Forbidden pattern detected: {pattern}")
    
    print("Input validation patterns:")
    print("""
    def safe_generate(prompt: str) -> str:
        # Validate input
        validate_prompt(prompt)
        
        # Sanitize if needed
        prompt = sanitize_input(prompt)
        
        # Validate output
        response = model.generate(prompt)
        validate_response(response)
        
        return response.text
    """)
    
    print("\nCommon validations:")
    print("  • Length limits")
    print("  • Character encoding")
    print("  • Injection patterns")
    print("  • Format requirements")
    print("  • Business rules")


def example_async_error_handling():
    """Show error handling in async contexts."""
    print("\n\n=== Async Error Handling ===\n")
    
    print("Handling errors in async operations:")
    print("""
    import asyncio
    
    async def async_generate(prompt: str):
        try:
            response = await model.agenerate(prompt)
            return response
        except asyncio.TimeoutError:
            return "Request timed out"
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise
    
    # Batch processing with error handling
    async def process_batch(prompts: List[str]):
        tasks = [async_generate(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = []
        failed = []
        
        for prompt, result in zip(prompts, results):
            if isinstance(result, Exception):
                failed.append((prompt, result))
            else:
                successful.append((prompt, result))
        
        return successful, failed
    """)


def main():
    """Run all error handling examples."""
    print_section_header("Error Handling in AI Applications")
    
    print("🎯 Why Error Handling Matters:\n")
    print("• AI APIs can fail unexpectedly")
    print("• Rate limits are common")
    print("• Network issues happen")
    print("• Invalid responses occur")
    print("• Costs can escalate with retries")
    
    example_basic_error_handling()
    example_specific_exceptions()
    example_retry_strategies()
    example_fallback_patterns()
    example_circuit_breaker()
    example_error_context()
    example_validation_errors()
    example_async_error_handling()
    
    print("\n" + "="*50)
    print("✅ Error Handling Best Practices")
    print("="*50)
    print("\n1. Always handle specific exceptions first")
    print("2. Implement retry logic with backoff")
    print("3. Set reasonable timeout values")
    print("4. Log errors with context")
    print("5. Fail gracefully with fallbacks")
    print("6. Validate inputs before API calls")
    print("7. Monitor error rates and patterns")
    print("8. Use circuit breakers for critical paths")
    
    print("\n🔧 Error Handling Checklist:")
    print("□ Identify all failure modes")
    print("□ Define retry strategies")
    print("□ Implement fallback options")
    print("□ Add comprehensive logging")
    print("□ Set up monitoring/alerting")
    print("□ Test error scenarios")
    print("□ Document error behaviors")
    
    print("\nNext: Explore more examples in other directories!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())