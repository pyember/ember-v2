"""Error Handling - Building resilient AI applications with Ember's new API.

Difficulty: Intermediate
Time: ~10 minutes

Learning Objectives:
- Handle Ember's simplified exception types
- Implement retry strategies with @jit
- Build fallback chains with multiple models
- Create robust error recovery patterns

Example:
    >>> from ember.api import models
    >>> from ember._internal.exceptions import ModelNotFoundError, ProviderAPIError
    >>> 
    >>> try:
    >>>     response = models("gpt-4", "Hello")
    >>> except ModelNotFoundError:
    >>>     response = models("gpt-3.5-turbo", "Hello")  # Fallback
    >>> except ProviderAPIError as e:
    >>>     print(f"API error: {e}")
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import random

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import models, operators
from ember.api.xcs import jit


def main():
    """Learn error handling patterns with Ember's new API."""
    print_section_header("Error Handling in AI Applications")
    
    # Part 1: New Exception Types
    print("Part 1: Ember's Simplified Exception Types")
    print("=" * 50 + "\n")
    
    print("Ember's focused exception hierarchy:")
    print("  • ModelError (base)")
    print("    ├── ModelNotFoundError - Model doesn't exist")
    print("    ├── ModelProviderError - Missing/invalid API key")
    print("    └── ProviderAPIError - Provider errors (rate limits, etc.)")
    print("\nGone: Complex hierarchies, ambiguous exceptions")
    print("New: Clear, actionable exception types\n")
    
    # Simulate error handling
    def safe_model_call(model_name: str, prompt: str) -> str:
        """Demonstrate basic error handling."""
        try:
            # Simulate different error conditions
            if model_name == "invalid-model":
                raise Exception("ModelNotFoundError: Model 'invalid-model' not found")
            elif model_name == "no-api-key":
                raise Exception("ModelProviderError: Missing API key for provider")
            elif "error" in prompt.lower():
                raise Exception("ProviderAPIError: Rate limit exceeded")
            
            return f"Response from {model_name}: Success!"
            
        except Exception as e:
            error_type = str(e).split(":")[0]
            return f"Handled {error_type}"
    
    # Test different scenarios
    print("Error handling examples:")
    scenarios = [
        ("gpt-4", "Hello world"),
        ("invalid-model", "Hello"),
        ("gpt-4", "Trigger error please"),
        ("no-api-key", "Test")
    ]
    
    for model, prompt in scenarios:
        result = safe_model_call(model, prompt)
        print(f"  {model} → {result}")
    
    # Part 2: Retry Strategies with Functions
    print("\n" + "=" * 50)
    print("Part 2: Retry Strategies (Function-Based)")
    print("=" * 50 + "\n")
    
    def retry_with_backoff(
        func,
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        """Simple retry with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                delay = base_delay * (2 ** attempt)
                print(f"  Attempt {attempt + 1} failed. Retrying in {delay}s...")
                time.sleep(delay)
    
    # Example with simulated failures
    attempts = 0
    
    def flaky_api_call():
        """Simulate a flaky API call."""
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("Temporary failure")
        return "Success after retries!"
    
    print("Testing retry logic:")
    try:
        result = retry_with_backoff(flaky_api_call)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed after all retries: {e}")
    
    # Part 3: Model Fallback Chains
    print("\n" + "=" * 50)
    print("Part 3: Model Fallback Chains")
    print("=" * 50 + "\n")
    
    def create_fallback_chain(models_list: List[str]):
        """Create a function that tries models in order."""
        def call_with_fallback(prompt: str) -> Dict[str, Any]:
            errors = []
            
            for model_name in models_list:
                try:
                    # Simulate model calls
                    if model_name == "gpt-4" and random.random() < 0.5:
                        raise Exception("Rate limited")
                    
                    return {
                        "text": f"Response from {model_name}",
                        "model": model_name,
                        "fallback_count": len(errors)
                    }
                    
                except Exception as e:
                    errors.append((model_name, str(e)))
                    continue
            
            # All models failed
            return {
                "text": "All models failed",
                "model": None,
                "errors": errors
            }
        
        return call_with_fallback
    
    # Create fallback chain
    fallback_models = ["gpt-4", "claude-3", "gpt-3.5-turbo"]
    fallback_fn = create_fallback_chain(fallback_models)
    
    print("Testing fallback chain:")
    for i in range(3):
        result = fallback_fn("Test prompt")
        print(f"  Try {i+1}: {result['model']} (fallbacks: {result.get('fallback_count', 0)})")
    
    # Part 4: JIT-Optimized Error Handling
    print("\n" + "=" * 50)
    print("Part 4: Optimized Error Handling with @jit")
    print("=" * 50 + "\n")
    
    @jit
    def robust_text_processor(text: str) -> dict:
        """Process text with built-in error handling."""
        try:
            # Validate input
            if not text or len(text) > 10000:
                raise ValueError("Invalid text length")
            
            # Process (simulated)
            processed = text.strip().lower()
            word_count = len(processed.split())
            
            return {
                "status": "success",
                "processed": processed,
                "word_count": word_count
            }
            
        except ValueError as e:
            return {
                "status": "error",
                "error": str(e),
                "processed": None
            }
    
    print("JIT-optimized error handling:")
    test_texts = [
        "  Normal text  ",
        "",  # Empty
        "Good text"
    ]
    
    for text in test_texts:
        result = robust_text_processor(text)
        status = "✓" if result["status"] == "success" else "✗"
        print(f"  {status} '{text[:20]}...' → {result['status']}")
    
    # Part 5: Circuit Breaker Pattern
    print("\n" + "=" * 50)
    print("Part 5: Simple Circuit Breaker")
    print("=" * 50 + "\n")
    
    class CircuitBreaker:
        """Simple circuit breaker implementation."""
        
        def __init__(self, failure_threshold: int = 3, timeout: int = 60):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failures = 0
            self.last_failure_time = None
            self.is_open = False
        
        def call(self, func, *args, **kwargs):
            """Execute function with circuit breaker protection."""
            # Check if circuit should be closed
            if self.is_open and self.last_failure_time:
                if time.time() - self.last_failure_time > self.timeout:
                    self.is_open = False
                    self.failures = 0
            
            # If open, fail fast
            if self.is_open:
                raise Exception("Circuit breaker is OPEN")
            
            # Try the call
            try:
                result = func(*args, **kwargs)
                self.failures = 0  # Reset on success
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.is_open = True
                    print(f"  Circuit breaker opened after {self.failures} failures")
                
                raise
    
    # Demo circuit breaker
    breaker = CircuitBreaker(failure_threshold=2)
    call_count = 0
    
    def unreliable_service():
        """Simulate unreliable service."""
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise Exception("Service unavailable")
        return "Service recovered!"
    
    print("Circuit breaker demo:")
    for i in range(5):
        try:
            result = breaker.call(unreliable_service)
            print(f"  Call {i+1}: Success - {result}")
        except Exception as e:
            print(f"  Call {i+1}: Failed - {e}")
    
    # Part 6: Real-World Pattern
    print("\n" + "=" * 50)
    print("Part 6: Real-World Error Handling Pattern")
    print("=" * 50 + "\n")
    
    def create_robust_model_function(
        primary_model: str = "gpt-4",
        fallback_model: str = "gpt-3.5-turbo",
        max_retries: int = 2
    ):
        """Create a robust model calling function."""
        
        @jit
        def robust_call(prompt: str) -> dict:
            """Call model with full error handling."""
            # Input validation
            if not prompt or len(prompt) > 8000:
                return {
                    "success": False,
                    "error": "Invalid prompt length",
                    "text": None
                }
            
            # Try primary model with retries
            for attempt in range(max_retries):
                try:
                    # Simulate model call
                    if random.random() < 0.3:  # 30% failure rate
                        raise Exception("API error")
                    
                    return {
                        "success": True,
                        "text": f"Response from {primary_model}",
                        "model": primary_model,
                        "attempts": attempt + 1
                    }
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                    continue
            
            # Try fallback model
            try:
                return {
                    "success": True,
                    "text": f"Response from {fallback_model} (fallback)",
                    "model": fallback_model,
                    "attempts": max_retries + 1
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "text": None
                }
        
        return robust_call
    
    # Create and test robust function
    robust_model = create_robust_model_function()
    
    print("Testing robust model function:")
    for i in range(5):
        result = robust_model(f"Test prompt {i}")
        if result["success"]:
            print(f"  ✓ {result['model']} (attempts: {result['attempts']})")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Part 7: Best Practices Summary
    print("\n" + "=" * 50)
    print("✅ Error Handling Best Practices")
    print("=" * 50)
    
    print("\n1. Use Ember's focused exceptions:")
    print("   - ModelNotFoundError → Try different model")
    print("   - ProviderAPIError → Retry with backoff")
    print("   - ModelProviderError → Check API keys")
    
    print("\n2. Implement smart retries:")
    print("   - Exponential backoff for rate limits")
    print("   - Immediate retry for network blips")
    print("   - Circuit breaker for persistent failures")
    
    print("\n3. Build fallback chains:")
    print("   - Primary → Secondary → Fallback models")
    print("   - Cached responses as last resort")
    print("   - Graceful degradation")
    
    print("\n4. Optimize with @jit:")
    print("   - Error handling code runs faster")
    print("   - Validation is optimized")
    print("   - Retry logic is efficient")
    
    print("\n5. Monitor and log:")
    print("   - Track error rates")
    print("   - Log with context")
    print("   - Alert on anomalies")
    
    print("\nExample production pattern:")
    print("```python")
    print("from ember.api import models")
    print("from ember._internal.exceptions import ProviderAPIError")
    print("")
    print("@jit")
    print("def safe_generate(prompt: str) -> str:")
    print("    try:")
    print("        return models('gpt-4', prompt).text")
    print("    except ProviderAPIError:")
    print("        # Fallback to cheaper model")
    print("        return models('gpt-3.5-turbo', prompt).text")
    print("```")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())