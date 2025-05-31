"""Handle errors gracefully in LLM applications.

This example shows how to handle common failure modes.
You'll learn:
- How to catch and handle API errors
- How to implement retry logic
- How to provide fallback behavior

Requirements:
- ember
- Models: Any supported model

Expected output:
    Testing normal operation...
    Success: <response>
    
    Testing error handling...
    Attempt 1 failed: <error>
    Attempt 2 failed: <error>
    Attempt 3 succeeded: <response>
    
    Testing fallback...
    Primary failed, using fallback
    Result: <fallback response>
"""

import time
from typing import Optional
from ember.api import models
from ember.core.exceptions import ModelError, ProviderAPIError


def safe_query(prompt: str, max_retries: int = 3) -> Optional[str]:
    """Query with automatic retry on failure."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = models("gpt-4", prompt)
            return response.text
        except ProviderAPIError as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e.message}")
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                time.sleep(wait_time)
    
    print(f"All attempts failed. Last error: {last_error}")
    return None


def query_with_fallback(prompt: str, primary_model: str = "gpt-4", 
                       fallback_model: str = "gpt-3.5-turbo") -> str:
    """Try primary model, fall back to secondary on failure."""
    try:
        response = models(primary_model, prompt)
        return response.text
    except (ModelError, ProviderAPIError) as e:
        print(f"Primary model failed: {e.message}, using fallback")
        response = models(fallback_model, prompt)
        return response.text


class RobustQueryHandler:
    """Handle queries with circuit breaker pattern."""
    
    def __init__(self, failure_threshold: int = 3):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.circuit_open = False
        self.last_failure_time = None
    
    def query(self, prompt: str) -> Optional[str]:
        """Query with circuit breaker protection."""
        if self.circuit_open:
            # Check if we should try again
            if time.time() - self.last_failure_time > 60:  # 1 minute cooldown
                self.circuit_open = False
                self.failure_count = 0
            else:
                return "Circuit breaker open - service temporarily unavailable"
        
        try:
            response = models("gpt-4", prompt)
            self.failure_count = 0  # Reset on success
            return response.text
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.circuit_open = True
                return "Circuit breaker opened after repeated failures"
            
            raise


def main():
    # Test 1: Normal operation
    print("Testing normal operation...")
    try:
        response = models("gpt-4", "Say hello")
        print(f"Success: {response.text}\n")
    except Exception as e:
        print(f"Unexpected error: {e}\n")
    
    # Test 2: Retry logic (simulated with a working query)
    print("Testing retry logic...")
    result = safe_query("What is 2+2?")
    if result:
        print(f"Final result: {result}\n")
    
    # Test 3: Fallback behavior
    print("Testing fallback...")
    result = query_with_fallback("Explain quantum computing in one sentence")
    print(f"Result: {result}\n")
    
    # Test 4: Circuit breaker (demonstration)
    print("Circuit breaker example:")
    handler = RobustQueryHandler(failure_threshold=3)
    
    # Simulate some queries
    for i in range(5):
        try:
            result = handler.query("Test query")
            print(f"Query {i+1}: Success")
        except Exception as e:
            print(f"Query {i+1}: Failed - {e}")


if __name__ == "__main__":
    main()


# Best practices:
# 1. Always wrap LLM calls in try-except
# 2. Implement exponential backoff for retries
# 3. Have fallback models for critical paths
# 4. Log errors for debugging
# 5. Set reasonable timeouts

# Next steps:
# - See examples/patterns/retry_with_feedback.py
# - Learn about graceful degradation strategies
# - Implement caching to reduce API calls