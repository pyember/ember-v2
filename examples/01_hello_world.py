"""Simplest possible Ember example - make your first LLM call.

This example shows the absolute minimum code needed to use Ember.
You'll learn:
- How to import the models API
- How to make a basic LLM call
- How to access the response

Requirements:
- ember
- Models: Any supported model (gpt-4, gpt-3.5-turbo, claude-3, etc.)

Expected output:
    Response: Paris
    Tokens used: ~15
"""

from ember.api import models


def main():
    # Make a simple request to an LLM
    response = models("gpt-4", "What is the capital of France? Answer in one word.")
    
    # Print the response
    print(f"Response: {response.text}")
    print(f"Tokens used: {response.usage['total_tokens']}")


if __name__ == "__main__":
    main()