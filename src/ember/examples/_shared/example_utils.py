"""Common utilities for Ember examples."""

import time
from typing import Any, Dict, Optional
from contextlib import contextmanager


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    width = max(50, len(title) + 4)
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_example_output(label: str, value: Any, indent: int = 2) -> None:
    """Print formatted example output."""
    prefix = " " * indent
    print(f"{prefix}{label}: {value}")


@contextmanager
def timer(name: str):
    """Context manager for timing code execution."""
    start = time.time()
    print(f"Starting: {name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"Completed: {name} (took {elapsed:.2f}s)")


def format_model_response(response: Any) -> str:
    """Format model response for display."""
    if hasattr(response, 'text'):
        return response.text
    elif isinstance(response, dict) and 'text' in response:
        return response['text']
    elif isinstance(response, str):
        return response
    else:
        return str(response)


def ensure_api_key(provider: str) -> bool:
    """Check if API key is configured for provider."""
    import os
    
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
    }
    
    env_var = key_mapping.get(provider.lower())
    if env_var and os.getenv(env_var):
        return True
    
    print(f"Warning: {env_var} not found in environment")
    print(f"Please set your {provider} API key to run this example")
    return False


def example_metadata(
    title: str,
    difficulty: str,
    time_estimate: str,
    prerequisites: Optional[list] = None,
    learning_objectives: Optional[list] = None,
    concepts: Optional[list] = None
) -> Dict[str, Any]:
    """Generate standardized example metadata."""
    return {
        "title": title,
        "difficulty": difficulty,
        "time_estimate": time_estimate,
        "prerequisites": prerequisites or [],
        "learning_objectives": learning_objectives or [],
        "concepts": concepts or []
    }


def print_example_metadata(metadata: Dict[str, Any]) -> None:
    """Print example metadata in a formatted way."""
    print_section_header(metadata["title"])
    print(f"Difficulty: {metadata['difficulty']}")
    print(f"Time: {metadata['time_estimate']}")
    
    if metadata["prerequisites"]:
        print("\nPrerequisites:")
        for prereq in metadata["prerequisites"]:
            print(f"  - {prereq}")
    
    if metadata["learning_objectives"]:
        print("\nLearning Objectives:")
        for obj in metadata["learning_objectives"]:
            print(f"  - {obj}")
    
    if metadata["concepts"]:
        print("\nKey Concepts:")
        for concept in metadata["concepts"]:
            print(f"  - {concept}")
    
    print()  # Empty line