"""Shared utilities for Ember examples."""

import time
import sys
import random
from typing import Any, Dict, Optional, List
from contextlib import contextmanager


def print_section_header(title: str) -> None:
    """Print formatted section header."""
    width = max(50, len(title) + 4)
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_example_output(label: str, value: Any, indent: int = 2) -> None:
    """Print labeled output with indentation."""
    prefix = " " * indent
    print(f"{prefix}{label}: {value}")


@contextmanager
def timer(name: str):
    """Time code execution."""
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


def simulate_model_response(prompt: str, style: str = "default") -> str:
    """Simulate model responses for examples that run without API keys."""
    responses = {
        "classification": [
            "positive", "negative", "neutral", "mixed"
        ],
        "summary": [
            "This is a concise summary of the main points.",
            "The text discusses key concepts in AI development.",
            "Main ideas center around efficient system design."
        ],
        "analysis": [
            "The analysis reveals interesting patterns in the data.",
            "Key insights suggest improvements in system architecture.",
            "Results indicate strong performance across metrics."
        ],
        "creative": [
            "Once upon a time, in a world of endless possibilities...",
            "Imagine a future where AI and humans collaborate seamlessly...",
            "The story begins with a revolutionary discovery..."
        ],
        "technical": [
            "The implementation follows standard design patterns.",
            "Performance optimization yields 2.5x improvement.",
            "Code structure maintains clean separation of concerns."
        ],
        "default": [
            "This is a simulated model response for demonstration.",
            "Ember provides powerful tools for AI development.",
            "The framework simplifies complex AI workflows."
        ]
    }
    
    style_responses = responses.get(style, responses["default"])
    return random.choice(style_responses)


def safe_model_call(model_fn, prompt: str, fallback_style: str = "default") -> str:
    """Safely call model with fallback to simulation."""
    try:
        response = model_fn(prompt)
        return format_model_response(response)
    except Exception as e:
        print(f"Note: Using simulated response (API call failed: {e})")
        return simulate_model_response(prompt, fallback_style)


def print_comparison_table(data: List[Dict[str, Any]], title: str = "Comparison") -> None:
    """Print a formatted comparison table."""
    if not data:
        return
    
    print(f"\n{title}:")
    print("-" * 60)
    
    # Get all unique keys
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    headers = list(all_keys)
    
    # Print headers
    header_line = " | ".join(f"{h:15}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    
    # Print data rows
    for item in data:
        row_values = [str(item.get(h, "N/A"))[:15] for h in headers]
        print(" | ".join(f"{v:15}" for v in row_values))
    print()


def measure_performance(func, *args, runs: int = 3, **kwargs) -> Dict[str, float]:
    """Measure function performance across multiple runs."""
    times = []
    
    for _ in range(runs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        times.append(end - start)
    
    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_time": sum(times),
        "runs": runs
    }


def print_performance_summary(perf_data: Dict[str, float], operation: str = "Operation") -> None:
    """Print formatted performance summary."""
    print(f"\n{operation} Performance:")
    print(f"  Average: {perf_data['avg_time']:.4f}s")
    print(f"  Range: {perf_data['min_time']:.4f}s - {perf_data['max_time']:.4f}s")
    print(f"  Total: {perf_data['total_time']:.4f}s ({perf_data['runs']} runs)")


def generate_sample_texts(count: int = 5, style: str = "varied") -> List[str]:
    """Generate sample texts for examples."""
    templates = {
        "short": [
            "Hello world!",
            "AI is amazing.",
            "Quick test text.",
            "Simple example.",
            "Fast processing."
        ],
        "medium": [
            "This is a medium-length text for testing purposes.",
            "Ember makes AI development simple and efficient.",
            "The framework provides powerful tools for building AI systems.",
            "Modern AI requires robust and scalable architectures.",
            "Performance optimization is key to production success."
        ],
        "long": [
            "This is a longer text sample that contains multiple sentences and more detailed content. It's designed to test how systems handle more complex inputs with various linguistic patterns and structures.",
            "Artificial intelligence has revolutionized many aspects of modern technology. From natural language processing to computer vision, AI systems continue to push the boundaries of what's possible in automated reasoning and decision-making.",
            "Building production-ready AI systems requires careful consideration of scalability, reliability, and performance. The Ember framework addresses these challenges by providing a simplified yet powerful API for AI development."
        ],
        "varied": [
            "Short one.",
            "This is a medium-length text with some complexity and detail.",
            "This is a comprehensive long-form text that includes multiple sentences, various linguistic patterns, and demonstrates how systems handle complex inputs with different structural characteristics and semantic content."
        ]
    }
    
    if style not in templates:
        style = "varied"
    
    texts = templates[style]
    if count <= len(texts):
        return texts[:count]
    else:
        # Repeat and cycle through if we need more
        result = []
        for i in range(count):
            result.append(texts[i % len(texts)])
        return result


def check_example_dependencies() -> Dict[str, bool]:
    """Check if example dependencies are available."""
    dependencies = {}
    
    # Check for Ember imports
    try:
        import ember
        dependencies['ember'] = True
    except ImportError:
        dependencies['ember'] = False
    
    # Check for API keys
    import os
    dependencies['openai_key'] = bool(os.getenv('OPENAI_API_KEY'))
    dependencies['anthropic_key'] = bool(os.getenv('ANTHROPIC_API_KEY'))
    
    return dependencies


def print_dependency_status() -> None:
    """Print status of example dependencies."""
    deps = check_example_dependencies()
    
    print("📋 Dependency Status:")
    for name, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {name}")
    
    if not all(deps.values()):
        print("\n💡 Note: Examples will use simulated responses for missing dependencies")
    print()


def safe_exit(code: int = 0, message: Optional[str] = None) -> None:
    """Safely exit example with optional message."""
    if message:
        print(f"\n{message}")
    
    if code == 0:
        print("✅ Example completed successfully!")
    else:
        print("❌ Example encountered issues.")
    
    sys.exit(code)