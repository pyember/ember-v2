"""
Example: Hello World
Description: Your first Ember application
Concepts: Basic LLM calls, simple operators
"""

from ember.api import ember
import asyncio


async def main():
    # Simple LLM call
    print("=== Simple LLM Call ===")
    response = await ember.llm("What is the capital of France?")
    print(f"Response: {response}")
    print()

    # With specific model
    print("=== Specific Model ===")
    response = await ember.llm(
        "Explain quantum computing in one sentence", model="gpt-4"
    )
    print(f"GPT-4: {response}")
    print()

    # With parameters
    print("=== With Parameters ===")
    response = await ember.llm(
        "Write a haiku about Python programming", temperature=0.9, max_tokens=50
    )
    print(f"Creative response: {response}")
    print()

    # Creating a simple operator
    print("=== Simple Operator ===")

    @ember.op
    async def translate(text: str, target_language: str = "Spanish") -> str:
        """Translate text to target language."""
        prompt = f"Translate to {target_language}: {text}"
        return await ember.llm(prompt)

    translation = await translate("Hello, world!")
    print(f"Spanish: {translation}")

    translation = await translate("Hello, world!", "French")
    print(f"French: {translation}")
    print()

    # Composing operators
    print("=== Composing Operators ===")

    @ember.op
    async def summarize(text: str) -> str:
        """Summarize text in one sentence."""
        return await ember.llm(f"Summarize in one sentence: {text}")

    # Chain operations
    long_text = """
    Python is a high-level, interpreted programming language known for its 
    simplicity and readability. It supports multiple programming paradigms 
    including procedural, object-oriented, and functional programming. 
    Python's extensive standard library and vibrant ecosystem of third-party 
    packages make it suitable for various applications, from web development 
    to data science and machine learning.
    """

    summary = await summarize(long_text)
    translated_summary = await translate(summary, "Spanish")

    print(f"Original summary: {summary}")
    print(f"Translated summary: {translated_summary}")


if __name__ == "__main__":
    print("Ember Hello World Example")
    print("=" * 50)
    asyncio.run(main())
