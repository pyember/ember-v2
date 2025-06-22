"""Basic Prompt Engineering - Learn effective prompting techniques.

Master the fundamentals of prompt engineering to get better results
from language models through clear instructions, examples, and structure.

Example:
    >>> from ember.api import models
    >>> response = models("gpt-4", "List 3 tips for Python: <your detailed prompt>")
"""

import sys
from pathlib import Path

# Add the shared utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output, ensure_api_key


def example_basic_prompting():
    """Show the difference between vague and specific prompts."""
    print("\n=== Basic vs Specific Prompts ===\n")
    
    from ember.api import models
    
    # Vague prompt
    vague_prompt = "Tell me about Python"
    print("Vague prompt:")
    print(f'  "{vague_prompt}"')
    
    response = models("gpt-3.5-turbo", vague_prompt, max_tokens=100)
    print(f"\nResponse (truncated):")
    print(f"  {response.text.strip()[:200]}...\n")
    
    # Specific prompt
    specific_prompt = """List exactly 3 key advantages of Python for data science.
    For each advantage, provide:
    1. A clear title
    2. A one-sentence explanation
    3. A practical example
    
    Format as a numbered list."""
    
    print("Specific prompt:")
    print(f'  "{specific_prompt[:100]}..."')
    
    response = models("gpt-3.5-turbo", specific_prompt)
    print(f"\nResponse:")
    print(f"  {response.text.strip()}")


def example_few_shot_prompting():
    """Demonstrate few-shot learning with examples."""
    print("\n\n=== Few-Shot Prompting ===\n")
    
    from ember.api import models
    
    few_shot_prompt = """Convert these product descriptions to marketing taglines.

Examples:
Product: Wireless headphones with 30-hour battery life
Tagline: "Freedom that lasts all day, every day"

Product: Smart water bottle that tracks hydration
Tagline: "Stay healthy, effortlessly"

Product: Ergonomic keyboard with silent keys
Tagline: "Type in comfort, work in peace"

Now create a tagline for:
Product: Solar-powered phone charger for camping"""

    response = models("gpt-3.5-turbo", few_shot_prompt)
    print("Few-shot prompt with examples:")
    print("  [Shows 3 examples, then asks for new tagline]\n")
    print("Generated tagline:")
    print(f"  {response.text.strip()}")


def example_structured_output():
    """Show how to get structured outputs."""
    print("\n\n=== Structured Output Prompting ===\n")
    
    from ember.api import models
    
    structured_prompt = """Analyze this text and return a JSON object with the following structure:
{
    "sentiment": "positive/negative/neutral",
    "key_topics": ["topic1", "topic2", ...],
    "summary": "one sentence summary"
}

Text: "The new electric vehicle market is booming. Major automakers are investing billions 
in battery technology and charging infrastructure. However, concerns about range anxiety 
and charging times still persist among consumers."

Return only valid JSON:"""

    response = models("gpt-3.5-turbo", structured_prompt)
    print("Structured output prompt:")
    print("  [Requests specific JSON format]\n")
    print("Response:")
    print(response.text.strip())
    
    # Try to parse it
    try:
        import json
        parsed = json.loads(response.text.strip())
        print("\n✅ Successfully parsed as JSON!")
    except:
        print("\n⚠️  Response wasn't valid JSON - may need prompt refinement")


def example_role_based_prompting():
    """Demonstrate role-based prompting."""
    print("\n\n=== Role-Based Prompting ===\n")
    
    from ember.api import models
    
    # Expert role
    expert_prompt = """You are a senior software architect with 20 years of experience.
    Explain microservices architecture to a junior developer, focusing on:
    - When to use it
    - Key benefits
    - Common pitfalls to avoid
    Keep it practical and concise."""
    
    response = models("gpt-3.5-turbo", expert_prompt)
    print("Expert role prompt:")
    print("  [Assigns role of senior software architect]\n")
    print("Response:")
    print(f"  {response.text.strip()[:300]}...\n")
    
    # Teacher role
    teacher_prompt = """You are a patient elementary school teacher.
    Explain what artificial intelligence is to a 10-year-old student.
    Use simple words and a friendly analogy."""
    
    response = models("gpt-3.5-turbo", teacher_prompt)
    print("\nTeacher role prompt:")
    print("  [Assigns role of elementary school teacher]\n")
    print("Response:")
    print(f"  {response.text.strip()}")


def example_chain_of_thought():
    """Show chain-of-thought prompting."""
    print("\n\n=== Chain-of-Thought Prompting ===\n")
    
    from ember.api import models
    
    # Without chain of thought
    direct_prompt = """If a store has 120 apples and sells 3/4 of them on Monday,
    then sells half of the remaining on Tuesday, how many are left?"""
    
    response = models("gpt-3.5-turbo", direct_prompt)
    print("Direct prompt:")
    print(f"  {direct_prompt}\n")
    print("Response:")
    print(f"  {response.text.strip()}\n")
    
    # With chain of thought
    cot_prompt = """If a store has 120 apples and sells 3/4 of them on Monday,
    then sells half of the remaining on Tuesday, how many are left?
    
    Let's solve this step by step:
    1. First, calculate how many apples were sold on Monday
    2. Then, find how many remained after Monday
    3. Calculate how many were sold on Tuesday
    4. Finally, determine how many are left
    
    Show your work for each step."""
    
    response = models("gpt-3.5-turbo", cot_prompt)
    print("\nChain-of-thought prompt:")
    print("  [Asks for step-by-step reasoning]\n")
    print("Response:")
    print(f"  {response.text.strip()}")


def example_prompt_templates():
    """Show reusable prompt templates."""
    print("\n\n=== Prompt Templates ===\n")
    
    from ember.api import models
    
    # Create a reusable template
    def create_analysis_prompt(topic, aspects, max_points=3):
        return f"""Analyze {topic} considering these aspects:
{chr(10).join(f'- {aspect}' for aspect in aspects)}

Provide exactly {max_points} key insights.
Format each insight as:
• [Insight]: [Brief explanation]"""
    
    # Use the template for different analyses
    topics = [
        ("remote work", ["productivity", "work-life balance", "team collaboration"]),
        ("electric vehicles", ["environmental impact", "cost", "infrastructure"])
    ]
    
    for topic, aspects in topics:
        prompt = create_analysis_prompt(topic, aspects)
        response = models("gpt-3.5-turbo", prompt)
        
        print(f"\nAnalysis of {topic}:")
        print(response.text.strip())


def demo_mode():
    """Show prompt engineering principles without API calls."""
    print("\n=== Demo Mode: Prompt Engineering Principles ===\n")
    
    print("1. Be Specific and Clear:")
    print('   ❌ "Tell me about data"')
    print('   ✅ "List 5 common data structures in Python with examples"\n')
    
    print("2. Provide Structure:")
    print('   ❌ "Explain machine learning"')
    print('   ✅ "Explain machine learning in 3 paragraphs: definition, applications, example"\n')
    
    print("3. Use Examples (Few-Shot):")
    print("   Show 2-3 examples of desired input/output format\n")
    
    print("4. Assign Roles:")
    print('   "You are an expert data scientist..."')
    print('   "Act as a helpful coding tutor..."\n')
    
    print("5. Request Step-by-Step Reasoning:")
    print('   "Think step by step..."')
    print('   "Show your reasoning..."\n')
    
    print("6. Specify Output Format:")
    print('   "Return as JSON..."')
    print('   "Format as a bulleted list..."')
    print('   "Provide exactly 3 examples..."')


def main():
    """Run all prompt engineering examples."""
    print_section_header("Basic Prompt Engineering")
    
    # Check for API key
    if not ensure_api_key("openai"):
        print("\n⚠️  No API key found.")
        print("This example demonstrates prompt engineering techniques.\n")
        demo_mode()
        print("\n📝 To run real examples:")
        print("export OPENAI_API_KEY='your-key-here'")
        return 0
    
    try:
        example_basic_prompting()
        example_few_shot_prompting()
        example_structured_output()
        example_role_based_prompting()
        example_chain_of_thought()
        example_prompt_templates()
        
        print("\n" + "="*50)
        print("✅ Prompt Engineering Best Practices")
        print("="*50)
        print("\n1. Start specific - vague prompts get vague answers")
        print("2. Show examples - few-shot learning improves results")
        print("3. Request structured output - JSON, lists, specific formats")
        print("4. Use role prompts - expertise affects response style")
        print("5. Ask for reasoning - chain-of-thought improves accuracy")
        print("6. Create templates - reuse successful patterns")
        
        print("\n🎯 Next Steps:")
        print("• Experiment with different prompt styles")
        print("• Build a library of effective prompts")
        print("• Test prompts across different models")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())