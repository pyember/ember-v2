"""
Example: Operators Basics - Understanding Ember's Core Abstraction
Difficulty: Basic
Time: ~5 minutes
Prerequisites: 01_getting_started/hello_world.py

Learning Objectives:
- Understand what operators are and why they matter
- Create your first real operator
- Learn about operator composition

Key Concepts:
- Operators as composable units
- Input/output contracts
- Operator chaining
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import models, operators


def main():
    """Learn the fundamentals of Ember operators."""
    print_section_header("Understanding Operators")
    
    # Part 1: Why Operators?
    print("🎯 Why Operators?\n")
    print("Operators are Ember's core abstraction for composable AI systems.")
    print("They provide:")
    print("  • Type safety and validation")
    print("  • Automatic parallelization")
    print("  • Easy composition and reuse")
    print("  • Clear interfaces between components\n")
    
    # Part 2: Simple Operator
    print("="*50)
    print("Part 1: A Simple Text Processing Operator")
    print("="*50 + "\n")
    
    class TextCleanerOperator(operators.Operator):
        """Cleans and normalizes text input."""
        
        specification = operators.Specification()
        
        def forward(self, *, inputs):
            # Get text from inputs
            text = inputs.get("text", "")
            
            # Clean the text
            cleaned = text.strip().lower()
            cleaned = " ".join(cleaned.split())  # Normalize whitespace
            
            return {
                "original": text,
                "cleaned": cleaned,
                "length": len(cleaned)
            }
    
    # Use the operator
    cleaner = TextCleanerOperator()
    result = cleaner(text="  Hello   WORLD!  ")
    
    print("Text Cleaner Results:")
    print_example_output("Original", repr(result["original"]))
    print_example_output("Cleaned", result["cleaned"])
    print_example_output("Length", result["length"])
    
    # Part 3: Operator with Configuration
    print("\n" + "="*50)
    print("Part 2: Configurable Operator")
    print("="*50 + "\n")
    
    class WordCounterOperator(operators.Operator):
        """Counts words with configurable options."""
        
        specification = operators.Specification()
        
        def __init__(self, *, min_length: int = 1):
            self.min_length = min_length
        
        def forward(self, *, inputs):
            text = inputs.get("text", "")
            words = text.split()
            
            # Filter by minimum length
            filtered_words = [w for w in words if len(w) >= self.min_length]
            
            return {
                "total_words": len(words),
                "filtered_words": len(filtered_words),
                "words": filtered_words[:5]  # First 5 as sample
            }
    
    # Create with configuration
    counter = WordCounterOperator(min_length=3)
    result = counter(text="I am learning to use Ember operators")
    
    print("Word Counter Results:")
    print_example_output("Total words", result["total_words"])
    print_example_output("Words >= 3 chars", result["filtered_words"])
    print_example_output("Sample words", result["words"])
    
    # Part 4: Operator Composition
    print("\n" + "="*50)
    print("Part 3: Composing Operators")
    print("="*50 + "\n")
    
    class TextPipelineOperator(operators.Operator):
        """Combines multiple operators into a pipeline."""
        
        specification = operators.Specification()
        
        def __init__(self):
            self.cleaner = TextCleanerOperator()
            self.counter = WordCounterOperator(min_length=4)
        
        def forward(self, *, inputs):
            # Step 1: Clean the text
            cleaned_result = self.cleaner(text=inputs.get("text", ""))
            
            # Step 2: Count words in cleaned text
            count_result = self.counter(text=cleaned_result["cleaned"])
            
            # Combine results
            return {
                "cleaned_text": cleaned_result["cleaned"],
                "stats": {
                    "original_length": cleaned_result["length"],
                    "total_words": count_result["total_words"],
                    "significant_words": count_result["filtered_words"]
                }
            }
    
    # Use the pipeline
    pipeline = TextPipelineOperator()
    result = pipeline(text="  The QUICK brown fox jumps!  ")
    
    print("Pipeline Results:")
    print_example_output("Cleaned", result["cleaned_text"])
    print_example_output("Stats", result["stats"])
    
    # Part 5: Real-World Pattern
    print("\n" + "="*50)
    print("Part 4: Practical Example - Question Analyzer")
    print("="*50 + "\n")
    
    class QuestionAnalyzer(operators.Operator):
        """Analyzes questions to determine their type and complexity."""
        
        specification = operators.Specification()
        
        def forward(self, *, inputs):
            question = inputs.get("question", "").strip()
            
            # Simple analysis (in practice, could use LLM)
            question_lower = question.lower()
            
            # Determine type
            if question_lower.startswith(("what", "which")):
                q_type = "factual"
            elif question_lower.startswith(("why", "how")):
                q_type = "explanatory"
            elif question_lower.startswith(("is", "are", "do", "does")):
                q_type = "yes/no"
            else:
                q_type = "other"
            
            # Estimate complexity
            word_count = len(question.split())
            complexity = "simple" if word_count < 10 else "complex"
            
            return {
                "question": question,
                "type": q_type,
                "complexity": complexity,
                "word_count": word_count
            }
    
    # Analyze some questions
    analyzer = QuestionAnalyzer()
    
    questions = [
        "What is machine learning?",
        "Why does gravity exist?",
        "Is Python a good programming language?",
        "How do neural networks learn from data?"
    ]
    
    print("Question Analysis:")
    for q in questions:
        result = analyzer(question=q)
        print(f"\nQ: {q}")
        print(f"   Type: {result['type']}, Complexity: {result['complexity']}")
    
    print("\n" + "="*50)
    print("✅ Key Takeaways")
    print("="*50)
    print("\n1. Operators encapsulate reusable logic")
    print("2. They can be configured and composed")
    print("3. Clear input/output contracts")
    print("4. Foundation for building complex AI systems")
    
    print("\nNext: Learn about type safety in 'type_safety.py'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())