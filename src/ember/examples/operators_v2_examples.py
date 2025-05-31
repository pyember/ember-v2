"""Examples of the new operator system in action.

These examples show how the v2 operator design creates simple, composable
code that works naturally with Python and the existing models API.
"""

from ember.api import models
from ember.xcs import jit, vmap
from ember.core.operators_v2.ensemble import Ensemble, ensemble
from ember.core.operators_v2.selectors import most_common, best_of, MajorityVote
from ember.core.operators_v2.judges import create_selector_judge, create_synthesis_judge


def example_simple_function():
    """Any function is an operator - no base class needed."""
    
    def sentiment_classifier(text: str) -> str:
        """Classify sentiment using an LLM."""
        response = models("gpt-4", f"Classify sentiment as positive/negative/neutral: {text}")
        return response.text.strip().lower()
    
    # Use it directly
    result = sentiment_classifier("I love this product!")
    print(f"Sentiment: {result}")
    
    # Make it fast with JIT
    fast_classifier = jit(sentiment_classifier)
    
    # Make it work on batches
    batch_classifier = vmap(sentiment_classifier)
    results = batch_classifier([
        "Great service!",
        "Terrible experience",
        "It was okay"
    ])
    print(f"Batch results: {results}")


def example_ensemble_pattern():
    """Ensemble multiple models for robustness."""
    
    # Create models with different parameters
    creative = models.instance("gpt-4", temperature=0.9)
    balanced = models.instance("gpt-4", temperature=0.5)
    precise = models.instance("gpt-4", temperature=0.1)
    
    # Create a story generator ensemble
    def generate_story(prompt: str) -> str:
        full_prompt = f"Write a 2-sentence story about: {prompt}"
        return full_prompt
    
    # Ensemble approach 1: Using the class
    story_ensemble = Ensemble([
        lambda p: creative(generate_story(p)).text,
        lambda p: balanced(generate_story(p)).text,
        lambda p: precise(generate_story(p)).text
    ])
    
    stories = story_ensemble("a robot learning to paint")
    print("\nEnsemble stories:")
    for i, story in enumerate(stories):
        print(f"{i+1}. {story}")
    
    # Select the best one
    judge = create_selector_judge(models.instance("gpt-4"))
    result = judge("Write a creative story about a robot learning to paint", stories)
    print(f"\nBest story: {result.selected}")
    print(f"Reasoning: {result.reasoning}")


def example_classification_pipeline():
    """Build a robust classification pipeline."""
    
    # Step 1: Create multiple classifiers
    classifiers = [
        models.instance("gpt-4", temperature=0.3),
        models.instance("gpt-3.5-turbo", temperature=0.3),
        models.instance("claude-3", temperature=0.3)
    ]
    
    def classify(model, text):
        response = model(f"Classify topic as tech/sports/politics/other: {text}")
        return response.text.strip().lower()
    
    # Step 2: Ensemble them
    ensemble_classify = ensemble(
        *[lambda t, m=model: classify(m, t) for model in classifiers]
    )
    
    # Step 3: Use majority vote
    majority = MajorityVote(threshold=0.6)
    
    def robust_classifier(text: str) -> str:
        predictions = ensemble_classify(text)
        return majority(predictions)
    
    # Make it fast
    fast_robust_classifier = jit(robust_classifier)
    
    # Test it
    texts = [
        "The new iPhone has amazing features",
        "The Lakers won the championship",
        "Congress passed the new bill"
    ]
    
    for text in texts:
        result = fast_robust_classifier(text)
        print(f"\n'{text[:30]}...' -> {result}")


def example_generative_pipeline():
    """Build a generate-then-verify pipeline."""
    
    # Generator
    generator = models.instance("gpt-4", temperature=0.8)
    
    # Verifier  
    verifier = models.instance("gpt-4", temperature=0.1)
    
    def generate_code(description: str) -> str:
        prompt = f"Write Python code to: {description}\nCode:"
        response = generator(prompt)
        return response.text
    
    def verify_code(description: str, code: str) -> bool:
        prompt = f"""Does this code correctly implement: {description}

Code:
{code}

Answer YES or NO with a brief reason."""
        response = verifier(prompt)
        return "yes" in response.text.lower()
    
    def generate_verified_code(description: str, max_attempts: int = 3) -> str:
        """Generate code with verification."""
        for attempt in range(max_attempts):
            code = generate_code(description)
            if verify_code(description, code):
                return code
        
        # If no valid code after attempts, return last attempt
        return code
    
    # Use it
    code = generate_verified_code("calculate fibonacci numbers")
    print(f"\nGenerated code:\n{code}")


def example_advanced_composition():
    """Show how operators compose into complex systems."""
    
    # Create a research assistant that:
    # 1. Generates multiple perspectives on a topic
    # 2. Synthesizes them into a comprehensive answer
    # 3. Fact-checks the result
    
    # Perspective generators
    historian = models.instance("gpt-4", system="You are a historian")
    scientist = models.instance("gpt-4", system="You are a scientist")  
    philosopher = models.instance("gpt-4", system="You are a philosopher")
    
    def get_perspective(expert, question):
        return expert(question).text
    
    # Synthesis
    synthesizer = create_synthesis_judge(models.instance("gpt-4"))
    
    # Fact checker
    def fact_check(claim: str) -> str:
        response = models("gpt-4", f"Fact-check this claim: {claim}")
        return response.text
    
    # Compose into research assistant
    def research_assistant(question: str) -> dict:
        # Get perspectives
        perspectives = [
            get_perspective(historian, question),
            get_perspective(scientist, question),
            get_perspective(philosopher, question)
        ]
        
        # Synthesize
        synthesis = synthesizer(question, perspectives)
        
        # Fact check
        verification = fact_check(synthesis)
        
        return {
            "answer": synthesis,
            "perspectives": perspectives,
            "verification": verification
        }
    
    # Make it fast
    fast_research = jit(research_assistant)
    
    # Use it
    result = fast_research("What caused the fall of the Roman Empire?")
    print(f"\nResearch result: {result['answer'][:200]}...")


if __name__ == "__main__":
    print("=== Simple Function Example ===")
    example_simple_function()
    
    print("\n=== Ensemble Pattern Example ===")
    example_ensemble_pattern()
    
    print("\n=== Classification Pipeline Example ===")
    example_classification_pipeline()
    
    print("\n=== Generative Pipeline Example ===")
    example_generative_pipeline()
    
    print("\n=== Advanced Composition Example ===")
    example_advanced_composition()