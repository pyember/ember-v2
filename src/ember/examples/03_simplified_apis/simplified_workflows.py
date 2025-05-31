"""Simplified Workflows - Complete AI pipelines with minimal boilerplate.

Shows how to build production-ready workflows using Ember's simplified APIs.
No base classes, no complex configuration - just Python that works.
"""

from ember.api import models, data, eval
from ember.api.xcs import jit, vmap
import json


def example_content_moderation_workflow():
    """Build a content moderation pipeline with multiple checks."""
    print("\n=== Content Moderation Workflow ===\n")
    
    # Step 1: Define moderation functions - they're just Python functions!
    def check_toxicity(text: str) -> dict:
        """Check for toxic content."""
        prompt = f"Is this text toxic or harmful? Answer YES/NO with reason:\n{text}"
        response = models("gpt-3.5-turbo", prompt).text
        
        is_toxic = "yes" in response.lower()
        reason = response.split(":", 1)[-1].strip() if ":" in response else response
        
        return {"toxic": is_toxic, "reason": reason}
    
    def check_spam(text: str) -> dict:
        """Check for spam content."""
        prompt = f"Is this spam? Answer YES/NO with confidence 0-1:\n{text}"
        response = models("gpt-3.5-turbo", prompt).text
        
        is_spam = "yes" in response.lower()
        confidence = 0.5  # Default
        if "confidence" in response.lower():
            try:
                confidence = float(response.split()[-1])
            except:
                pass
                
        return {"spam": is_spam, "confidence": confidence}
    
    def check_pii(text: str) -> dict:
        """Check for personally identifiable information."""
        prompt = f"Does this contain PII (names, emails, phones)? List any found:\n{text}"
        response = models("gpt-3.5-turbo", prompt).text
        
        has_pii = any(keyword in response.lower() for keyword in ["yes", "contains", "found"])
        return {"has_pii": has_pii, "details": response}
    
    # Step 2: Compose into a complete workflow
    @jit  # Make the entire workflow fast!
    def moderate_content(text: str) -> dict:
        """Complete moderation workflow."""
        # Run all checks
        toxicity = check_toxicity(text)
        spam = check_spam(text)
        pii = check_pii(text)
        
        # Aggregate results
        is_safe = not (toxicity["toxic"] or spam["spam"] or pii["has_pii"])
        
        return {
            "text": text,
            "safe": is_safe,
            "checks": {
                "toxicity": toxicity,
                "spam": spam,
                "pii": pii
            }
        }
    
    # Step 3: Use it on real content
    test_messages = [
        "Check out this amazing offer! Click here now!!!",
        "Thanks for the helpful response to my question.",
        "My email is john@example.com and phone is 555-1234"
    ]
    
    for message in test_messages:
        result = moderate_content(message)
        print(f"Message: '{message[:50]}...'")
        print(f"Safe: {result['safe']}")
        print(f"Details: {json.dumps(result['checks'], indent=2)}\n")


def example_data_processing_workflow():
    """Process datasets with automatic optimization."""
    print("\n=== Data Processing Workflow ===\n")
    
    # Define processing functions
    def analyze_question(item: dict) -> dict:
        """Analyze a question's difficulty and topic."""
        question = item.get("question", "")
        
        # Analyze difficulty
        difficulty_prompt = f"Rate difficulty (easy/medium/hard): {question}"
        difficulty = models("gpt-3.5-turbo", difficulty_prompt).text.strip()
        
        # Identify topic
        topic_prompt = f"Main topic (math/science/history/other): {question}"
        topic = models("gpt-3.5-turbo", topic_prompt).text.strip()
        
        return {
            **item,  # Keep original data
            "difficulty": difficulty,
            "topic": topic
        }
    
    def generate_explanation(item: dict) -> dict:
        """Generate an explanation for the answer."""
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        prompt = f"Explain why the answer to '{question}' is '{answer}' in 1-2 sentences."
        explanation = models("gpt-3.5-turbo", prompt).text
        
        return {
            **item,
            "explanation": explanation
        }
    
    # Compose into pipeline with automatic batching
    @vmap  # Process multiple items in parallel!
    def process_batch(items):
        """Process a batch of items."""
        analyzed = analyze_question(items)
        with_explanations = generate_explanation(analyzed)
        return with_explanations
    
    # Simulate dataset processing
    sample_data = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
        {"question": "What is the speed of light?", "answer": "299,792,458 m/s"}
    ]
    
    # Process with automatic optimization
    results = process_batch(sample_data)
    
    for result in results:
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Difficulty: {result.get('difficulty', 'N/A')}")
        print(f"Topic: {result.get('topic', 'N/A')}")
        print(f"Explanation: {result.get('explanation', 'N/A')}\n")


def example_evaluation_workflow():
    """Build an evaluation pipeline for model outputs."""
    print("\n=== Evaluation Workflow ===\n")
    
    # Define evaluation criteria
    def evaluate_accuracy(prediction: str, ground_truth: str) -> float:
        """Check if prediction matches ground truth."""
        prompt = f"""
        Prediction: {prediction}
        Ground truth: {ground_truth}
        
        Are these equivalent? Score 0-1 where 1 is perfect match.
        """
        response = models("gpt-3.5-turbo", prompt).text
        
        try:
            # Extract score from response
            score = float(''.join(c for c in response if c.isdigit() or c == '.'))
            return min(max(score, 0), 1)  # Clamp to [0, 1]
        except:
            return 0.0
    
    def evaluate_quality(text: str) -> dict:
        """Evaluate text quality on multiple dimensions."""
        prompts = {
            "clarity": f"Rate clarity 0-1: {text}",
            "completeness": f"Rate completeness 0-1: {text}",
            "correctness": f"Rate factual correctness 0-1: {text}"
        }
        
        scores = {}
        for dimension, prompt in prompts.items():
            response = models("gpt-3.5-turbo", prompt).text
            try:
                score = float(''.join(c for c in response if c.isdigit() or c == '.'))
                scores[dimension] = min(max(score, 0), 1)
            except:
                scores[dimension] = 0.5
                
        return scores
    
    # Compose into evaluation pipeline
    def evaluate_model_output(task: str, prediction: str, ground_truth: str = None) -> dict:
        """Complete evaluation of model output."""
        result = {
            "task": task,
            "prediction": prediction,
            "quality_scores": evaluate_quality(prediction)
        }
        
        if ground_truth:
            result["accuracy"] = evaluate_accuracy(prediction, ground_truth)
            
        # Overall score
        quality_avg = sum(result["quality_scores"].values()) / len(result["quality_scores"])
        if ground_truth:
            result["overall_score"] = (result["accuracy"] + quality_avg) / 2
        else:
            result["overall_score"] = quality_avg
            
        return result
    
    # Make it fast for large-scale evaluation
    fast_evaluate = jit(evaluate_model_output)
    
    # Example evaluations
    test_cases = [
        {
            "task": "What is the capital of France?",
            "prediction": "The capital of France is Paris.",
            "ground_truth": "Paris"
        },
        {
            "task": "Explain photosynthesis",
            "prediction": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "ground_truth": None  # No single correct answer
        }
    ]
    
    for test in test_cases:
        result = fast_evaluate(**test)
        print(f"Task: {test['task']}")
        print(f"Prediction: {test['prediction']}")
        print(f"Scores: {json.dumps(result['quality_scores'], indent=2)}")
        if "accuracy" in result:
            print(f"Accuracy: {result['accuracy']:.2f}")
        print(f"Overall: {result['overall_score']:.2f}\n")


def example_rag_workflow():
    """Retrieval-Augmented Generation workflow."""
    print("\n=== RAG Workflow ===\n")
    
    # Simulate a document store
    documents = [
        "The Eiffel Tower is 330 meters tall and located in Paris, France.",
        "The Great Wall of China is over 21,000 kilometers long.",
        "The Statue of Liberty was a gift from France to the United States.",
        "Mount Everest is 8,849 meters tall and located on the Nepal-Tibet border.",
        "The Amazon River is approximately 6,400 kilometers long."
    ]
    
    def retrieve_relevant_docs(query: str, docs: list, top_k: int = 2) -> list:
        """Retrieve relevant documents for a query."""
        # Score each document's relevance
        scored_docs = []
        for doc in docs:
            prompt = f"Rate relevance 0-1 of this document to the query:\nQuery: {query}\nDoc: {doc}"
            response = models("gpt-3.5-turbo", prompt).text
            
            try:
                score = float(''.join(c for c in response if c.isdigit() or c == '.'))
                scored_docs.append((score, doc))
            except:
                scored_docs.append((0.5, doc))
        
        # Sort by score and return top-k
        scored_docs.sort(reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
    
    def generate_answer(query: str, context: list) -> str:
        """Generate answer using retrieved context."""
        context_str = "\n".join(f"- {doc}" for doc in context)
        prompt = f"""
        Answer this question using the provided context:
        
        Context:
        {context_str}
        
        Question: {query}
        
        Answer:
        """
        return models("gpt-3.5-turbo", prompt).text
    
    # Complete RAG pipeline
    @jit
    def rag_pipeline(query: str) -> dict:
        """Full RAG workflow."""
        # Retrieve
        relevant_docs = retrieve_relevant_docs(query, documents)
        
        # Generate
        answer = generate_answer(query, relevant_docs)
        
        return {
            "query": query,
            "context": relevant_docs,
            "answer": answer.strip()
        }
    
    # Test queries
    queries = [
        "How tall is the Eiffel Tower?",
        "What is the longest river?",
        "Which monument was a gift from France?"
    ]
    
    for query in queries:
        result = rag_pipeline(query)
        print(f"Q: {query}")
        print(f"Context: {result['context'][0][:50]}...")
        print(f"A: {result['answer']}\n")


def main():
    """Run all workflow examples."""
    import os
    
    if not any(os.environ.get(key) for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']):
        print("\n⚠️  No API keys found. This example requires model API access.")
        print("Run 'ember init' to configure your API keys.\n")
        print("Workflow examples showcase:")
        print("1. Content moderation - Multi-step safety checks")
        print("2. Data processing - Batch operations with vmap")
        print("3. Evaluation - Quality and accuracy scoring")
        print("4. RAG - Retrieval-augmented generation")
        print("\nAll with minimal code and automatic optimization!")
        return
    
    example_content_moderation_workflow()
    example_data_processing_workflow()
    example_evaluation_workflow()
    example_rag_workflow()
    
    print("\n✨ Key takeaway: Complex workflows are just composed functions!")


if __name__ == "__main__":
    main()