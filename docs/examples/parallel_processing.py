"""
Example: Parallel Processing
Description: Efficiently process multiple items in parallel
Concepts: Parallelization, batching, progress tracking, performance optimization
"""

from ember.api import ember
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import time


class DocumentSummary(BaseModel):
    title: str
    summary: str
    key_points: List[str]
    word_count: int
    category: str


class SentimentAnalysis(BaseModel):
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    emotions: List[str]


async def main():
    # Example 1: Basic Parallel Processing
    print("=== Basic Parallel Processing ===")
    
    documents = [
        "The new quantum computer achieved 100 qubits, marking a significant milestone in computing.",
        "Climate change effects are accelerating, with record temperatures observed globally.",
        "The stock market reached new highs despite ongoing economic uncertainty.",
        "Breakthrough in cancer research shows promising results in early trials.",
        "New archaeological discovery sheds light on ancient civilization.",
    ]
    
    @ember.op
    async def summarize_document(doc: str) -> str:
        """Summarize a document in one sentence."""
        return await ember.llm(f"Summarize in one sentence: {doc}")
    
    # Process all documents in parallel
    start_time = time.time()
    summaries = await ember.parallel([
        summarize_document(doc) for doc in documents
    ])
    parallel_time = time.time() - start_time
    
    print(f"Processed {len(documents)} documents in {parallel_time:.2f} seconds")
    for i, summary in enumerate(summaries):
        print(f"{i+1}. {summary}")
    
    print()

    # Example 2: Batch Processing with Progress
    print("=== Batch Processing with Progress ===")
    
    # Generate more documents for batching example
    large_dataset = [f"Document {i}: " + doc for i in range(20) for doc in documents]
    
    @ember.op
    async def analyze_batch(batch: List[str]) -> List[Dict[str, Any]]:
        """Analyze a batch of documents."""
        results = await ember.parallel([
            ember.llm(
                f"Analyze this text and return sentiment and main topic: {doc}",
                output_type=dict
            ) for doc in batch
        ])
        return results
    
    # Process in batches
    batch_size = 5
    all_results = []
    
    print(f"Processing {len(large_dataset)} documents in batches of {batch_size}...")
    for i, batch in enumerate(ember.batch(large_dataset, size=batch_size)):
        print(f"Processing batch {i+1}/{len(large_dataset)//batch_size + 1}...", end="", flush=True)
        batch_results = await analyze_batch(batch)
        all_results.extend(batch_results)
        print(" Done!")
    
    print(f"Total processed: {len(all_results)} documents")
    print()

    # Example 3: Streaming with Parallel Processing
    print("=== Streaming with Parallel Processing ===")
    
    @ember.op
    async def enrich_document(doc: str) -> DocumentSummary:
        """Enrich document with detailed analysis."""
        prompt = f"""Analyze this document and provide:
        - A title
        - A summary
        - 3-5 key points
        - Word count estimate
        - Category (tech, science, business, health, or other)
        
        Document: {doc}"""
        
        return await ember.llm(prompt, output_type=DocumentSummary)
    
    # Stream results as they complete
    async for i, result in ember.stream(documents[:5], enrich_document):
        print(f"\nDocument {i+1} completed:")
        print(f"  Title: {result.title}")
        print(f"  Category: {result.category}")
        print(f"  Key points: {len(result.key_points)}")
    
    print()

    # Example 4: Multi-Model Parallel Comparison
    print("=== Multi-Model Parallel Comparison ===")
    
    test_prompt = "What are the three most important factors in machine learning model performance?"
    
    models = ["gpt-4", "claude-3", "gemini-pro"]  # Use actual available models
    
    @ember.op
    async def get_model_response(model_name: str, prompt: str) -> Dict[str, Any]:
        """Get response from a specific model."""
        try:
            response = await ember.llm(prompt, model=model_name)
            return {
                "model": model_name,
                "response": response,
                "success": True
            }
        except Exception as e:
            return {
                "model": model_name,
                "error": str(e),
                "success": False
            }
    
    # Query all models in parallel
    responses = await ember.parallel([
        get_model_response(model, test_prompt) for model in models
    ])
    
    print("Model Comparison Results:")
    for resp in responses:
        if resp["success"]:
            print(f"\n{resp['model']}:")
            print(f"  {resp['response'][:150]}...")
        else:
            print(f"\n{resp['model']}: Error - {resp['error']}")
    
    print()

    # Example 5: Complex Pipeline with Mixed Operations
    print("=== Complex Pipeline ===")
    
    @ember.op
    async def extract_entities(text: str) -> List[str]:
        """Extract named entities from text."""
        result = await ember.llm(
            f"Extract all named entities (people, places, organizations) from: {text}",
            output_type=List[str]
        )
        return result
    
    @ember.op
    async def analyze_sentiment(text: str) -> SentimentAnalysis:
        """Analyze sentiment of text."""
        return await ember.llm(
            f"Analyze the sentiment of: {text}",
            output_type=SentimentAnalysis
        )
    
    @ember.op
    async def generate_tags(text: str, entities: List[str]) -> List[str]:
        """Generate tags based on text and entities."""
        prompt = f"""Generate 3-5 relevant tags for this text.
        Text: {text}
        Entities found: {', '.join(entities)}"""
        
        return await ember.llm(prompt, output_type=List[str])
    
    # Process documents through pipeline
    async def process_document_pipeline(doc: str) -> Dict[str, Any]:
        """Run document through analysis pipeline."""
        # Run entity extraction and sentiment analysis in parallel
        entities, sentiment = await ember.parallel([
            extract_entities(doc),
            analyze_sentiment(doc)
        ])
        
        # Generate tags based on results
        tags = await generate_tags(doc, entities)
        
        return {
            "original": doc[:50] + "...",
            "entities": entities,
            "sentiment": sentiment.sentiment,
            "confidence": sentiment.confidence,
            "tags": tags
        }
    
    # Process multiple documents through pipeline
    pipeline_results = await ember.parallel([
        process_document_pipeline(doc) for doc in documents[:3]
    ])
    
    print("Pipeline Results:")
    for i, result in enumerate(pipeline_results):
        print(f"\nDocument {i+1}:")
        print(f"  Entities: {', '.join(result['entities'])}")
        print(f"  Sentiment: {result['sentiment']} ({result['confidence']:.2f})")
        print(f"  Tags: {', '.join(result['tags'])}")
    
    # Example 6: Performance Comparison
    print("\n=== Performance Comparison ===")
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for doc in documents[:5]:
        result = await summarize_document(doc)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"Sequential processing: {sequential_time:.2f} seconds")
    print(f"Parallel processing: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Example 7: Error Handling in Parallel Operations
    print("\n=== Error Handling in Parallel Operations ===")
    
    @ember.op
    async def risky_operation(text: str) -> str:
        """Operation that might fail."""
        if "error" in text.lower():
            raise ValueError("Found error keyword!")
        return await ember.llm(f"Process: {text}")
    
    mixed_inputs = [
        "Normal text 1",
        "This contains ERROR",
        "Normal text 2",
        "Another ERROR here",
        "Normal text 3"
    ]
    
    # Process with error handling
    async def safe_process(text: str) -> Dict[str, Any]:
        try:
            result = await risky_operation(text)
            return {"input": text, "output": result, "success": True}
        except Exception as e:
            return {"input": text, "error": str(e), "success": False}
    
    results = await ember.parallel([
        safe_process(text) for text in mixed_inputs
    ])
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"Processed {len(results)} items: {successful} successful, {failed} failed")
    for r in results:
        if not r["success"]:
            print(f"  Failed: {r['input'][:20]}... - {r['error']}")


if __name__ == "__main__":
    print("Ember Parallel Processing Example")
    print("=" * 50)
    asyncio.run(main())