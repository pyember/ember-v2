"""Clean example of the new graph system.

No compatibility layers, no special calling conventions. Just functions.
"""

import time
from typing import Any, Dict, List

from ember.xcs.graph.graph import Graph, pipeline, parallel, fan_out_fan_in


def example_1_basic():
    """Basic graph execution."""
    print("\n=== Example 1: Basic Graph ===")
    
    # Simple functions - no special signatures
    def double(x: int) -> int:
        return x * 2
    
    def add_ten(x: int) -> int:
        return x + 10
    
    def square(x: int) -> int:
        return x ** 2
    
    # Build graph
    graph = Graph()
    n1 = graph.add(double)
    n2 = graph.add(add_ten, deps=[n1])
    n3 = graph.add(square, deps=[n2])
    
    # Execute
    result = graph.execute({"data": 5})
    
    print(f"Input: 5")
    print(f"After double: {result[n1]}")
    print(f"After add_ten: {result[n2]}")
    print(f"After square: {result[n3]}")
    print(f"Final: ((5 * 2) + 10) ** 2 = {result[n3]}")


def example_2_parallel():
    """Parallel execution."""
    print("\n\n=== Example 2: Parallel Execution ===")
    
    def slow_operation(name: str, delay: float):
        def op(data: Dict[str, Any]) -> str:
            print(f"  {name} starting...")
            time.sleep(delay)
            print(f"  {name} done!")
            return f"{name} processed {data}"
        op.__name__ = name
        return op
    
    # Build graph with parallel operations
    graph = Graph()
    
    # These will run in parallel
    op1 = graph.add(slow_operation("Operation1", 0.2))
    op2 = graph.add(slow_operation("Operation2", 0.2))
    op3 = graph.add(slow_operation("Operation3", 0.2))
    
    # This depends on all three
    def combine(deps: Dict[str, str]) -> str:
        return f"Combined: {', '.join(deps.values())}"
    
    final = graph.add(combine, deps=[op1, op2, op3])
    
    # Time execution
    start = time.time()
    result = graph.execute({"input": "data"})
    duration = time.time() - start
    
    print(f"\nResult: {result[final]}")
    print(f"Time: {duration:.2f}s (should be ~0.2s with parallelism)")


def example_3_pipeline():
    """Using the pipeline helper."""
    print("\n\n=== Example 3: Pipeline ===")
    
    # Chain of transformations
    def parse(data: Dict[str, str]) -> List[str]:
        return data["text"].split()
    
    def filter_short(words: List[str]) -> List[str]:
        return [w for w in words if len(w) > 3]
    
    def uppercase(words: List[str]) -> List[str]:
        return [w.upper() for w in words]
    
    def join(words: List[str]) -> str:
        return " ".join(words)
    
    # Create pipeline
    process = pipeline(parse, filter_short, uppercase, join)
    
    # Execute
    result = process({"text": "the quick brown fox jumps over the lazy dog"})
    print(f"Result: {result}")


def example_4_fan_out_fan_in():
    """Fan-out/fan-in pattern."""
    print("\n\n=== Example 4: Fan-out/Fan-in ===")
    
    def split_data(data: Dict[str, List[int]]) -> List[List[int]]:
        """Split data into chunks."""
        items = data["numbers"]
        chunk_size = len(items) // 3
        return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
    
    def process_chunk(chunk: List[int]) -> int:
        """Process a chunk (sum in this case)."""
        return sum(chunk)
    
    def combine_results(results: Dict[str, int]) -> int:
        """Combine processed chunks."""
        return sum(results.values())
    
    # Create fan-out/fan-in
    process = fan_out_fan_in(
        split_data,
        process_chunk, process_chunk, process_chunk,  # 3 parallel processors
        combine_results
    )
    
    # Execute
    numbers = list(range(1, 10))
    result = process({"numbers": numbers})
    print(f"Sum of {numbers} = {result}")


def example_5_real_world():
    """Real-world example: Text analysis pipeline."""
    print("\n\n=== Example 5: Real-World Text Analysis ===")
    
    # Analysis functions
    def count_words(text: str) -> int:
        return len(text.split())
    
    def count_sentences(text: str) -> int:
        return text.count('.') + text.count('!') + text.count('?')
    
    def extract_keywords(text: str) -> List[str]:
        words = text.lower().split()
        # Simple keyword extraction - words longer than 5 chars
        return list(set(w for w in words if len(w) > 5))
    
    def sentiment_score(text: str) -> float:
        # Mock sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing']
        negative_words = ['bad', 'terrible', 'awful', 'horrible']
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        total = pos_count + neg_count
        return (pos_count - neg_count) / total if total > 0 else 0.0
    
    def summarize(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all analysis results."""
        return {
            "word_count": analysis["words"],
            "sentence_count": analysis["sentences"],
            "keywords": analysis["keywords"],
            "sentiment": analysis["sentiment"],
            "complexity": analysis["words"] / analysis["sentences"]
        }
    
    # Build analysis graph
    graph = Graph()
    
    # Input processing
    def extract_text(data: Dict[str, str]) -> str:
        return data["document"]
    
    text_node = graph.add(extract_text)
    
    # Parallel analysis
    words = graph.add(count_words, deps=[text_node])
    sentences = graph.add(count_sentences, deps=[text_node])
    keywords = graph.add(extract_keywords, deps=[text_node])
    sentiment = graph.add(sentiment_score, deps=[text_node])
    
    # Combine results
    summary = graph.add(summarize, deps=[words, sentences, keywords, sentiment])
    
    # Execute
    document = """
    This is an excellent example of clean code. The graph system is amazing 
    and makes parallel execution simple. No more terrible compatibility layers 
    or awful special calling conventions. Just good, clean functions.
    """
    
    result = graph.execute({"document": document})
    
    print("Document analysis:")
    for key, value in result[summary].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("Clean Graph System Examples")
    print("=" * 50)
    
    example_1_basic()
    example_2_parallel()
    example_3_pipeline()
    example_4_fan_out_fan_in()
    example_5_real_world()