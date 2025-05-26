"""Demo of the new simplified XCS system.

Shows how much simpler and more powerful the new API is.
"""

import time
from typing import List

from ember.xcs.graph.graph import Graph
from ember.xcs.jit.simple_jit import jit, jit_ensemble
from ember.xcs.transforms.simple_transforms import vmap, pmap, compose, parallelize


def demo_basic_graph():
    """Basic graph execution - automatic parallelism discovery."""
    print("\n=== Basic Graph Demo ===")
    
    # Define some operations
    def load_data(source: str) -> dict:
        time.sleep(0.1)  # Simulate I/O
        return {"data": f"data from {source}", "source": source}
    
    def preprocess(inputs: dict) -> dict:
        time.sleep(0.05)  # Simulate processing
        return {"processed": inputs["data"].upper(), "source": inputs["source"]}
    
    def analyze(inputs: dict) -> dict:
        # Receives dict with multiple inputs
        results = []
        for key, val in inputs.items():
            results.append(f"Analysis of {val['source']}: {len(val['processed'])} chars")
        return {"analysis": results}
    
    # Build graph
    graph = Graph()
    
    # Add parallel data loading
    load1 = graph.add(load_data, args=["database"])
    load2 = graph.add(load_data, args=["api"]) 
    load3 = graph.add(load_data, args=["file"])
    
    # Add parallel preprocessing (automatically detected!)
    prep1 = graph.add(preprocess, deps=[load1])
    prep2 = graph.add(preprocess, deps=[load2])
    prep3 = graph.add(preprocess, deps=[load3])
    
    # Add analysis that depends on all
    final = graph.add(analyze, deps=[prep1, prep2, prep3])
    
    # Show structure
    print(graph.visualize())
    
    # Execute - automatic parallelism!
    start = time.time()
    results = graph()
    duration = time.time() - start
    
    print(f"\nExecution time: {duration:.2f}s")
    print(f"Result: {results[final]}")
    
    # Compare with sequential
    start = time.time()
    results_seq = graph(parallel=False)
    duration_seq = time.time() - start
    
    print(f"Sequential time: {duration_seq:.2f}s")
    print(f"Speedup: {duration_seq/duration:.1f}x")


def demo_jit_compilation():
    """JIT compilation - automatic optimization."""
    print("\n\n=== JIT Compilation Demo ===")
    
    # Define a function to compile
    @jit
    def compute_pipeline(x: int) -> int:
        # Multiple operations that can be optimized
        y = x * 2
        z = y + 10
        w = z ** 2
        return w // 3
    
    # First call - compilation
    start = time.time()
    result1 = compute_pipeline(5)
    time1 = time.time() - start
    
    # Second call - cached
    start = time.time()
    result2 = compute_pipeline(5)
    time2 = time.time() - start
    
    print(f"First call (with compilation): {time1*1000:.2f}ms, result: {result1}")
    print(f"Second call (cached): {time2*1000:.2f}ms, result: {result2}")
    print(f"Speedup: {time1/time2:.1f}x")


def demo_vmap_transform():
    """Vectorization - automatic parallel mapping."""
    print("\n\n=== Vectorization Demo ===")
    
    def expensive_computation(x: float) -> float:
        """Simulate expensive single-item processing."""
        time.sleep(0.01)  # 10ms per item
        return x ** 2 + 2 * x + 1
    
    # Vectorize it
    vectorized = vmap(expensive_computation)
    
    # Process batch
    batch = list(range(20))
    
    # Time vectorized version
    start = time.time()
    results_vmap = vectorized(batch)
    time_vmap = time.time() - start
    
    # Time sequential version
    start = time.time()
    results_seq = [expensive_computation(x) for x in batch]
    time_seq = time.time() - start
    
    print(f"Sequential: {time_seq:.2f}s")
    print(f"Vectorized: {time_vmap:.2f}s")
    print(f"Speedup: {time_seq/time_vmap:.1f}x")
    print(f"Results match: {results_vmap == results_seq}")


def demo_ensemble_pattern():
    """Ensemble-judge pattern - automatic detection."""
    print("\n\n=== Ensemble-Judge Pattern Demo ===")
    
    # Define judge functions
    def judge_quality(text: str) -> dict:
        time.sleep(0.05)
        return {"quality": len(text) > 50, "score": 0.8}
    
    def judge_accuracy(text: str) -> dict:
        time.sleep(0.05)
        return {"accuracy": "error" not in text.lower(), "score": 0.9}
    
    def judge_clarity(text: str) -> dict:
        time.sleep(0.05)
        return {"clarity": text.count(".") > 2, "score": 0.7}
    
    def synthesize_judgments(inputs: dict) -> dict:
        """Combine all judgments."""
        all_scores = []
        all_criteria = {}
        
        for judge_name, judgment in inputs.items():
            all_scores.append(judgment["score"])
            all_criteria.update(judgment)
        
        return {
            "final_score": sum(all_scores) / len(all_scores),
            "criteria": all_criteria,
            "passed": all(all_criteria.values())
        }
    
    # Create ensemble using simplified API
    ensemble = parallelize(
        [judge_quality, judge_accuracy, judge_clarity],
        synthesize_judgments
    )
    
    # Or manually with graph (more control)
    def manual_ensemble(text: str) -> dict:
        graph = Graph()
        
        # Add judges - automatically detected as parallel!
        j1 = graph.add(judge_quality, args=[text])
        j2 = graph.add(judge_accuracy, args=[text])
        j3 = graph.add(judge_clarity, args=[text])
        
        # Add synthesizer
        final = graph.add(synthesize_judgments, deps=[j1, j2, j3])
        
        # Execute
        results = graph()
        return results[final]
    
    # Test both versions
    test_text = "This is a test. It has multiple sentences. No errors here."
    
    start = time.time()
    result1 = ensemble(test_text)
    time1 = time.time() - start
    
    start = time.time()
    result2 = manual_ensemble(test_text)
    time2 = time.time() - start
    
    print(f"Parallel ensemble time: {time1:.3f}s")
    print(f"Result: {result1}")
    print(f"\nManual graph time: {time2:.3f}s")
    print(f"Result: {result2}")


def demo_composition():
    """Function composition with optimization."""
    print("\n\n=== Composition Demo ===")
    
    def step1(x: int) -> int:
        print(f"  step1({x}) = {x} + 1")
        return x + 1
    
    def step2(x: int) -> int:
        print(f"  step2({x}) = {x} * 2")
        return x * 2
    
    def step3(x: int) -> int:
        print(f"  step3({x}) = {x} ** 2")
        return x ** 2
    
    # Compose functions
    pipeline = compose(step3, step2, step1)
    
    # Execute composed function
    print("Executing: compose(step3, step2, step1)(5)")
    result = pipeline(5)
    print(f"Result: {result}")
    print(f"Expected: ((5 + 1) * 2) ** 2 = {((5 + 1) * 2) ** 2}")


def demo_advanced_patterns():
    """Advanced pattern detection and optimization."""
    print("\n\n=== Advanced Pattern Detection ===")
    
    # Build a complex graph
    graph = Graph()
    
    def process(x: int, tag: str) -> dict:
        return {"value": x * 2, "tag": tag}
    
    # Add map pattern (same function, different inputs)
    nodes_map = []
    for i in range(5):
        node = graph.add(process, args=[i, f"item_{i}"])
        nodes_map.append(node)
    
    # Add reduction
    def aggregate(inputs: dict) -> dict:
        values = [inputs[k]["value"] for k in inputs]
        return {"sum": sum(values), "count": len(values)}
    
    reducer = graph.add(aggregate, deps=nodes_map)
    
    # Analyze patterns
    analysis = graph._analyze_graph()
    print("Detected patterns:")
    for pattern_type, instances in analysis['patterns'].items():
        if instances:
            print(f"  {pattern_type}: {len(instances)} instance(s)")
    
    # Execute and show optimization
    start = time.time()
    results = graph()
    duration = time.time() - start
    
    print(f"\nExecution time: {duration:.3f}s")
    print(f"Final result: {results[reducer]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Simplified XCS System Demo")
    print("=" * 60)
    
    # Run all demos
    demo_basic_graph()
    demo_jit_compilation()
    demo_vmap_transform()
    demo_ensemble_pattern()
    demo_composition()
    demo_advanced_patterns()
    
    print("\n" + "=" * 60)
    print("Summary: Simpler API, More Power!")
    print("=" * 60)