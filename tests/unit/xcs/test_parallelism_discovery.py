"""Test and demonstrate automatic parallelism discovery."""

import time
from ember.xcs.graph import Graph


def test_ensemble_parallelism_discovery():
    """Demonstrate how ensemble parallelism is automatically discovered."""
    
    # Build an ensemble-judge graph
    graph = Graph()
    
    # Data source
    data = graph.add(lambda: {"features": [1, 2, 3, 4, 5]})
    
    # Preprocessing
    normalized = graph.add(
        lambda x: {"features": [f/5.0 for f in x["features"]]},
        deps=[data]
    )
    
    # Ensemble members - these will be discovered as parallelizable!
    model1 = graph.add(
        lambda x: sum(x["features"]) * 1.1,  # Model 1 logic
        deps=[normalized]
    )
    
    model2 = graph.add(
        lambda x: sum(x["features"]) * 1.2,  # Model 2 logic
        deps=[normalized]
    )
    
    model3 = graph.add(
        lambda x: sum(x["features"]) * 0.9,  # Model 3 logic
        deps=[normalized]
    )
    
    # Judge - automatically waits for all models
    judge = graph.add(
        lambda m1, m2, m3: (m1 + m2 + m3) / 3,  # Average ensemble
        deps=[model1, model2, model3]
    )
    
    # Let's examine the discovered waves
    print("\n=== Parallelism Discovery Demo ===\n")
    
    # The graph hasn't computed waves yet
    assert graph._waves is None
    
    # Accessing stats triggers wave computation
    stats = graph.stats
    
    print(f"Graph structure:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Waves: {stats['waves']}")
    print(f"  Max parallelism: {stats['parallelism']}")
    
    # Now let's see the actual waves
    print(f"\nDiscovered execution waves:")
    for i, wave in enumerate(graph._waves):
        print(f"  Wave {i+1}: {wave}")
        if len(wave) > 1:
            print(f"    ^ These {len(wave)} nodes run IN PARALLEL!")
    
    # Verify the waves are correct
    assert len(graph._waves) == 4
    assert len(graph._waves[0]) == 1  # Data source
    assert len(graph._waves[1]) == 1  # Normalization
    assert len(graph._waves[2]) == 3  # Three models in parallel!
    assert len(graph._waves[3]) == 1  # Judge
    
    # Execute and verify results
    results = graph.run()
    
    # The three models should have run in parallel
    # Let's verify the math
    features_sum = sum([1/5, 2/5, 3/5, 4/5, 5/5])  # 3.0
    expected_m1 = features_sum * 1.1  # 3.3
    expected_m2 = features_sum * 1.2  # 3.6
    expected_m3 = features_sum * 0.9  # 2.7
    expected_judge = (expected_m1 + expected_m2 + expected_m3) / 3  # 3.2
    
    assert abs(results[judge] - expected_judge) < 0.0001
    
    print(f"\nExecution results:")
    print(f"  Models ran in parallel: {[results[m] for m in [model1, model2, model3]]}")
    print(f"  Judge result: {results[judge]}")


def test_complex_dag_parallelism():
    """Test parallelism discovery in a complex DAG."""
    
    graph = Graph()
    
    # Layer 1: Two parallel data sources
    source1 = graph.add(lambda: 10, deps=[])
    source2 = graph.add(lambda: 20, deps=[])
    
    # Layer 2: Diamond pattern
    proc1 = graph.add(lambda x: x * 2, deps=[source1])
    proc2 = graph.add(lambda x, y: x + y, deps=[source1, source2])
    proc3 = graph.add(lambda x: x / 2, deps=[source2])
    
    # Layer 3: More processing
    combine1 = graph.add(lambda a, b: a + b, deps=[proc1, proc2])
    combine2 = graph.add(lambda b, c: b * c, deps=[proc2, proc3])
    
    # Layer 4: Final
    final = graph.add(lambda x, y: x + y, deps=[combine1, combine2])
    
    # Analyze waves
    stats = graph.stats
    waves = graph._waves
    
    print("\n=== Complex DAG Parallelism ===\n")
    print("Graph structure:")
    print(f"  {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"  Critical path length: {stats['critical_path']}")
    print(f"  Maximum parallelism: {stats['parallelism']}")
    
    print("\nExecution waves:")
    for i, wave in enumerate(waves):
        parallel_marker = " (PARALLEL)" if len(wave) > 1 else ""
        print(f"  Wave {i+1}: {len(wave)} nodes{parallel_marker}")
        for node_id in wave:
            node = graph._nodes[node_id]
            deps_str = f" <- {node.deps}" if node.deps else ""
            print(f"    - {node_id}{deps_str}")
    
    # Verify parallelism was discovered correctly
    assert len(waves[0]) == 2  # source1 and source2 in parallel
    assert len(waves[1]) == 3  # proc1, proc2, proc3 in parallel
    assert len(waves[2]) == 2  # combine1 and combine2 in parallel
    assert len(waves[3]) == 1  # final alone


def test_performance_benefit():
    """Demonstrate actual performance benefit of parallel discovery."""
    
    def slow_model(x, delay=0.1):
        """Simulate a model that takes time."""
        time.sleep(delay)
        return x * 2
    
    # Build graph with parallel opportunities
    graph = Graph()
    
    data = graph.add(lambda: 1)
    
    # Add 5 independent models
    models = []
    for i in range(5):
        model = graph.add(
            lambda x, i=i: slow_model(x, delay=0.05),
            deps=[data]
        )
        models.append(model)
    
    # Aggregate results
    final = graph.add(
        lambda *args: sum(args),
        deps=models
    )
    
    # Time execution
    start = time.perf_counter()
    results = graph.run()
    parallel_time = time.perf_counter() - start
    
    print("\n=== Performance Benefit ===\n")
    print(f"5 models with 50ms each:")
    print(f"  Sequential time (theoretical): 250ms")
    print(f"  Parallel time (actual): {parallel_time*1000:.1f}ms")
    print(f"  Speedup: {250/(parallel_time*1000):.1f}x")
    
    # Should be much faster than sequential
    assert parallel_time < 0.15  # Should take ~50ms, not 250ms
    assert results[final] == 10  # 5 models * 2


def test_wave_algorithm_step_by_step():
    """Step through the wave algorithm to show how it works."""
    
    # Simple ensemble graph
    graph = Graph()
    
    # Build graph
    data = graph.add(lambda: 1, deps=[])
    m1 = graph.add(lambda x: x, deps=[data])
    m2 = graph.add(lambda x: x, deps=[data]) 
    judge = graph.add(lambda a, b: a + b, deps=[m1, m2])
    
    print("\n=== Wave Algorithm Step-by-Step ===\n")
    
    # Manually trace through the algorithm
    print("1. Initial in-degrees:")
    in_degree = {}
    for node_id, node in graph._nodes.items():
        in_degree[node_id] = len(node.deps)
        print(f"   {node_id}: {in_degree[node_id]} dependencies")
    
    print("\n2. Find nodes with in-degree 0:")
    wave0 = [nid for nid, degree in in_degree.items() if degree == 0]
    print(f"   Wave 0: {wave0}")
    
    print("\n3. Process wave 0:")
    for node_id in wave0:
        print(f"   Processing {node_id}...")
        for dependent in graph._edges.get(node_id, []):
            in_degree[dependent] -= 1
            print(f"     Reduced {dependent} in-degree to {in_degree[dependent]}")
    
    print("\n4. Find next wave (in-degree now 0):")
    wave1 = [nid for nid, degree in in_degree.items() 
             if degree == 0 and nid not in wave0]
    print(f"   Wave 1: {wave1} <- PARALLEL OPPORTUNITY!")
    
    # And so on...
    print("\nFinal waves:", graph._compute_waves())


if __name__ == "__main__":
    test_ensemble_parallelism_discovery()
    test_complex_dag_parallelism()
    test_performance_benefit()
    test_wave_algorithm_step_by_step()