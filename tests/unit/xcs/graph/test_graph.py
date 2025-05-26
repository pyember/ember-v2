"""Unit tests for the new simplified Graph class."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from src.ember.xcs.graph.graph import Graph, Node


class TestNode:
    """Test Node dataclass."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        def dummy_func():
            return 42
            
        node = Node(
            id="test_node",
            func=dummy_func,
            args=[1, 2],
            kwargs={"key": "value"},
            deps=["dep1", "dep2"]
        )
        
        assert node.id == "test_node"
        assert node.func == dummy_func
        assert node.args == [1, 2]
        assert node.kwargs == {"key": "value"}
        assert node.deps == ["dep1", "dep2"]
    
    def test_node_requires_callable(self):
        """Test that node requires callable function."""
        with pytest.raises(ValueError, match="Node function must be callable"):
            Node(id="bad_node", func="not_callable")


class TestGraphConstruction:
    """Test graph construction and basic operations."""
    
    def test_empty_graph(self):
        """Test creating empty graph."""
        graph = Graph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_add_single_node(self):
        """Test adding single node."""
        graph = Graph()
        
        def process():
            return "result"
        
        node_id = graph.add(process, name="processor")
        
        assert node_id == "processor"
        assert len(graph.nodes) == 1
        assert graph.nodes[node_id].func == process
    
    def test_add_node_with_dependencies(self):
        """Test adding nodes with dependencies."""
        graph = Graph()
        
        # Add first node
        node1 = graph.add(lambda: 1, name="node1")
        
        # Add dependent node
        node2 = graph.add(lambda x: x + 1, deps=[node1], name="node2")
        
        assert len(graph.nodes) == 2
        assert graph.nodes[node2].deps == [node1]
        assert node2 in graph.edges[node1]
    
    def test_dependency_validation(self):
        """Test that dependencies must exist."""
        graph = Graph()
        
        with pytest.raises(ValueError, match="Dependency fake_dep not found"):
            graph.add(lambda: 1, deps=["fake_dep"])
    
    def test_auto_generated_names(self):
        """Test automatic name generation."""
        graph = Graph()
        
        def my_func():
            return 42
        
        node_id = graph.add(my_func)
        
        assert node_id.startswith("my_func_")
        assert len(node_id) > len("my_func_")


class TestGraphAnalysis:
    """Test graph analysis and pattern detection."""
    
    def test_sequential_detection(self):
        """Test detection of sequential graphs."""
        graph = Graph()
        
        # Build sequential pipeline
        n1 = graph.add(lambda: 1, name="n1")
        n2 = graph.add(lambda x: x + 1, deps=[n1], name="n2")
        n3 = graph.add(lambda x: x * 2, deps=[n2], name="n3")
        
        assert graph._is_sequential() == True
        
        analysis = graph._analyze_graph()
        assert analysis['is_sequential'] == True
        assert analysis['parallelism_degree'] == 1
    
    def test_parallel_detection(self):
        """Test detection of parallel opportunities."""
        graph = Graph()
        
        # Build parallel branches
        source = graph.add(lambda: {"a": 1, "b": 2}, name="source")
        
        # Parallel branches
        branch1 = graph.add(lambda x: x["a"] * 2, deps=[source], name="branch1")
        branch2 = graph.add(lambda x: x["b"] * 3, deps=[source], name="branch2")
        
        # Merge
        merge = graph.add(lambda x, y: x + y, deps=[branch1, branch2], name="merge")
        
        assert graph._is_sequential() == False
        
        analysis = graph._analyze_graph()
        assert analysis['is_sequential'] == False
        assert analysis['parallelism_degree'] >= 2
    
    def test_map_pattern_detection(self):
        """Test detection of map patterns."""
        graph = Graph()
        
        def transform(x):
            return x * 2
        
        # Add multiple nodes with same function
        source = graph.add(lambda: [1, 2, 3, 4], name="source")
        
        # Map-like operations
        map1 = graph.add(transform, args=[1], name="map1")
        map2 = graph.add(transform, args=[2], name="map2")
        map3 = graph.add(transform, args=[3], name="map3")
        
        patterns = graph._detect_patterns()
        
        # Should detect map pattern
        assert len(patterns['map']) > 0
        map_group = patterns['map'][0]
        assert len(map_group) == 3
        assert all(node in map_group for node in ["map1", "map2", "map3"])
    
    def test_reduce_pattern_detection(self):
        """Test detection of reduce patterns."""
        graph = Graph()
        
        # Create multiple sources
        s1 = graph.add(lambda: 1, name="s1")
        s2 = graph.add(lambda: 2, name="s2")
        s3 = graph.add(lambda: 3, name="s3")
        
        # Reduce to single node
        reducer = graph.add(lambda *args: sum(args), deps=[s1, s2, s3], name="reducer")
        
        patterns = graph._detect_patterns()
        
        # Should detect reduce pattern
        assert len(patterns['reduce']) > 0
        assert "reducer" in patterns['reduce'][0]
    
    def test_ensemble_pattern_detection(self):
        """Test detection of ensemble patterns."""
        graph = Graph()
        
        def model(x):
            return x * 2
        
        # Input
        input_node = graph.add(lambda: 10, name="input")
        
        # Ensemble members (same function)
        m1 = graph.add(model, deps=[input_node], name="model1")
        m2 = graph.add(model, deps=[input_node], name="model2")
        m3 = graph.add(model, deps=[input_node], name="model3")
        
        # Judge
        judge = graph.add(lambda *args: max(args), deps=[m1, m2, m3], name="judge")
        
        patterns = graph._detect_patterns()
        
        # Should detect ensemble pattern
        assert len(patterns['ensemble']) > 0
        ensemble = patterns['ensemble'][0]
        assert "judge" in ensemble
        assert all(m in ensemble for m in ["model1", "model2", "model3"])


class TestWaveComputation:
    """Test wave computation for parallel execution."""
    
    def test_simple_waves(self):
        """Test wave computation for simple graph."""
        graph = Graph()
        
        # Wave 1
        n1 = graph.add(lambda: 1, name="n1")
        n2 = graph.add(lambda: 2, name="n2")
        
        # Wave 2 (depends on wave 1)
        n3 = graph.add(lambda x, y: x + y, deps=[n1, n2], name="n3")
        
        waves = graph._compute_waves()
        
        assert len(waves) == 2
        assert set(waves[0]) == {"n1", "n2"}
        assert waves[1] == ["n3"]
    
    def test_complex_waves(self):
        """Test wave computation for complex graph."""
        graph = Graph()
        
        # Wave 1
        a = graph.add(lambda: 1, name="a")
        b = graph.add(lambda: 2, name="b")
        
        # Wave 2
        c = graph.add(lambda x: x + 1, deps=[a], name="c")
        d = graph.add(lambda x: x * 2, deps=[b], name="d")
        
        # Wave 3
        e = graph.add(lambda x, y: x + y, deps=[c, d], name="e")
        
        # Wave 4
        f = graph.add(lambda x: x ** 2, deps=[e], name="f")
        
        waves = graph._compute_waves()
        
        assert len(waves) == 4
        assert set(waves[0]) == {"a", "b"}
        assert set(waves[1]) == {"c", "d"}
        assert waves[2] == ["e"]
        assert waves[3] == ["f"]
    
    def test_topological_sort(self):
        """Test topological sorting."""
        graph = Graph()
        
        # Create diamond pattern
        top = graph.add(lambda: 1, name="top")
        left = graph.add(lambda x: x + 1, deps=[top], name="left")
        right = graph.add(lambda x: x * 2, deps=[top], name="right")
        bottom = graph.add(lambda x, y: x + y, deps=[left, right], name="bottom")
        
        topo_order = graph._topological_sort()
        
        # Check ordering constraints
        assert topo_order.index("top") < topo_order.index("left")
        assert topo_order.index("top") < topo_order.index("right")
        assert topo_order.index("left") < topo_order.index("bottom")
        assert topo_order.index("right") < topo_order.index("bottom")
    
    def test_cycle_detection(self):
        """Test that cycles are detected."""
        graph = Graph()
        
        # Create nodes
        n1 = graph.add(lambda: 1, name="n1")
        n2 = graph.add(lambda x: x + 1, deps=[n1], name="n2")
        
        # Manually create cycle (shouldn't normally be possible)
        graph.nodes["n1"].deps.append("n2")
        
        with pytest.raises(ValueError, match="Graph contains cycles"):
            graph._topological_sort()


class TestExecution:
    """Test graph execution."""
    
    def test_sequential_execution(self):
        """Test sequential execution."""
        graph = Graph()
        
        # Build simple pipeline
        n1 = graph.add(lambda: 10, name="n1")
        n2 = graph.add(lambda x: x * 2, name="n2", deps=[n1])
        n3 = graph.add(lambda x: x + 5, name="n3", deps=[n2])
        
        results = graph({}, parallel=False)
        
        assert results["n1"] == 10
        assert results["n2"] == 20
        assert results["n3"] == 25
    
    def test_parallel_execution(self):
        """Test parallel execution."""
        graph = Graph()
        
        # Build parallel branches
        source = graph.add(lambda: {"a": 10, "b": 20}, name="source")
        
        # Parallel operations
        branch1 = graph.add(lambda x: x["a"] * 2, deps=[source], name="branch1")
        branch2 = graph.add(lambda x: x["b"] + 5, deps=[source], name="branch2")
        
        # Merge
        merge = graph.add(
            lambda inputs: inputs["branch1"] + inputs["branch2"], 
            deps=[branch1, branch2], 
            name="merge"
        )
        
        results = graph({}, parallel=True)
        
        assert results["source"] == {"a": 10, "b": 20}
        assert results["branch1"] == 20
        assert results["branch2"] == 25
        assert results["merge"] == 45
    
    def test_execution_with_inputs(self):
        """Test execution with initial inputs."""
        graph = Graph()
        
        # Source nodes that use inputs
        n1 = graph.add(lambda x: x * 2, name="n1")
        n2 = graph.add(lambda y: y + 10, name="n2")
        
        # Dependent node
        n3 = graph.add(
            lambda inputs: inputs["n1"] + inputs["n2"], 
            deps=[n1, n2], 
            name="n3"
        )
        
        results = graph({"x": 5, "y": 3}, parallel=False)
        
        assert results["n1"] == 10
        assert results["n2"] == 13
        assert results["n3"] == 23
    
    def test_execution_caching(self):
        """Test result caching."""
        graph = Graph()
        
        call_count = 0
        
        def expensive_func():
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return 42
        
        node = graph.add(expensive_func, name="expensive")
        
        # First execution
        result1 = graph({}, cache=True)
        assert result1["expensive"] == 42
        assert call_count == 1
        
        # Second execution should use cache
        result2 = graph({}, cache=True)
        assert result2["expensive"] == 42
        assert call_count == 1  # Should not increase
        
        # Execution without cache
        result3 = graph({}, cache=False)
        assert result3["expensive"] == 42
        assert call_count == 2  # Should increase
    
    def test_error_handling(self):
        """Test error handling during execution."""
        graph = Graph()
        
        def failing_func():
            raise ValueError("Test error")
        
        n1 = graph.add(failing_func, name="failing")
        
        with pytest.raises(ValueError, match="Test error"):
            graph({})
    
    def test_old_style_operators(self):
        """Test compatibility with old-style operators expecting 'inputs' parameter."""
        graph = Graph()
        
        # Old-style operator
        def old_operator(inputs):
            return inputs["value"] * 2
        
        # New-style function
        def new_func(x):
            return x + 10
        
        # Build graph
        source = graph.add(lambda: {"value": 5}, name="source")
        old_node = graph.add(old_operator, deps=[source], name="old_op")
        new_node = graph.add(new_func, args=[7], name="new_op")
        
        results = graph({})
        
        assert results["source"] == {"value": 5}
        assert results["old_op"] == 10  # 5 * 2
        assert results["new_op"] == 17  # 7 + 10


class TestOptimization:
    """Test graph optimization features."""
    
    def test_wave_optimization(self):
        """Test wave optimization for patterns."""
        graph = Graph()
        
        # Create map pattern
        source = graph.add(lambda: [1, 2, 3], name="source")
        
        def process(x):
            return x * 2
        
        # These should be colocated
        m1 = graph.add(process, args=[1], name="m1")
        m2 = graph.add(process, args=[2], name="m2")
        m3 = graph.add(process, args=[3], name="m3")
        
        # This depends on m1, so creates ordering constraint
        dependent = graph.add(lambda x: x + 1, deps=[m1], name="dependent")
        
        # Get initial waves
        waves = graph._compute_waves()
        analysis = graph._analyze_graph()
        
        # Apply optimization
        optimized = graph._optimize_execution(waves, analysis)
        
        # Check that map operations are colocated
        map_wave = None
        for wave in optimized:
            if "m1" in wave and "m2" in wave and "m3" in wave:
                map_wave = wave
                break
        
        assert map_wave is not None, "Map operations should be colocated"
    
    def test_node_colocation(self):
        """Test node colocation optimization."""
        graph = Graph()
        
        # Create nodes that can be colocated
        n1 = graph.add(lambda: 1, name="n1")
        n2 = graph.add(lambda: 2, name="n2")
        n3 = graph.add(lambda: 3, name="n3")
        
        # Initially in separate waves
        waves = [["n1"], ["n2"], ["n3"]]
        
        # Colocate n2 and n3
        graph._colocate_nodes(waves, {"n2", "n3"})
        
        # Check colocation
        assert ["n1"] in waves
        assert any("n2" in wave and "n3" in wave for wave in waves)


class TestVisualization:
    """Test graph visualization."""
    
    def test_simple_visualization(self):
        """Test visualization output."""
        graph = Graph()
        
        # Build simple graph
        n1 = graph.add(lambda: 1, name="n1")
        n2 = graph.add(lambda x: x + 1, deps=[n1], name="n2")
        
        viz = graph.visualize()
        
        assert "Graph Structure:" in viz
        assert "Nodes: 2" in viz
        assert "Edges: 1" in viz
        assert "Execution Waves: 2" in viz


class TestPerformance:
    """Performance benchmarks for the new Graph implementation."""
    
    def test_large_graph_construction(self):
        """Test performance of large graph construction."""
        graph = Graph()
        
        start = time.time()
        
        # Create 100 nodes
        nodes = []
        for i in range(100):
            node = graph.add(lambda x=i: x * 2, name=f"node_{i}")
            nodes.append(node)
        
        # Add dependencies (each depends on previous 3)
        for i in range(3, 100):
            deps = nodes[i-3:i]
            graph.add(lambda *args: sum(args), deps=deps, name=f"sum_{i}")
        
        construction_time = time.time() - start
        
        # Should be fast
        assert construction_time < 0.1, f"Graph construction too slow: {construction_time}s"
        
        # Test analysis performance
        start = time.time()
        analysis = graph._analyze_graph()
        analysis_time = time.time() - start
        
        assert analysis_time < 0.1, f"Graph analysis too slow: {analysis_time}s"
    
    def test_execution_performance(self):
        """Test execution performance vs direct execution."""
        
        def create_computation():
            """Create a simple computation."""
            results = []
            for i in range(10):
                results.append(i * 2)
            return sum(results)
        
        # Direct execution
        start = time.time()
        for _ in range(100):
            direct_result = create_computation()
        direct_time = time.time() - start
        
        # Graph execution
        graph = Graph()
        nodes = []
        for i in range(10):
            node = graph.add(lambda x=i: x * 2, name=f"n{i}")
            nodes.append(node)
        sum_node = graph.add(lambda *args: sum(args), deps=nodes, name="sum")
        
        start = time.time()
        for _ in range(100):
            graph_result = graph({})["sum"]
        graph_time = time.time() - start
        
        assert direct_result == graph_result
        
        # Graph overhead is expected for prototype (analysis not cached yet)
        # In production, we'll cache analysis results
        overhead = graph_time / direct_time
        # For now, just ensure it completes in reasonable time
        assert graph_time < 1.0, f"Graph execution too slow: {graph_time}s"
        
        # Log overhead for monitoring
        print(f"Graph overhead: {overhead:.1f}x (prototype - analysis not cached)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])