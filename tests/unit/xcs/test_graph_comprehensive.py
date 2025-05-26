"""Comprehensive tests for the simplified Graph.

What would Jeff Dean and Sanjay Ghemawat test?
1. Correctness under all conditions
2. Performance characteristics
3. Edge cases and error handling
4. Scalability
"""

import time
import threading
import pytest
from ember.xcs.graph import Graph, Node, detect_patterns, vmap


class TestGraphCorrectness:
    """Test correctness of graph execution."""
    
    def test_empty_graph(self):
        """Empty graph should return empty results."""
        g = Graph()
        assert g.run() == {}
        assert g.stats['nodes'] == 0
    
    def test_single_node(self):
        """Single node execution."""
        g = Graph()
        n = g.add(lambda: 42)
        assert g.run()[n] == 42
    
    def test_linear_chain(self):
        """Linear dependency chain."""
        g = Graph()
        n1 = g.add(lambda: 10)
        n2 = g.add(lambda x: x * 2, deps=[n1])
        n3 = g.add(lambda x: x + 5, deps=[n2])
        n4 = g.add(lambda x: x / 5, deps=[n3])
        
        results = g.run()
        assert results[n1] == 10
        assert results[n2] == 20
        assert results[n3] == 25
        assert results[n4] == 5.0
    
    def test_diamond_pattern(self):
        """Diamond dependency pattern."""
        g = Graph()
        top = g.add(lambda: 100)
        left = g.add(lambda x: x / 2, deps=[top])
        right = g.add(lambda x: x / 4, deps=[top])
        bottom = g.add(lambda a, b: a + b, deps=[left, right])
        
        results = g.run()
        assert results[bottom] == 50 + 25
    
    def test_fan_out_fan_in(self):
        """Fan-out then fan-in pattern."""
        g = Graph()
        source = g.add(lambda: 1)
        
        # Fan out to 10 nodes
        branches = []
        for i in range(10):
            branch = g.add(lambda x, i=i: x * (i + 1), deps=[source])
            branches.append(branch)
        
        # Fan in to sum
        total = g.add(lambda *args: sum(args), deps=branches)
        
        results = g.run()
        assert results[total] == sum(range(1, 11))  # 1+2+3+...+10 = 55
    
    def test_complex_dag(self):
        """Complex DAG with multiple patterns."""
        g = Graph()
        
        # Layer 1: sources
        s1 = g.add(lambda: 1)
        s2 = g.add(lambda: 2)
        s3 = g.add(lambda: 3)
        
        # Layer 2: combinations
        c1 = g.add(lambda a, b: a + b, deps=[s1, s2])
        c2 = g.add(lambda b, c: b * c, deps=[s2, s3])
        
        # Layer 3: more processing
        p1 = g.add(lambda x: x ** 2, deps=[c1])
        p2 = g.add(lambda x: x - 1, deps=[c2])
        
        # Layer 4: final
        final = g.add(lambda a, b: max(a, b), deps=[p1, p2])
        
        results = g.run()
        assert results[c1] == 3
        assert results[c2] == 6
        assert results[p1] == 9
        assert results[p2] == 5
        assert results[final] == 9


class TestFunctionSignatures:
    """Test various function signature handling."""
    
    def test_no_args(self):
        """Functions with no arguments."""
        g = Graph()
        n = g.add(lambda: "hello")
        assert g.run()[n] == "hello"
    
    def test_single_positional(self):
        """Single positional argument."""
        g = Graph()
        source = g.add(lambda: 42)
        double = g.add(lambda x: x * 2, deps=[source])
        assert g.run()[double] == 84
    
    def test_multiple_positional(self):
        """Multiple positional arguments."""
        g = Graph()
        a = g.add(lambda: 10)
        b = g.add(lambda: 20)
        c = g.add(lambda: 30)
        sum_node = g.add(lambda x, y, z: x + y + z, deps=[a, b, c])
        assert g.run()[sum_node] == 60
    
    def test_varargs(self):
        """Functions with *args."""
        g = Graph()
        nums = [g.add(lambda i=i: i) for i in range(5)]
        product = g.add(lambda *args: sum(x**2 for x in args), deps=nums)
        assert g.run()[product] == 0 + 1 + 4 + 9 + 16
    
    def test_kwargs_from_dict(self):
        """Functions expecting keyword arguments from dict result."""
        g = Graph()
        config = g.add(lambda: {"multiplier": 3, "offset": 10})
        
        def process(multiplier, offset):
            return multiplier * 5 + offset
        
        result = g.add(process, deps=[config])
        assert g.run()[result] == 15 + 10
    
    def test_mixed_dependencies(self):
        """Mix of different dependency patterns."""
        g = Graph()
        
        # Different sources
        num = g.add(lambda: 42)
        config = g.add(lambda: {"scale": 2})
        flag = g.add(lambda: True)
        
        # Function that uses all
        def compute(n, cfg, flag):
            if flag:
                return n * cfg["scale"]
            return n
        
        result = g.add(compute, deps=[num, config, flag])
        assert g.run()[result] == 84


class TestErrorHandling:
    """Test error conditions and edge cases."""
    
    def test_circular_dependency_detection(self):
        """Circular dependencies should be caught."""
        g = Graph()
        a = g.add(lambda: 1)
        
        # Manually create cycle (API prevents this normally)
        g._nodes[a] = Node(a, lambda: 1, deps=(a))
        
        with pytest.raises(ValueError, match="cycle"):
            g.run()
    
    def test_missing_dependency(self):
        """Missing dependencies should fail gracefully."""
        g = Graph()
        with pytest.raises(ValueError, match="Unknown dependency"):
            g.add(lambda x: x + 1, deps=["nonexistent"])
    
    def test_non_callable(self):
        """Non-callable should be rejected."""
        with pytest.raises(TypeError):
            Node("bad", "not a function")
    
    def test_exception_propagation(self):
        """Exceptions in nodes should propagate."""
        g = Graph()
        
        def failing_func():
            raise ValueError("Intentional error")
        
        n = g.add(failing_func)
        
        with pytest.raises(ValueError, match="Intentional error"):
            g.run()
    
    def test_exception_in_parallel(self):
        """Exceptions in parallel execution should propagate."""
        g = Graph()
        
        # Create parallel branches
        ok1 = g.add(lambda: 1)
        ok2 = g.add(lambda: 2)
        bad = g.add(lambda: 1/0)  # Will raise ZeroDivisionError
        
        with pytest.raises(ZeroDivisionError):
            g.run()


class TestPerformance:
    """Test performance characteristics."""
    
    def test_lazy_wave_computation(self):
        """Waves should be computed lazily."""
        g = Graph()
        
        # Add many nodes
        for i in range(100):
            g.add(lambda i=i: i)
        
        # Waves not computed yet
        assert g._waves is None
        
        # Access stats triggers computation
        stats = g.stats
        assert g._waves is not None
        assert stats['nodes'] == 100
    
    def test_parallel_speedup(self):
        """Parallel execution should be faster for CPU-bound work."""
        def cpu_work(n=100000):
            # Simulate CPU-bound work
            total = 0
            for i in range(n):
                total += i * i
            return total
        
        # Sequential graph
        seq_graph = Graph()
        prev = None
        for i in range(4):
            prev = seq_graph.add(cpu_work, deps=[prev] if prev else [])
        
        # Parallel graph
        par_graph = Graph()
        nodes = [par_graph.add(cpu_work) for _ in range(4)]
        par_graph.add(lambda *args: sum(args), deps=nodes)
        
        # Time sequential
        seq_start = time.perf_counter()
        seq_graph.run()
        seq_time = time.perf_counter() - seq_start
        
        # Time parallel
        par_start = time.perf_counter()
        par_graph.run()
        par_time = time.perf_counter() - par_start
        
        # Parallel should be faster
        assert par_time < seq_time
        speedup = seq_time / par_time
        assert speedup > 1.5  # At least 1.5x speedup
    
    def test_no_overhead_for_sequential(self):
        """Sequential graphs should have minimal overhead."""
        g = Graph()
        
        # Linear chain
        prev = None
        for i in range(10):
            prev = g.add(lambda x=i: x + 1, deps=[prev] if prev else [])
        
        # Should detect as sequential
        stats = g.stats
        assert stats['parallelism'] == 1
        
        # Execution should be fast
        start = time.perf_counter()
        g.run()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.01  # Less than 10ms


class TestPatternDetection:
    """Test pattern detection for optimization."""
    
    def test_map_pattern(self):
        """Detect map patterns."""
        g = Graph()
        transform = lambda x: x * 2
        
        # Same function, independent execution
        nodes = []
        for i in range(5):
            n = g.add(transform)
            nodes.append(n)
        
        patterns = detect_patterns(g)
        assert 'map' in patterns
        assert len(patterns['map']) == 1
        assert len(patterns['map'][0]) == 5
    
    def test_reduce_pattern(self):
        """Detect reduce patterns."""
        g = Graph()
        
        # Many to one
        sources = [g.add(lambda i=i: i) for i in range(10)]
        reducer = g.add(lambda *args: sum(args), deps=sources)
        
        patterns = detect_patterns(g)
        assert 'reduce' in patterns
        assert any(reducer in p for p in patterns['reduce'])
    
    def test_no_false_patterns(self):
        """Don't detect patterns that aren't there."""
        g = Graph()
        
        # Different functions
        g.add(lambda: 1)
        g.add(lambda: 2)
        g.add(lambda x: x + 1)
        
        patterns = detect_patterns(g)
        # No map pattern - different functions
        assert not patterns.get('map', [])


class TestTransformations:
    """Test functional transformations."""
    
    def test_vmap_basic(self):
        """Basic vmap functionality."""
        def square(x):
            return x ** 2
        
        vsquare = vmap(square)
        result = vsquare([1, 2, 3, 4, 5])
        assert result == [1, 4, 9, 16, 25]
    
    def test_vmap_parallelism(self):
        """vmap should parallelize."""
        def slow_func(x):
            time.sleep(0.01)  # 10ms per item
            return x * 2
        
        vfunc = vmap(slow_func)
        
        start = time.perf_counter()
        result = vfunc(list(range(10)))
        elapsed = time.perf_counter() - start
        
        # Should be much faster than sequential (100ms)
        assert elapsed < 0.05  # Less than 50ms
        assert result == [i * 2 for i in range(10)]


class TestConcurrency:
    """Test thread safety and concurrent access."""
    
    def test_concurrent_graph_creation(self):
        """Multiple threads creating graphs."""
        graphs = [None] * 10
        
        def create_graph(idx):
            g = Graph()
            for i in range(10):
                g.add(lambda i=i, idx=idx: i * idx)
            graphs[idx] = g
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_graph, args=(i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All graphs should be created successfully
        assert all(g is not None for g in graphs)
        assert all(g.stats['nodes'] == 10 for g in graphs)
    
    def test_concurrent_execution(self):
        """Multiple threads executing same graph."""
        g = Graph()
        
        # Build a simple graph
        a = g.add(lambda: 10)
        b = g.add(lambda x: x * 2, deps=[a])
        c = g.add(lambda x: x + 5, deps=[b])
        
        results = [None] * 10
        
        def run_graph(idx):
            results[idx] = g.run()[c]
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=run_graph, args=(i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All executions should produce same result
        assert all(r == 25 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])