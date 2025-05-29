"""Tests for the simplified Graph implementation.

Clean, focused, fast. No fluff.
"""

import time
import pytest
from ember.xcs import Graph, Node, detect_patterns, simple_vmap as vmap


class TestNode:
    """Node is just data."""
    
    def test_immutable(self):
        node = Node("test", lambda: 42)
        assert node.id == "test"
        assert node.func() == 42
        assert node.deps == ()
        
        # Should be hashable (frozen)
        assert hash(node)
    
    def test_validates_callable(self):
        with pytest.raises(TypeError):
            Node("bad", "not callable")


class TestGraph:
    """Graph does one thing well: execute computations."""
    
    def test_simple_pipeline(self):
        g = Graph()
        a = g.add(lambda: 10)
        b = g.add(lambda x: x * 2, deps=[a])
        c = g.add(lambda x: x + 5, deps=[b])
        
        result = g.run()
        assert result[a] == 10
        assert result[b] == 20
        assert result[c] == 25
    
    def test_parallel_branches(self):
        g = Graph()
        source = g.add(lambda: {"x": 10, "y": 20})
        
        # Two independent branches
        left = g.add(lambda d: d["x"] * 2, deps=[source])
        right = g.add(lambda d: d["y"] + 5, deps=[source])
        
        # Merge
        final = g.add(lambda a, b: a + b, deps=[left, right])
        
        result = g.run()
        assert result[final] == 20 + 25
    
    def test_detects_cycles(self):
        g = Graph()
        a = g.add(lambda: 1)
        b = g.add(lambda x: x + 1, deps=[a])
        
        # Manually create cycle (normally impossible through API)
        g._nodes[a] = Node(a, lambda: 1, deps=(b))
        
        with pytest.raises(ValueError, match="cycle"):
            g.run()
    
    def test_unknown_dependency(self):
        g = Graph()
        with pytest.raises(ValueError, match="Unknown dependency"):
            g.add(lambda: 1, deps=["fake"])
    
    def test_stats(self):
        g = Graph()
        a = g.add(lambda: 1)
        b = g.add(lambda: 2)
        c = g.add(lambda x, y: x + y, deps=[a, b])
        
        stats = g.stats
        assert stats['nodes'] == 3
        assert stats['edges'] == 2
        assert stats['waves'] == 2
        assert stats['parallelism'] == 2  # a and b in parallel
        assert stats['critical_path'] == 2


class TestExecution:
    """Execution correctness and performance."""
    
    def test_no_args_function(self):
        g = Graph()
        n = g.add(lambda: 42)
        assert g.run()[n] == 42
    
    def test_varargs_function(self):
        g = Graph()
        a = g.add(lambda: 1)
        b = g.add(lambda: 2)
        c = g.add(lambda: 3)
        s = g.add(lambda *args: sum(args), deps=[a, b, c])
        
        assert g.run()[s] == 6
    
    def test_kwargs_function(self):
        g = Graph()
        n = g.add(lambda x=10, y=20: x + y)
        assert g.run({"x": 5, "y": 15})[n] == 20
    
    def test_old_style_inputs(self):
        def old_style(inputs):
            return inputs["value"] * 2
        
        g = Graph()
        n = g.add(old_style)
        assert g.run({"value": 21})[n] == 42
    
    def test_performance_overhead(self):
        """Graph overhead should be minimal."""
        # Direct execution
        def computation():
            return sum(i * 2 for i in range(100))
        
        direct_time = min(timeit(computation) for _ in range(100))
        
        # Graph execution
        g = Graph()
        nodes = [g.add(lambda x=i: x * 2) for i in range(100)]
        final = g.add(lambda *args: sum(args), deps=nodes)
        
        graph_time = min(timeit(lambda: g.run()[final]) for _ in range(100))
        
        overhead = graph_time / direct_time
        assert overhead < 50  # Reasonable for 100 nodes


class TestPatterns:
    """Pattern detection for optimization."""
    
    def test_map_pattern(self):
        g = Graph()
        func = lambda x: x * 2
        
        # Same function, different inputs
        nodes = [g.add(func) for _ in range(5)]
        
        patterns = detect_patterns(g)
        assert len(patterns.get('map', [])) == 1
        assert len(patterns['map'][0]) == 5
    
    def test_reduce_pattern(self):
        g = Graph()
        sources = [g.add(lambda: i) for i in range(5)]
        reducer = g.add(lambda *args: sum(args), deps=sources)
        
        patterns = detect_patterns(g)
        assert 'reduce' in patterns
        assert reducer in patterns['reduce'][0]


class TestTransforms:
    """Functional transformations."""
    
    def test_vmap(self):
        def double(x):
            return x * 2
        
        vdouble = vmap(double)
        result = vdouble([1, 2, 3, 4, 5])
        
        assert result == [2, 4, 6, 8, 10]


def timeit(func, iterations=1):
    """Simple timing utility."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return (time.perf_counter() - start) / iterations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])