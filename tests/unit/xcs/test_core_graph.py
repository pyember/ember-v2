'''Core Graph functionality tests.

Clean, focused tests for the simplified Graph API.
'''

import pytest
import time
from ember.xcs import Graph, Node

class TestGraph:
    '''Test Graph construction and execution.'''
    
    def test_empty_graph(self):
        '''Empty graph returns empty results.'''
        g = Graph()
        assert g.run() == {}
    
    def test_single_node(self):
        '''Single node execution.'''
        g = Graph()
        n = g.add(lambda: 42)
        result = g.run()
        assert result[n] == 42
    
    def test_linear_pipeline(self):
        '''Linear dependency chain.'''
        g = Graph()
        n1 = g.add(lambda: 10)
        n2 = g.add(lambda x: x * 2, deps=(n1,))
        n3 = g.add(lambda x: x + 5, deps=(n2,))
        
        result = g.run()
        assert result[n1] == 10
        assert result[n2] == 20
        assert result[n3] == 25
    
    def test_parallel_branches(self):
        '''Independent parallel branches.'''
        g = Graph()
        source = g.add(lambda: {'a': 10, 'b': 20})
        left = g.add(lambda d: d['a'] * 2, deps=(source,))
        right = g.add(lambda d: d['b'] + 5, deps=(source,))
        merge = g.add(lambda a, b: a + b, deps=(left, right))
        
        result = g.run()
        assert result[left] == 20
        assert result[right] == 25  
        assert result[merge] == 45
    
    def test_diamond_pattern(self):
        '''Diamond dependency pattern.'''
        g = Graph()
        top = g.add(lambda: 100)
        left = g.add(lambda x: x / 2, deps=(top,))
        right = g.add(lambda x: x / 4, deps=(top,))
        bottom = g.add(lambda a, b: a + b, deps=(left, right))
        
        result = g.run()
        assert result[bottom] == 75.0
    
    def test_error_handling(self):
        '''Test error propagation.'''
        g = Graph()
        
        def failing_func():
            raise ValueError("Test error")
        
        n = g.add(failing_func)
        with pytest.raises(ValueError, match="Test error"):
            g.run()
    
    def test_unknown_dependency(self):
        '''Unknown dependencies are rejected.'''
        g = Graph()
        with pytest.raises(ValueError, match="Unknown dependency"):
            g.add(lambda: 1, deps=("nonexistent",))

class TestGraphStats:
    '''Test graph analysis and statistics.'''
    
    def test_basic_stats(self):
        '''Basic graph statistics.'''
        g = Graph()
        a = g.add(lambda: 1)
        b = g.add(lambda: 2)
        c = g.add(lambda x, y: x + y, deps=(a, b))
        
        stats = g.stats
        assert stats['nodes'] == 3
        assert stats['edges'] == 2
        assert stats['waves'] == 2
        assert stats['parallelism'] == 2
    
    def test_sequential_detection(self):
        '''Sequential patterns detected correctly.'''
        g = Graph()
        n1 = g.add(lambda: 1)
        n2 = g.add(lambda x: x + 1, deps=(n1,))
        n3 = g.add(lambda x: x * 2, deps=(n2,))
        
        stats = g.stats
        assert stats['waves'] == 3
        assert stats['parallelism'] == 1

class TestParallelExecution:
    '''Test parallel execution performance.'''
    
    def test_io_parallelism(self):
        '''I/O bound operations benefit from parallelism.'''
        g = Graph()
        
        def slow_io(delay=0.1):
            time.sleep(delay)
            return delay
        
        # Create 3 independent I/O operations
        nodes = []
        for i in range(3):
            n = g.add(lambda d=0.1: slow_io(d))
            nodes.append(n)
        
        # Should execute in parallel, not sequential
        start = time.time()
        result = g.run()
        elapsed = time.time() - start
        
        # Parallel: ~0.1s, Sequential: ~0.3s
        assert elapsed < 0.2  # Much faster than sequential
        assert all(result[n] == 0.1 for n in nodes)
