"""Tests for the simplified Graph implementation."""

import time
import pytest
from ember.xcs.graph import Graph, Node, detect_patterns, vmap


class TestSimpleGraph:
    """Test the new simplified graph."""
    
    def test_basic_construction(self):
        """Test basic graph construction."""
        g = Graph()
        
        # Add some nodes
        n1 = g.add(lambda: 10)
        n2 = g.add(lambda x: x * 2, deps=[n1])
        n3 = g.add(lambda x: x + 5, deps=[n2])
        
        # Execute
        results = g.run()
        
        assert results[n1] == 10
        assert results[n2] == 20
        assert results[n3] == 25
    
    def test_parallel_execution(self):
        """Test parallel branch execution."""
        g = Graph()
        
        # Source
        source = g.add(lambda: {"a": 10, "b": 20})
        
        # Parallel branches
        left = g.add(lambda d: d["a"] * 2, deps=[source])
        right = g.add(lambda d: d["b"] + 5, deps=[source])
        
        # Merge
        final = g.add(lambda a, b: a + b, deps=[left, right])
        
        results = g.run()
        assert results[source] == {"a": 10, "b": 20}
        assert results[left] == 20
        assert results[right] == 25
        assert results[final] == 45
    
    def test_stats(self):
        """Test graph statistics."""
        g = Graph()
        
        # Create diamond pattern
        top = g.add(lambda: 1)
        left = g.add(lambda x: x * 2, deps=[top])
        right = g.add(lambda x: x + 3, deps=[top]) 
        bottom = g.add(lambda a, b: a + b, deps=[left, right])
        
        stats = g.stats
        assert stats['nodes'] == 4
        assert stats['edges'] == 4  # top->left, top->right, left->bottom, right->bottom
        assert stats['waves'] == 3  # top -> left/right -> bottom
        assert stats['parallelism'] == 2  # left and right in parallel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])