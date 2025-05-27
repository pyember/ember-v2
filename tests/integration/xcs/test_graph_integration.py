'''Integration tests for Graph functionality.

End-to-end tests of realistic graph usage patterns.
'''

import pytest
import time
from ember.xcs import Graph

class TestRealWorldPatterns:
    '''Test realistic usage patterns.'''
    
    def test_ensemble_pattern(self):
        '''Ensemble of models with aggregation.'''
        g = Graph()
        
        # Data source
        data = g.add(lambda: [1, 2, 3, 4, 5])
        
        # Ensemble models (can run in parallel)
        model1 = g.add(lambda x: sum(x) * 1.1, deps=(data,))
        model2 = g.add(lambda x: sum(x) * 1.2, deps=(data,))
        model3 = g.add(lambda x: sum(x) * 0.9, deps=(data,))
        
        # Judge/aggregator
        judge = g.add(
            lambda m1, m2, m3: (m1 + m2 + m3) / 3,
            deps=(model1, model2, model3)
        )
        
        result = g.run()
        expected = (15 * 1.1 + 15 * 1.2 + 15 * 0.9) / 3
        assert abs(result[judge] - expected) < 0.01
    
    def test_data_pipeline(self):
        '''Data processing pipeline.'''
        g = Graph()
        
        # Raw data
        raw = g.add(lambda: [1, 2, 3, 4, 5])
        
        # Processing stages
        filtered = g.add(lambda x: [i for i in x if i > 2], deps=(raw,))
        doubled = g.add(lambda x: [i * 2 for i in x], deps=(filtered,))
        summed = g.add(lambda x: sum(x), deps=(doubled,))
        
        result = g.run()
        assert result[filtered] == [3, 4, 5]
        assert result[doubled] == [6, 8, 10]
        assert result[summed] == 24
    
    def test_map_reduce_pattern(self):
        '''Map-reduce style computation.'''
        g = Graph()
        
        # Data source
        data = g.add(lambda: list(range(10)))
        
        # Map phase (parallel processing)
        mapped = []
        for i in range(5):
            node = g.add(
                lambda x, start=i*2: sum(x[start:start+2]), 
                deps=(data,)
            )
            mapped.append(node)
        
        # Reduce phase
        total = g.add(lambda *args: sum(args), deps=mapped)
        
        result = g.run()
        assert result[total] == sum(range(10))  # 45
