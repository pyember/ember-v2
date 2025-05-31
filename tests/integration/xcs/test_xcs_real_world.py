"""Real-world integration tests for XCS.

Tests realistic usage patterns through the public API only.
No internal implementation details.
"""

import time
import pytest
from ember.api import xcs, models, data
from ember.api.operators import Operator, Specification


class TestRealWorldJITPatterns:
    """Test @jit in realistic scenarios."""
    
    def test_ensemble_reasoning_pattern(self):
        """Test ensemble pattern with automatic parallelization."""
        @xcs.jit
        class EnsembleReasoner(Operator):
            specification = Specification()
            
            def forward(self, *, inputs):
                query = inputs.get("query", "")
                
                # These operations should parallelize automatically
                results = []
                for i in range(3):
                    # Simulate model calls
                    time.sleep(0.01)  # Simulate I/O latency
                    results.append(f"Model {i}: Answer to '{query}'")
                
                # Aggregate results
                return {
                    "answers": results,
                    "consensus": f"Consensus on '{query}'"
                }
        
        reasoner = EnsembleReasoner()
        
        start = time.time()
        result = reasoner(query="What is consciousness?")
        elapsed = time.time() - start
        
        # Should run in parallel (~0.01s) not sequential (~0.03s)
        assert elapsed < 0.02  # Allow some overhead
        assert len(result["answers"]) == 3
        assert "consensus" in result
    
    def test_data_processing_pipeline(self):
        """Test data pipeline with @jit optimization."""
        @xcs.jit
        class DataProcessor(Operator):
            specification = Specification()
            
            def forward(self, *, inputs):
                items = inputs.get("items", [])
                
                # Transform phase (parallelizable)
                transformed = []
                for item in items:
                    # Simulate processing
                    transformed.append({
                        "original": item,
                        "processed": item * 2,
                        "metadata": {"length": len(str(item))}
                    })
                
                # Aggregate phase
                total = sum(t["processed"] for t in transformed)
                
                return {
                    "transformed": transformed,
                    "total": total,
                    "count": len(items)
                }
        
        processor = DataProcessor()
        result = processor(items=[1, 2, 3, 4, 5])
        
        assert result["count"] == 5
        assert result["total"] == 30  # (1+2+3+4+5) * 2
        assert len(result["transformed"]) == 5
    
    def test_conditional_execution_optimization(self):
        """Test that @jit handles conditional logic efficiently."""
        @xcs.jit
        class ConditionalProcessor(Operator):
            specification = Specification()
            
            def forward(self, *, inputs):
                mode = inputs.get("mode", "fast")
                value = inputs.get("value", 0)
                
                if mode == "fast":
                    return {"result": value * 2}
                elif mode == "accurate":
                    # Simulate more complex processing
                    time.sleep(0.01)
                    return {"result": value * 2, "confidence": 0.99}
                else:
                    return {"result": 0, "error": "Unknown mode"}
        
        processor = ConditionalProcessor()
        
        # Fast path
        fast_result = processor(mode="fast", value=10)
        assert fast_result == {"result": 20}
        
        # Accurate path
        accurate_result = processor(mode="accurate", value=10)
        assert accurate_result["result"] == 20
        assert accurate_result["confidence"] == 0.99


class TestVMapRealWorld:
    """Test vmap in realistic scenarios."""
    
    def test_batch_text_processing(self):
        """Test batch processing of text data."""
        def analyze_text(text):
            return {
                "length": len(text),
                "words": len(text.split()),
                "uppercase": text.upper()
            }
        
        batch_analyze = xcs.vmap(analyze_text)
        
        texts = [
            "Hello world",
            "The quick brown fox",
            "Python is awesome"
        ]
        
        results = batch_analyze(texts)
        
        assert len(results) == 3
        assert results[0]["words"] == 2
        assert results[1]["words"] == 4
        assert results[2]["uppercase"] == "PYTHON IS AWESOME"
    
    def test_batch_numerical_computation(self):
        """Test batch numerical operations."""
        def compute_stats(numbers):
            return {
                "sum": sum(numbers),
                "mean": sum(numbers) / len(numbers),
                "max": max(numbers),
                "min": min(numbers)
            }
        
        batch_stats = xcs.vmap(compute_stats)
        
        datasets = [
            [1, 2, 3, 4, 5],
            [10, 20, 30],
            [100, 200, 300, 400]
        ]
        
        results = batch_stats(datasets)
        
        assert results[0]["mean"] == 3.0
        assert results[1]["sum"] == 60
        assert results[2]["max"] == 400
    
    def test_vmap_with_multiple_arguments(self):
        """Test vmap with multiple input arguments."""
        def combine(x, y, z):
            return x * y + z
        
        batch_combine = xcs.vmap(combine)
        
        xs = [1, 2, 3]
        ys = [10, 20, 30]
        zs = [100, 200, 300]
        
        results = batch_combine(xs, ys, zs)
        assert results == [110, 240, 390]


class TestPerformanceCharacteristics:
    """Test that optimizations provide real benefits."""
    
    def test_jit_provides_measurable_speedup(self):
        """Verify @jit actually improves performance."""
        def slow_unoptimized(*, inputs):
            results = []
            for i in range(inputs.get("count", 5)):
                time.sleep(0.005)  # Simulate I/O
                results.append(i * 2)
            return {"results": results}
        
        @xcs.jit
        def fast_optimized(*, inputs):
            results = []
            for i in range(inputs.get("count", 5)):
                time.sleep(0.005)  # Simulate I/O
                results.append(i * 2)
            return {"results": results}
        
        # Time unoptimized
        start = time.time()
        slow_result = slow_unoptimized(inputs={"count": 5})
        slow_time = time.time() - start
        
        # Time optimized
        start = time.time()
        fast_result = fast_optimized(inputs={"count": 5})
        fast_time = time.time() - start
        
        # Results should be identical
        assert slow_result == fast_result
        
        # Optimized should be noticeably faster for parallel operations
        # (This test might be flaky - in production we'd use better benchmarking)
        assert fast_time < slow_time * 0.7  # At least 30% faster
    
    def test_get_jit_stats_provides_insights(self):
        """Test that stats help understand performance."""
        @xcs.jit
        class InstrumentedOperator(Operator):
            specification = Specification()
            
            def forward(self, *, inputs):
                return {"processed": True}
        
        op = InstrumentedOperator()
        
        # Execute multiple times
        for _ in range(10):
            op(inputs={})
        
        stats = xcs.get_jit_stats()
        
        # Should have meaningful metrics
        assert stats is not None
        assert isinstance(stats, dict)
        # Exact keys depend on implementation, but should have something
        assert len(stats) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])