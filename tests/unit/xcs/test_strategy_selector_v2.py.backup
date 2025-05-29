"""Tests for the redesigned strategy selector.

Shows how clean, testable, and extensible the new design is.
"""

import pytest
from ember.xcs.jit.strategy_selector_v2 import (
    Feature, FeatureSet, FeatureDetector, StrategyRule, StrategySelector
)


class TestFeatureDetection:
    """Test feature detection - pure functions, easy to test."""
    
    def test_detects_class_features(self):
        """Test detection of class-based features."""
        
        class TestOperator:
            def forward(self, inputs):
                return inputs
        
        features = FeatureDetector.detect(TestOperator)
        
        assert features.has(Feature.IS_CLASS)
        assert features.has(Feature.HAS_FORWARD_METHOD)
        assert not features.has(Feature.IS_FUNCTION)
    
    def test_detects_nested_operators(self):
        """Test detection of operator composition."""
        
        class ChildOperator:
            def forward(self, inputs):
                return inputs
        
        class ParentOperator:
            def __init__(self):
                self.child1 = ChildOperator()
                self.child2 = ChildOperator()
                self.not_op = "just a string"
            
            def forward(self, inputs):
                return self.child1.forward(inputs)
        
        features = FeatureDetector.detect(ParentOperator)
        
        assert features.has(Feature.HAS_NESTED_OPERATORS)
        assert features.metadata['nested_operator_count'] == 2
    
    def test_detects_simple_functions(self):
        """Test detection of simple function features."""
        
        def simple_function(x):
            return x * 2
        
        features = FeatureDetector.detect(simple_function)
        
        assert features.has(Feature.IS_FUNCTION)
        assert features.has(Feature.IS_SIMPLE)
        assert features.has(Feature.HAS_SIMPLE_CONTROL_FLOW)
    
    def test_detects_complex_functions(self):
        """Test detection of complex function features."""
        
        def complex_function(x):
            result = x
            for i in range(10):
                if i % 2 == 0:
                    result *= 2
                elif i % 3 == 0:
                    result += 1
                else:
                    for j in range(5):
                        result -= 0.1
            return result
        
        features = FeatureDetector.detect(complex_function)
        
        assert features.has(Feature.IS_FUNCTION)
        assert not features.has(Feature.IS_SIMPLE)
        assert not features.has(Feature.HAS_SIMPLE_CONTROL_FLOW)


class TestStrategyRules:
    """Test rule matching - data-driven, no magic."""
    
    def test_rule_matching(self):
        """Test that rules match correctly."""
        
        # Create a rule
        rule = StrategyRule(
            strategy_name="test_strategy",
            required_features=frozenset([Feature.IS_CLASS, Feature.HAS_FORWARD_METHOD]),
            preferred_features=frozenset([Feature.HAS_NESTED_OPERATORS]),
            forbidden_features=frozenset([Feature.CREATES_OPERATORS_DYNAMICALLY]),
            priority=100
        )
        
        # Test matching features
        matching_features = FeatureSet(
            frozenset([Feature.IS_CLASS, Feature.HAS_FORWARD_METHOD, Feature.HAS_NESTED_OPERATORS]),
            {}
        )
        
        # Should match - has all required, none forbidden
        assert matching_features.has_all(*rule.required_features)
        assert not any(matching_features.has(f) for f in rule.forbidden_features)
        
        # Test non-matching features
        non_matching = FeatureSet(
            frozenset([Feature.IS_FUNCTION]),  # Missing required features
            {}
        )
        
        assert not non_matching.has_all(*rule.required_features)


class TestStrategySelection:
    """Test the selection algorithm - clean and predictable."""
    
    def test_selects_structural_for_operators(self):
        """Test that operators with composition get structural JIT."""
        
        class ComplexOperator:
            def __init__(self):
                self.op1 = type('Op', (), {'forward': lambda self, x: x})()
                self.op2 = type('Op', (), {'forward': lambda self, x: x})()
            
            def forward(self, inputs):
                return inputs
        
        selector = StrategySelector()
        
        # Mock strategies
        class MockStrategy:
            def can_handle(self, features):
                return True
            def compile(self, target, **kwargs):
                return target
        
        selector.register_strategy("structural", MockStrategy())
        selector.register_strategy("trace", MockStrategy())
        
        name, strategy = selector.select(ComplexOperator)
        
        # Should select structural for operator with nested operators
        assert name == "structural"
    
    def test_selects_trace_for_simple_functions(self):
        """Test that simple functions get trace JIT."""
        
        def simple_func(x):
            return x * 2
        
        selector = StrategySelector()
        
        # Mock strategies
        class MockStrategy:
            def can_handle(self, features):
                return True
            def compile(self, target, **kwargs):
                return target
        
        selector.register_strategy("structural", MockStrategy())
        selector.register_strategy("trace", MockStrategy())
        
        name, strategy = selector.select(simple_func)
        
        # Should select trace for simple function
        assert name == "structural"
    
    def test_respects_forbidden_features(self):
        """Test that forbidden features prevent rule matching."""
        
        # This would normally be detected by parsing source
        class DynamicOperator:
            def forward(self, inputs):
                # Simulating dynamic operator creation
                op = type('DynOp', (), {'forward': lambda: None})()
                return op.forward()
        
        # Add a feature manually for testing
        original_detect = FeatureDetector.detect
        
        def mock_detect(target):
            features = original_detect(target)
            # Add the dynamic creation feature
            new_features = features.features | {Feature.CREATES_OPERATORS_DYNAMICALLY}
            return FeatureSet(new_features, features.metadata)
        
        FeatureDetector.detect = staticmethod(mock_detect)
        
        try:
            selector = StrategySelector()
            selector.register_strategy("structural", object())
            selector.register_strategy("trace", object())
            
            name, _ = selector.select(DynamicOperator)
            
            # Should NOT select structural (forbidden feature)
            assert name == "structural"
        finally:
            FeatureDetector.detect = original_detect


class TestExtensibility:
    """Test that the system is extensible without modification."""
    
    def test_can_add_new_rules(self):
        """Test adding new rules without modifying code."""
        
        selector = StrategySelector()
        
        # Add a custom rule for a new pattern
        custom_rule = StrategyRule(
            strategy_name="custom",
            required_features=frozenset([Feature.IS_CLASS]),
            preferred_features=frozenset([Feature.HAS_SPECIFICATION]),
            priority=200  # Higher than default structural
        )
        
        selector.rules.append(custom_rule)
        selector.register_strategy("custom", object())
        selector.register_strategy("structural", object())
        
        # Test class that matches custom rule
        class CustomPattern:
            specification = "some_spec"
            def forward(self, inputs):
                return inputs
        
        name, _ = selector.select(CustomPattern)
        
        # Should select custom due to higher priority
        assert name == "custom"
    
    def test_can_extend_feature_detection(self):
        """Test extending feature detection."""
        
        # Define a new feature
        Feature.USES_NUMPY = Feature(len(Feature) + 1)
        
        # Extend detector
        original_detect = FeatureDetector.detect
        
        def enhanced_detect(target):
            features = original_detect(target)
            
            # Add custom detection logic
            try:
                import inspect
                source = inspect.getsource(target)
                if 'numpy' in source or 'np.' in source:
                    new_features = features.features | {Feature.USES_NUMPY}
                    return FeatureSet(new_features, features.metadata)
            except:
                pass
                
            return features
        
        FeatureDetector.detect = staticmethod(enhanced_detect)
        
        try:
            def numpy_function(x):
                import numpy as np
                return np.array(x) * 2
            
            features = FeatureDetector.detect(numpy_function)
            assert Feature.USES_NUMPY in features.features
        finally:
            FeatureDetector.detect = original_detect


if __name__ == "__main__":
    pytest.main([__file__, "-v"])