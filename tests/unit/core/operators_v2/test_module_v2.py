"""Tests for the simplified EmberModule system."""

import pytest
from dataclasses import fields, FrozenInstanceError
from ember.core.module_v2 import EmberModule, static_field, Chain, Ensemble


class SimpleModule(EmberModule):
    """Test module with basic fields."""
    value: int
    multiplier: float = 2.0
    
    def __call__(self, x: int) -> float:
        return (self.value + x) * self.multiplier


class StaticFieldModule(EmberModule):
    """Test module with static fields."""
    dynamic_value: int
    config: dict = static_field(default_factory=dict)
    name: str = static_field(default="test")
    
    def __call__(self, x: int) -> int:
        return self.dynamic_value + x


def test_automatic_dataclass():
    """EmberModule subclasses are automatically dataclasses."""
    # Can create instances like dataclasses
    module = SimpleModule(value=10)
    assert module.value == 10
    assert module.multiplier == 2.0
    
    # Has dataclass features
    assert len(fields(module)) == 2


def test_automatic_freezing():
    """EmberModule instances are frozen (immutable)."""
    module = SimpleModule(value=10)
    
    # Cannot modify fields
    with pytest.raises(FrozenInstanceError):
        module.value = 20
    
    with pytest.raises(FrozenInstanceError):
        module.multiplier = 3.0


def test_callable_protocol():
    """EmberModule instances are callable."""
    module = SimpleModule(value=10)
    result = module(5)
    assert result == 30.0  # (10 + 5) * 2.0


def test_replace_method():
    """Can create modified copies with replace()."""
    module1 = SimpleModule(value=10)
    module2 = module1.replace(value=20)
    
    # Original unchanged
    assert module1.value == 10
    assert module1(5) == 30.0
    
    # New instance updated
    assert module2.value == 20
    assert module2(5) == 50.0  # (20 + 5) * 2.0


def test_static_fields():
    """Static fields are excluded from PyTree operations."""
    module = StaticFieldModule(
        dynamic_value=10,
        config={"key": "value"},
        name="custom"
    )
    
    # Static fields are accessible
    assert module.config == {"key": "value"}
    assert module.name == "custom"
    
    # But marked as static in metadata
    for field in fields(module):
        if field.name in ["config", "name"]:
            assert field.metadata.get("static") == True
        else:
            assert field.metadata.get("static") != True


def test_ensemble_module():
    """Test the Ensemble module."""
    def add_one(x): 
        return x + 1
    
    def double(x): 
        return x * 2
    
    def square(x): 
        return x ** 2
    
    ensemble = Ensemble(operators=(add_one, double, square))
    results = ensemble(5)
    
    assert results == [6, 10, 25]
    assert len(results) == 3


def test_chain_module():
    """Test the Chain module."""
    def add_one(x): 
        return x + 1
    
    def double(x): 
        return x * 2
    
    def square(x): 
        return x ** 2
    
    chain = Chain(operators=(add_one, double, square))
    result = chain(5)
    
    # (5 + 1) * 2 = 12, then 12^2 = 144
    assert result == 144


def test_nested_modules():
    """Modules can contain other modules."""
    inner1 = SimpleModule(value=10)
    inner2 = SimpleModule(value=20, multiplier=3.0)
    
    ensemble = Ensemble(operators=(inner1, inner2))
    results = ensemble(5)
    
    assert results == [30.0, 75.0]  # (10+5)*2, (20+5)*3


def test_no_init_method():
    """Modules work without defining __init__."""
    class NoInitModule(EmberModule):
        value: int = 42
        
        def __call__(self, x):
            return self.value + x
    
    module = NoInitModule()
    assert module.value == 42
    assert module(8) == 50


def test_type_annotations_preserved():
    """Type annotations are preserved on fields."""
    # Check that type info is available
    value_field = next(f for f in fields(SimpleModule) if f.name == "value")
    assert value_field.type == int
    
    multiplier_field = next(f for f in fields(SimpleModule) if f.name == "multiplier")
    assert multiplier_field.type == float