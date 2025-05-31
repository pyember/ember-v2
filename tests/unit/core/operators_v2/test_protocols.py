"""Tests for the minimal operator protocols."""

import pytest
from typing import List
from ember.core.operators_v2 import Operator


def test_function_is_operator():
    """Any function is an operator."""
    def double(x: int) -> int:
        return x * 2
    
    assert isinstance(double, Operator)
    assert double(5) == 10


def test_callable_class_is_operator():
    """Any callable class is an operator."""
    class Multiplier:
        def __init__(self, factor: int):
            self.factor = factor
        
        def __call__(self, x: int) -> int:
            return x * self.factor
    
    triple = Multiplier(3)
    assert isinstance(triple, Operator)
    assert triple(5) == 15


def test_lambda_is_operator():
    """Lambdas are operators."""
    add_one = lambda x: x + 1
    assert isinstance(add_one, Operator)
    assert add_one(5) == 6


def test_operator_composition():
    """Operators compose naturally."""
    def add_two(x: int) -> int:
        return x + 2
    
    def multiply_by_three(x: int) -> int:
        return x * 3
    
    # Manual composition
    def composed(x: int) -> int:
        return multiply_by_three(add_two(x))
    
    assert composed(5) == 21  # (5 + 2) * 3
    assert isinstance(composed, Operator)


def test_operator_with_multiple_args():
    """Operators can have multiple arguments."""
    def add(x: float, y: float) -> float:
        return x + y
    
    # Still an operator (transforms tuple of inputs to output)
    assert add(3, 4) == 7


def test_generic_operators():
    """Operators work with any types."""
    def first_word(text: str) -> str:
        return text.split()[0] if text else ""
    
    def word_count(text: str) -> int:
        return len(text.split())
    
    def join_words(words: List[str]) -> str:
        return " ".join(words)
    
    assert isinstance(first_word, Operator)
    assert isinstance(word_count, Operator)
    assert isinstance(join_words, Operator)
    
    assert first_word("Hello world") == "Hello"
    assert word_count("Hello world") == 2
    assert join_words(["Hello", "world"]) == "Hello world"