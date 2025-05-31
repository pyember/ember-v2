"""Integration tests showing v2 operators with models and XCS."""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List

from ember.core.operators_v2 import Operator
from ember.core.module_v2 import EmberModule, Ensemble, Chain


# Mock for models API
def create_mock_model(name: str, response_text: str):
    """Create a mock model for testing."""
    mock = Mock()
    mock.name = name
    response = Mock()
    response.text = response_text
    mock.return_value = response
    return mock


def test_simple_function_with_model():
    """Test simple function operator with model."""
    mock_model = create_mock_model("gpt-4", "positive")
    
    def sentiment_classifier(text: str) -> str:
        response = mock_model(f"Classify sentiment: {text}")
        return response.text
    
    # It's an operator
    assert isinstance(sentiment_classifier, Operator)
    
    # It works
    result = sentiment_classifier("I love this!")
    assert result == "positive"
    mock_model.assert_called_with("Classify sentiment: I love this!")


def test_stateful_operator_with_model():
    """Test EmberModule operator with model."""
    mock_model = create_mock_model("gpt-4", "The answer is 42")
    
    class QuestionAnswerer(EmberModule):
        model: Mock  # In real code, this would be a model binding
        style: str = "concise"
        
        def __call__(self, question: str) -> str:
            prompt = f"Answer {self.style}ly: {question}"
            response = self.model(prompt)
            return response.text
    
    qa = QuestionAnswerer(model=mock_model)
    answer = qa("What is the meaning of life?")
    
    assert answer == "The answer is 42"
    mock_model.assert_called_with("Answer concisely: What is the meaning of life?")
    
    # Can create variations
    verbose_qa = qa.replace(style="verbose")
    assert verbose_qa.style == "verbose"


def test_ensemble_with_models():
    """Test ensemble of model-based operators."""
    model1 = create_mock_model("gpt-4", "positive")
    model2 = create_mock_model("gpt-3.5", "positive")
    model3 = create_mock_model("claude", "negative")
    
    # Create sentiment classifiers with different models
    classifiers = [
        lambda text: model1(f"Sentiment: {text}").text,
        lambda text: model2(f"Sentiment: {text}").text,
        lambda text: model3(f"Sentiment: {text}").text,
    ]
    
    ensemble = Ensemble(operators=tuple(classifiers))
    results = ensemble("Great product!")
    
    assert results == ["positive", "positive", "negative"]
    
    # Can use with selectors
    from collections import Counter
    most_common = Counter(results).most_common(1)[0][0]
    assert most_common == "positive"


def test_chain_with_typed_data():
    """Test chain with structured data flow."""
    
    @dataclass
    class Query:
        text: str
        max_length: int = 100
    
    @dataclass
    class EnhancedQuery:
        original: str
        expanded: str
        keywords: List[str]
    
    # Mock model for query expansion
    expander_model = create_mock_model("gpt-4", "What is machine learning and how does it work?")
    
    def expand_query(query: Query) -> EnhancedQuery:
        expanded = expander_model(f"Expand: {query.text}").text
        keywords = query.text.split()[:3]  # Simple keyword extraction
        return EnhancedQuery(
            original=query.text,
            expanded=expanded,
            keywords=keywords
        )
    
    def format_for_search(enhanced: EnhancedQuery) -> str:
        return f"{enhanced.expanded} Keywords: {', '.join(enhanced.keywords)}"
    
    # Chain them
    pipeline = Chain(operators=(expand_query, format_for_search))
    
    result = pipeline(Query("What is ML?"))
    expected = "What is machine learning and how does it work? Keywords: What, is, ML?"
    assert result == expected


def test_nested_operators():
    """Test operators containing operators."""
    mock_model = create_mock_model("gpt-4", "Summary: Great product")
    
    class Summarizer(EmberModule):
        model: Mock
        max_length: int = 100
        
        def __call__(self, text: str) -> str:
            prompt = f"Summarize in {self.max_length} chars: {text}"
            return self.model(prompt).text
    
    class MultiSummarizer(EmberModule):
        summarizers: tuple
        
        def __call__(self, text: str) -> List[str]:
            return [s(text) for s in self.summarizers]
    
    # Create different summarizers
    brief = Summarizer(model=mock_model, max_length=50)
    normal = Summarizer(model=mock_model, max_length=100)
    detailed = Summarizer(model=mock_model, max_length=200)
    
    multi = MultiSummarizer(summarizers=(brief, normal, detailed))
    results = multi("Long product review...")
    
    assert len(results) == 3
    assert all(r == "Summary: Great product" for r in results)


def test_mixed_operator_types():
    """Test mixing functions and modules."""
    mock_model = create_mock_model("gpt-4", "clean text")
    
    # Simple function
    def lowercase(text: str) -> str:
        return text.lower()
    
    # EmberModule
    class Cleaner(EmberModule):
        model: Mock
        
        def __call__(self, text: str) -> str:
            return self.model(f"Clean: {text}").text
    
    # Another function  
    def remove_punctuation(text: str) -> str:
        return "".join(c for c in text if c.isalnum() or c.isspace())
    
    # Chain them all
    pipeline = Chain(operators=(
        lowercase,
        Cleaner(model=mock_model),
        remove_punctuation
    ))
    
    result = pipeline("HELLO WORLD!")
    assert result == "clean text"  # mock returns "clean text"


# Mock XCS transformations for testing
def mock_jit(func):
    """Mock jit transformation."""
    func._is_jitted = True
    return func

def mock_vmap(func):
    """Mock vmap transformation."""
    def vmapped(inputs):
        return [func(x) for x in inputs]
    vmapped._is_vmapped = True
    return vmapped


def test_transformations_with_operators():
    """Test that operators work with XCS transformations."""
    def classifier(text: str) -> str:
        return "positive" if "good" in text else "negative"
    
    # Apply transformations
    fast_classifier = mock_jit(classifier)
    batch_classifier = mock_vmap(classifier)
    
    # Single call
    assert fast_classifier("good day") == "positive"
    assert hasattr(fast_classifier, "_is_jitted")
    
    # Batch call
    results = batch_classifier(["good day", "bad day", "good morning"])
    assert results == ["positive", "negative", "positive"]
    assert hasattr(batch_classifier, "_is_vmapped")