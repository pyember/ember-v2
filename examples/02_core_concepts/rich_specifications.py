"""Rich Specifications - Complex input/output validation with EmberModel.

Difficulty: Intermediate
Time: ~10 minutes

Learning Objectives:
- Define rich structured inputs/outputs with EmberModel
- Use Pydantic validation features (constraints, validators)
- Create nested and complex data structures
- Build type-safe operators with specifications
- Handle validation errors gracefully

This showcases Ember's powerful specification system that provides:
- Zero-overhead type safety (EmberModel = Pydantic BaseModel)
- Rich validation with constraints
- Nested structures with full validation
- Custom validators for complex logic
- Excellent error messages
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Literal

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import operators
from ember._internal.types import EmberModel
from pydantic import Field, field_validator, model_validator


def main():
    """Explore Ember's rich specification capabilities."""
    print_section_header("Rich Input/Output Specifications")
    
    # Part 1: The Problem - Unvalidated Data
    print("Part 1: The Problem with Unvalidated Data")
    print("=" * 50 + "\n")
    
    print("🤔 Without validation, AI applications break in subtle ways...")
    print()
    
    # Show what goes wrong with plain dictionaries
    print("# Using plain dictionaries (what most people start with):")
    def analyze_sentiment_unsafe(data):
        """Analyze sentiment without validation."""
        # Assumes data has correct structure
        text = data["text"]
        model = data.get("model", "gpt-3.5-turbo")
        temperature = data.get("temperature", 0.7)
        return f"Analyzing '{text}' with {model} (temp={temperature})"
    
    print()
    print("❌ What breaks:")
    
    # Demonstrate failures
    bad_inputs = [
        {},  # Missing required field
        {"text": ""},  # Empty text
        {"text": "Valid text", "temperature": 2.5},  # Invalid temperature
    ]
    
    for i, bad_input in enumerate(bad_inputs, 1):
        try:
            result = analyze_sentiment_unsafe(bad_input)
            print(f"  {i}. Silently wrong: {result}")
        except Exception as e:
            print(f"  {i}. Runtime error: {type(e).__name__}")
    
    print("\n✅ Rich specifications solve this!")
    print()
    
    # Part 2: Basic Solution with EmberModel
    print("=" * 50)
    print("Part 2: Solution with EmberModel")
    print("=" * 50 + "\n")
    
    class PromptRequest(EmberModel):
        """Validated prompt request for LLM analysis."""
        text: str = Field(min_length=1, max_length=2000)
        model: str = Field(default="gpt-3.5-turbo", pattern="^(gpt-3.5-turbo|gpt-4|claude-3)$")
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        max_tokens: int = Field(default=100, ge=1, le=1000)
        
        @field_validator("text")
        def clean_text(cls, v):
            """Clean and validate text input."""
            v = v.strip()
            if not v:
                raise ValueError("Text cannot be empty or just whitespace")
            return v
    
    print("Now the same inputs are validated:")
    print()
    
    # Test with the same bad inputs
    for i, bad_input in enumerate(bad_inputs, 1):
        try:
            request = PromptRequest(**bad_input)
            print(f"  {i}. ✓ Valid: {request.text[:20]}... ({request.model})")
        except Exception as e:
            print(f"  {i}. ✗ Caught: {str(e)[:100]}...")
    
    # Show valid usage
    print("\nValid request:")
    try:
        good_request = PromptRequest(
            text="Analyze the sentiment of this text",
            model="gpt-4",
            temperature=0.3
        )
        print_example_output("Text", good_request.text)
        print_example_output("Model", good_request.model)
        print_example_output("Temperature", good_request.temperature)
    except Exception as e:
        print(f"Error: {e}")
    
    # Part 3: Structured API Response Parsing
    print("\n" + "=" * 50)
    print("Part 3: Parsing LLM API Responses")
    print("=" * 50 + "\n")
    
    print("🎯 Real-world use case: Parsing structured LLM responses")
    print()
    
    class AnalysisResult(EmberModel):
        """Structured LLM analysis response."""
        sentiment: Literal["positive", "negative", "neutral"]
        confidence: float = Field(ge=0.0, le=1.0)
        key_phrases: List[str] = Field(default_factory=list, max_items=5)
        language: str = Field(default="en", pattern="^[a-z]{2}$")
        
        @field_validator("key_phrases")
        def validate_phrases(cls, v):
            """Ensure phrases are non-empty."""
            return [phrase.strip() for phrase in v if phrase.strip()]
    
    class BatchAnalysisRequest(EmberModel):
        """Batch processing request with validation."""
        texts: List[str] = Field(min_items=1, max_items=10)
        prompt_config: Optional[PromptRequest] = None
        batch_id: str = Field(pattern="^[A-Z0-9-]+$")
        
        @model_validator(mode="after")
        def validate_batch(self):
            """Cross-field validation for batch processing."""
            # Ensure all texts are valid
            for i, text in enumerate(self.texts):
                if len(text.strip()) < 5:
                    raise ValueError(f"Text {i+1} is too short (minimum 5 characters)")
            return self
    
    # Demo structured response parsing
    print("Parsing LLM responses with validation:")
    
    # Simulate API responses (some valid, some invalid)
    api_responses = [
        {
            "sentiment": "positive",
            "confidence": 0.95,
            "key_phrases": ["great product", "highly recommend"],
            "language": "en"
        },
        {
            "sentiment": "invalid_sentiment",  # Invalid enum
            "confidence": 1.5,  # Out of range
            "key_phrases": [],
        },
        {
            "sentiment": "negative",
            "confidence": 0.8,
            "key_phrases": ["", "  ", "disappointed"],  # Empty phrases
            "language": "fr"
        }
    ]
    
    for i, response in enumerate(api_responses, 1):
        try:
            result = AnalysisResult(**response)
            print(f"  {i}. ✓ Valid: {result.sentiment} ({result.confidence:.2f})")
            if result.key_phrases:
                print(f"      Phrases: {result.key_phrases}")
        except Exception as e:
            print(f"  {i}. ✗ Invalid response: {str(e)[:60]}...")
    
    # Demo batch processing validation
    print("\nBatch processing with validation:")
    try:
        batch_request = BatchAnalysisRequest(
            texts=["Great product!", "Not bad", "Absolutely terrible"],
            batch_id="BATCH-001"
        )
        print(f"✓ Valid batch: {len(batch_request.texts)} texts")
    except Exception as e:
        print(f"✗ Batch validation failed: {e}")
    
    # Part 4: Operators with input_spec and output_spec
    print("\n" + "=" * 50)
    print("Part 4: Type-Safe Operators")
    print("=" * 50 + "\n")
    
    print("🚀 Building operators with automatic validation:")
    print()
    
    # Simple operator with specs
    class SentimentAnalyzer(operators.Operator):
        """Analyzes sentiment with automatic input/output validation."""
        
        # Ember automatically validates inputs and outputs!
        input_spec = PromptRequest
        output_spec = AnalysisResult
        
        def forward(self, request: PromptRequest) -> AnalysisResult:
            """Process validated input, return validated output."""
            # Input is already validated by EmberModel
            text = request.text.lower()
            
            # Simple sentiment analysis (normally you'd call an LLM)
            positive_words = ["great", "good", "excellent", "amazing", "love"]
            negative_words = ["bad", "terrible", "awful", "hate", "horrible"]
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                sentiment = "positive"
                confidence = 0.8
            elif neg_count > pos_count:
                sentiment = "negative" 
                confidence = 0.8
            else:
                sentiment = "neutral"
                confidence = 0.6
            
            # Output is automatically validated by EmberModel
            return AnalysisResult(
                sentiment=sentiment,
                confidence=confidence,
                key_phrases=[word for word in positive_words + negative_words if word in text][:3],
                language="en"
            )
    
    # Test the type-safe operator
    analyzer = SentimentAnalyzer()
    
    print("Testing with valid inputs:")
    test_cases = [
        {"text": "This product is amazing and excellent!", "model": "gpt-4"},
        {"text": "Terrible service, really bad experience", "temperature": 0.2},
        {"text": "It's okay, nothing special"}
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        try:
            # Operator automatically validates input dict -> PromptRequest
            request = PromptRequest(**test_input)
            result = analyzer(request)
            print(f"  {i}. ✓ {result.sentiment} (confidence: {result.confidence:.1f})")
            if result.key_phrases:
                print(f"      Key phrases: {result.key_phrases}")
        except Exception as e:
            print(f"  {i}. ✗ Input validation failed: {str(e)[:40]}...")
    
    print("\nTesting with invalid inputs:")
    invalid_cases = [
        {"text": ""},  # Empty text
        {"text": "Valid text", "temperature": 3.0},  # Invalid temperature
        {"model": "invalid-model"}  # Missing required text
    ]
    
    for i, invalid_input in enumerate(invalid_cases, 1):
        try:
            request = PromptRequest(**invalid_input)
            result = analyzer(request)
            print(f"  {i}. ✗ Should have failed validation!")
        except Exception as e:
            print(f"  {i}. ✓ Correctly rejected: {str(e)[:40]}...")
    
    # Part 5: Advanced Patterns for AI Workflows
    print("\n" + "=" * 50)
    print("Part 5: Advanced AI Validation Patterns")
    print("=" * 50 + "\n")
    
    print("🧠 Complex validation for multi-step AI workflows:")
    print()
    
    class MultiModalRequest(EmberModel):
        """Multi-modal AI request with conditional validation."""
        text: Optional[str] = None
        image_url: Optional[str] = None
        audio_url: Optional[str] = None
        
        task: Literal["transcribe", "analyze", "generate", "translate"]
        target_language: Optional[str] = Field(None, pattern="^[a-z]{2}$")
        max_tokens: int = Field(default=500, ge=1, le=2000)
        
        @model_validator(mode="after")
        def validate_multimodal_logic(self):
            """Conditional validation based on task type."""
            # At least one input required
            if not any([self.text, self.image_url, self.audio_url]):
                raise ValueError("At least one input (text, image, or audio) required")
            
            # Task-specific validation
            if self.task == "transcribe" and not self.audio_url:
                raise ValueError("Transcription requires audio_url")
            
            if self.task == "translate":
                if not self.text:
                    raise ValueError("Translation requires text input")
                if not self.target_language:
                    raise ValueError("Translation requires target_language")
            
            if self.task == "analyze" and self.image_url and not self.text:
                self.text = "Analyze this image"  # Auto-fill for image analysis
            
            return self
    
    class WorkflowStep(EmberModel):
        """Individual step in an AI processing workflow."""
        step_id: str
        operator_name: str
        input_mapping: Dict[str, str] = Field(default_factory=dict)
        dependencies: List[str] = Field(default_factory=list)
        
        @field_validator("step_id")
        def validate_step_id(cls, v):
            """Ensure step IDs are valid identifiers."""
            if not v.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Step ID must be alphanumeric with - or _")
            return v
    
    print("Testing multi-modal validation:")
    
    # Test cases for multi-modal requests
    multimodal_tests = [
        {
            "task": "translate",
            "text": "Hello world",
            "target_language": "es"
        },
        {
            "task": "transcribe",
            "text": "Some text"  # Missing audio_url
        },
        {
            "task": "analyze",
            "image_url": "https://example.com/image.jpg"
        }
    ]
    
    for i, test in enumerate(multimodal_tests, 1):
        try:
            request = MultiModalRequest(**test)
            print(f"  {i}. ✓ Valid {request.task} request")
            if hasattr(request, 'text') and request.text and "Analyze" in request.text:
                print(f"      Auto-filled text: '{request.text}'")
        except Exception as e:
            print(f"  {i}. ✗ Validation failed: {str(e)[:50]}...")
    
    # Part 6: Error Messages
    print("\n" + "=" * 50)
    print("Part 6: Helpful Error Messages")
    print("=" * 50 + "\n")
    
    print("EmberModel provides clear, actionable error messages:")
    print()
    
    try:
        # Deliberately create invalid request
        bad_request = PromptRequest(
            text="",  # Empty
            model="invalid-model",  # Bad pattern
            temperature=5.0,  # Out of range
            max_tokens=-1  # Negative
        )
    except Exception as e:
        print("Example validation errors:")
        error_lines = str(e).split("\n")[:5]
        for line in error_lines:
            if line.strip():
                print(f"  {line.strip()}")
        print("  (errors show exact field and constraint violated)")
    
    print("\n✨ EmberModel catches errors early and clearly!")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Rich Specifications Summary")
    print("=" * 50)
    
    print("\n🎯 Why Use Rich Specifications?")
    print("  • Catch errors early, not at runtime")
    print("  • Clear validation messages for debugging")
    print("  • Type-safe AI applications")
    print("  • Automatic input/output validation in operators")
    print("  • Better API contracts and documentation")
    
    print("\n💡 When to Use:")
    print("  🟢 Use for: API inputs, LLM responses, structured data")
    print("  🟢 Use for: Multi-step workflows, batch processing")
    print("  🔴 Skip for: Simple scripts, quick prototypes")
    
    print("\n📚 Quick Reference:")
    print("```python")
    print("# Basic pattern")
    print("class MyInput(EmberModel):")
    print("    text: str = Field(min_length=1, max_length=1000)")
    print("    model: str = Field(default='gpt-3.5-turbo')")
    print("    ")
    print("    @field_validator('text')")
    print("    def clean_text(cls, v):")
    print("        return v.strip()")
    print("")
    print("# Use in operators")
    print("class MyOperator(operators.Operator):")
    print("    input_spec = MyInput  # Auto-validation!")
    print("    output_spec = MyOutput")
    print("```")
    
    print("\nNext: Learn about simplified APIs in 'error_handling.py'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())