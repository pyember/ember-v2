"""Structured Output - Type-safe LLM outputs.

Guarantee valid structured data from LLMs using validation,
retry logic, and type-safe models.

Example:
    >>> extractor = StructuredExtractor(output_model=ProductInfo)
    >>> product = extractor(text="iPhone 15 Pro, $999, in stock")
    >>> print(f"{product.name}: ${product.price}")
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import random

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import operators
from ember.api.operators import EmberModel, Field


def main():
    """Example demonstrating the simplified XCS architecture."""
    """Demonstrate structured output patterns."""
    print_section_header("Structured Output Patterns")
    
    # Part 1: Define Structured Models
    print("📋 Part 1: Defining Structured Output Models\n")
    
    class ProductInfo(EmberModel):
        """Structured product information."""
        name: str = Field(description="Product name")
        price: float = Field(description="Price in USD")
        category: str = Field(description="Product category")
        in_stock: bool = Field(description="Whether item is in stock")
        tags: List[str] = Field(default_factory=list, description="Product tags")
    
    class AnalysisResult(EmberModel):
        """Structured analysis output."""
        summary: str = Field(description="Brief summary")
        sentiment: str = Field(description="Sentiment: positive/negative/neutral")
        key_points: List[str] = Field(description="Key points extracted")
        confidence: float = Field(description="Confidence score 0-1")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    print("Defined structured models:")
    print("- ProductInfo: name, price, category, in_stock, tags")
    print("- AnalysisResult: summary, sentiment, key_points, confidence, metadata")
    
    # Part 2: JSON Parser Operator
    print("\n" + "="*50)
    print("🔧 Part 2: JSON Parsing with Validation")
    print("="*50 + "\n")
    
    class JSONParserOperator(operators.Operator):
        """Parses and validates JSON from text."""
        
        specification = operators.Specification()
        
        def extract_json(self, text: str) -> Optional[Dict]:
            """Extract JSON from text, handling common LLM quirks."""
            # Try direct parsing first
            try:
                return json.loads(text)
            except:
                pass
            
            # Try to find JSON block in markdown
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                if end > start:
                    try:
                        return json.loads(text[start:end].strip())
                    except:
                        pass
            
            # Try to find JSON-like structure
            for start_char in ['{', '[']:
                if start_char in text:
                    start = text.find(start_char)
                    # Find matching bracket
                    bracket_count = 0
                    for i, char in enumerate(text[start:], start):
                        if char in '{[':
                            bracket_count += 1
                        elif char in '}]':
                            bracket_count -= 1
                            if bracket_count == 0:
                                try:
                                    return json.loads(text[start:i+1])
                                except:
                                    pass
            
            return None
        
        def forward(self, *, inputs):
            text = inputs.get("text", "")
            expected_type = inputs.get("expected_type", dict)
            
            # Try to extract JSON
            parsed = self.extract_json(text)
            
            if parsed is None:
                return {
                    "success": False,
                    "error": "No valid JSON found in text",
                    "parsed": None,
                    "original_text": text
                }
            
            # Validate type
            if not isinstance(parsed, expected_type):
                return {
                    "success": False,
                    "error": f"Expected {expected_type.__name__}, got {type(parsed).__name__}",
                    "parsed": parsed,
                    "original_text": text
                }
            
            return {
                "success": True,
                "parsed": parsed,
                "error": None,
                "original_text": text
            }
    
    # Test the parser
    parser = JSONParserOperator()
    
    test_cases = [
        '{"name": "Widget", "price": 19.99}',
        'Here is the JSON: ```json\n{"category": "tools", "in_stock": true}\n```',
        'The result is {"tags": ["new", "featured"], "confidence": 0.95}.'
    ]
    
    print("Testing JSON parser:")
    for text in test_cases:
        result = parser(text=text)
        if result["success"]:
            print(f"✓ Parsed: {result['parsed']}")
        else:
            print(f"✗ Failed: {result['error']}")
    
    # Part 3: Validation and Retry
    print("\n" + "="*50)
    print("🔄 Part 3: Validation with Retry Logic")
    print("="*50 + "\n")
    
    class ValidatedOutputOperator(operators.Operator):
        """Ensures output matches expected structure with retries."""
        
        specification = operators.Specification()
        
        def __init__(self, *, output_model: type, max_retries: int = 3):
            self.output_model = output_model
            self.max_retries = max_retries
            self.parser = JSONParserOperator()
        
        def simulate_llm_response(self, prompt: str, attempt: int) -> str:
            """Simulate LLM responses with varying quality."""
            if "product" in prompt.lower():
                if attempt == 0 and random.random() < 0.3:
                    # Sometimes return malformed JSON
                    return "The product is {name: 'Laptop', price: 999.99"
                else:
                    return json.dumps({
                        "name": "Laptop",
                        "price": 999.99,
                        "category": "Electronics",
                        "in_stock": True,
                        "tags": ["computers", "portable"]
                    })
            elif "analyze" in prompt.lower():
                return json.dumps({
                    "summary": "This is a positive review of the product",
                    "sentiment": "positive",
                    "key_points": ["High quality", "Good value", "Fast shipping"],
                    "confidence": 0.87,
                    "metadata": {"word_count": 150, "language": "en"}
                })
            else:
                return "{}"
        
        def forward(self, *, inputs):
            prompt = inputs.get("prompt", "")
            
            for attempt in range(self.max_retries):
                # Get response (simulated)
                response_text = self.simulate_llm_response(prompt, attempt)
                
                # Parse JSON
                parse_result = self.parser(text=response_text)
                
                if not parse_result["success"]:
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        return {
                            "success": False,
                            "error": f"Failed after {self.max_retries} attempts: {parse_result['error']}",
                            "data": None,
                            "attempts": attempt + 1
                        }
                
                # Validate against model
                try:
                    validated = self.output_model(**parse_result["parsed"])
                    return {
                        "success": True,
                        "data": validated,
                        "error": None,
                        "attempts": attempt + 1
                    }
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        return {
                            "success": False,
                            "error": f"Validation failed: {str(e)}",
                            "data": None,
                            "attempts": attempt + 1
                        }
            
            return {
                "success": False,
                "error": "Max retries exceeded",
                "data": None,
                "attempts": self.max_retries
            }
    
    # Test validation with retry
    validator = ValidatedOutputOperator(output_model=ProductInfo)
    
    print("Testing validated output with retry:")
    for i in range(3):
        result = validator(prompt="Get product information")
        if result["success"]:
            print(f"✓ Success on attempt {result['attempts']}: {result['data'].name}")
        else:
            print(f"✗ Failed: {result['error']}")
    
    # Part 4: Complete Structured Output Pipeline
    print("\n" + "="*50)
    print("🏗️ Part 4: Complete Structured Output Pipeline")
    print("="*50 + "\n")
    
    class StructuredLLMOperator(operators.Operator):
        """Complete operator for structured LLM outputs."""
        
        specification = operators.Specification()
        
        def __init__(self, *, output_model: type):
            self.output_model = output_model
            self.validator = ValidatedOutputOperator(output_model=output_model)
        
        def build_prompt(self, user_prompt: str) -> str:
            """Build prompt that encourages structured output."""
            schema = {}
            for field_name, field_info in self.output_model.model_fields.items():
                schema[field_name] = {
                    "type": field_info.annotation.__name__ if hasattr(field_info.annotation, '__name__') else str(field_info.annotation),
                    "description": field_info.description
                }
            
            return f"""
{user_prompt}

Please respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Remember to include all required fields.
"""
        
        def forward(self, *, inputs):
            user_prompt = inputs.get("prompt", "")
            
            # Build structured prompt
            full_prompt = self.build_prompt(user_prompt)
            
            # Get validated output
            result = self.validator(prompt=full_prompt)
            
            if result["success"]:
                return {
                    "success": True,
                    "data": result["data"],
                    "prompt": user_prompt,
                    "attempts": result["attempts"]
                }
            else:
                # Fallback with defaults
                try:
                    default_data = self.output_model()
                    return {
                        "success": False,
                        "data": default_data,
                        "prompt": user_prompt,
                        "attempts": result["attempts"],
                        "error": result["error"]
                    }
                except:
                    return {
                        "success": False,
                        "data": None,
                        "prompt": user_prompt,
                        "attempts": result["attempts"],
                        "error": result["error"]
                    }
    
    # Example usage
    print("Using Structured LLM Operators:\n")
    
    # Product extraction
    product_extractor = StructuredLLMOperator(output_model=ProductInfo)
    result = product_extractor(prompt="Extract product details from this text")
    
    if result["success"]:
        product = result["data"]
        print(f"Product Extraction Success:")
        print(f"  Name: {product.name}")
        print(f"  Price: ${product.price}")
        print(f"  Category: {product.category}")
        print(f"  In Stock: {product.in_stock}")
        print(f"  Tags: {', '.join(product.tags)}")
    
    # Analysis extraction
    print("\n")
    analyzer = StructuredLLMOperator(output_model=AnalysisResult)
    result = analyzer(prompt="Analyze this customer review")
    
    if result["success"]:
        analysis = result["data"]
        print(f"Analysis Extraction Success:")
        print(f"  Sentiment: {analysis.sentiment}")
        print(f"  Confidence: {analysis.confidence:.2%}")
        print(f"  Key Points: {', '.join(analysis.key_points[:2])}...")
    
    # Part 5: Error Recovery Patterns
    print("\n" + "="*50)
    print("🛡️ Part 5: Error Recovery Patterns")
    print("="*50 + "\n")
    
    class RobustStructuredOperator(operators.Operator):
        """Robust structured output with multiple fallback strategies."""
        
        specification = operators.Specification()
        
        def __init__(self, *, output_model: type):
            self.output_model = output_model
            self.primary = StructuredLLMOperator(output_model=output_model)
        
        def forward(self, *, inputs):
            prompt = inputs.get("prompt", "")
            
            # Try primary method
            result = self.primary(prompt=prompt)
            
            if result["success"]:
                return {
                    "data": result["data"],
                    "method": "primary",
                    "confidence": 1.0
                }
            
            # Fallback 1: Try with simplified prompt
            simple_prompt = f"Return JSON with these fields: {list(self.output_model.model_fields.keys())}"
            result = self.primary(prompt=simple_prompt)
            
            if result["success"]:
                return {
                    "data": result["data"],
                    "method": "simplified",
                    "confidence": 0.8
                }
            
            # Fallback 2: Return with defaults
            try:
                default_data = self.output_model()
                return {
                    "data": default_data,
                    "method": "defaults",
                    "confidence": 0.3
                }
            except:
                return {
                    "data": None,
                    "method": "failed",
                    "confidence": 0.0
                }
    
    # Test robust operator
    robust = RobustStructuredOperator(output_model=AnalysisResult)
    result = robust(prompt="Analyze sentiment")
    
    print(f"Robust Extraction:")
    print(f"  Method: {result['method']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    if result['data']:
        print(f"  Data: {result['data'].sentiment}")
    
    print("\n" + "="*50)
    print("✅ Key Takeaways")
    print("="*50)
    print("\n1. Always validate LLM outputs against expected schema")
    print("2. Implement retry logic for reliability")
    print("3. Use type-safe models (EmberModel) for structure")
    print("4. Provide clear schemas in prompts")
    print("5. Have fallback strategies for production")
    print("6. Consider partial extraction when full parsing fails")
    
    print("\nNext: Explore chain_of_thought.py for reasoning patterns!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())