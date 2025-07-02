"""Specifications with Progressive Disclosure - From simple to complex validation.

Difficulty: Intermediate to Advanced
Time: ~10 minutes

Learning Objectives:
- See how specifications scale from simple to complex
- Understand when to use each level of validation
- Learn to balance type safety with simplicity
- Master the specification progression

This shows how Ember's specification system follows progressive disclosure:
Level 1: No specs (just functions) - 90% of cases
Level 2: Simple types (native Python) - 8% of cases  
Level 3: EmberModel validation - 1.5% of cases
Level 4: Input/Output specs on operators - 0.5% of cases
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import operators, models
from ember._internal.types import EmberModel
from ember.api.xcs import jit
from pydantic import Field, field_validator, model_validator


def main():
    """Explore specification progression from simple to complex."""
    print_section_header("Specifications with Progressive Disclosure")
    
    # Level 1: No Specifications (90% of use cases)
    print("Level 1: No Specifications - Just Functions")
    print("=" * 50 + "\n")
    
    @operators.op
    def classify_sentiment(text: str) -> str:
        """Dead simple - no specs needed."""
        # In real use: return models("gpt-4", f"Classify sentiment: {text}").text
        if any(word in text.lower() for word in ["good", "great", "excellent"]):
            return "positive"
        elif any(word in text.lower() for word in ["bad", "terrible", "awful"]):
            return "negative"
        return "neutral"
    
    result = classify_sentiment("This is a great product!")
    print(f"Simple function result: {result}")
    print("✓ No validation needed - trust Python's type system")
    
    # Level 2: Native Python Types (8% of use cases)
    print("\n" + "=" * 50)
    print("Level 2: Native Python Types")
    print("=" * 50 + "\n")
    
    def analyze_metrics(
        values: List[float],
        threshold: float = 0.5,
        include_outliers: bool = True
    ) -> Dict[str, Any]:
        """Use native Python types for simple validation."""
        if not values:
            raise ValueError("Values list cannot be empty")
        
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        # Simple analysis
        mean = sum(values) / len(values)
        above_threshold = [v for v in values if v > threshold]
        
        result = {
            "mean": mean,
            "count": len(values),
            "above_threshold": len(above_threshold),
            "threshold": threshold
        }
        
        if include_outliers:
            result["min"] = min(values)
            result["max"] = max(values)
        
        return result
    
    metrics = analyze_metrics([0.1, 0.5, 0.9, 0.3], threshold=0.4)
    print("Native types result:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("✓ Python's type hints + simple validation")
    
    # Level 3: EmberModel for Structured Data (1.5% of use cases)
    print("\n" + "=" * 50)
    print("Level 3: EmberModel for Rich Validation")
    print("=" * 50 + "\n")
    
    class SearchQuery(EmberModel):
        """When you need field-level validation."""
        query: str = Field(min_length=1, max_length=200)
        max_results: int = Field(default=10, ge=1, le=100)
        filters: Dict[str, str] = Field(default_factory=dict)
        include_metadata: bool = Field(default=False)
    
    class SearchResult(EmberModel):
        """Structured output with validation."""
        id: str
        title: str
        score: float = Field(ge=0.0, le=1.0)
        metadata: Optional[Dict[str, Any]] = None
    
    def search_documents(query: SearchQuery) -> List[SearchResult]:
        """Operator with EmberModel validation."""
        # Validation happens automatically on input!
        results = []
        
        # Simulate search
        for i in range(min(3, query.max_results)):
            results.append(SearchResult(
                id=f"doc-{i}",
                title=f"Result for: {query.query}",
                score=0.9 - (i * 0.1),
                metadata={"rank": i + 1} if query.include_metadata else None
            ))
        
        return results
    
    # Use with automatic validation
    search_query = SearchQuery(
        query="machine learning",
        max_results=5,
        filters={"category": "research"},
        include_metadata=True
    )
    
    results = search_documents(search_query)
    print(f"Search for '{search_query.query}':")
    for r in results:
        print(f"  {r.id}: {r.title} (score: {r.score})")
    print("✓ Automatic validation on inputs and outputs")
    
    # Level 4: Input/Output Specs on Operators (0.5% of use cases)
    print("\n" + "=" * 50)
    print("Level 4: Input/Output Specs on Operators")
    print("=" * 50 + "\n")
    
    # Complex input model
    class InvestmentAnalysisInput(EmberModel):
        """Complex nested input structure."""
        portfolio: List[Dict[str, Any]] = Field(
            description="List of holdings",
            min_items=1,
            max_items=100
        )
        market_conditions: Dict[str, float] = Field(
            description="Current market indicators"
        )
        risk_tolerance: float = Field(
            ge=0.0, le=1.0,
            description="Risk tolerance from 0 (conservative) to 1 (aggressive)"
        )
        analysis_depth: str = Field(
            default="standard",
            pattern="^(quick|standard|detailed)$"
        )
        
        @model_validator(mode="after")
        def validate_portfolio(self):
            """Complex cross-field validation."""
            total_value = sum(h.get("value", 0) for h in self.portfolio)
            if total_value <= 0:
                raise ValueError("Portfolio must have positive total value")
            
            # High risk tolerance requires detailed analysis
            if self.risk_tolerance > 0.8 and self.analysis_depth == "quick":
                raise ValueError("High risk tolerance requires at least standard analysis")
            
            return self
    
    # Complex output model
    class InvestmentAnalysisOutput(EmberModel):
        """Structured analysis results."""
        summary: str = Field(max_length=500)
        risk_score: float = Field(ge=0.0, le=10.0)
        recommendations: List[Dict[str, Any]] = Field(max_items=10)
        projections: Dict[str, Dict[str, float]]
        warnings: Optional[List[str]] = None
        confidence: float = Field(ge=0.0, le=1.0)
        
        @field_validator("recommendations")
        def validate_recommendations(cls, v):
            """Ensure recommendations have required fields."""
            for rec in v:
                if "action" not in rec or "reasoning" not in rec:
                    raise ValueError("Each recommendation must have 'action' and 'reasoning'")
            return v
    
    # Operator using input_spec and output_spec directly
    class InvestmentAnalyzer(operators.Operator):
        """Complex operator with input/output specifications."""
        
        # Specify input and output types directly  
        input_spec = InvestmentAnalysisInput
        output_spec = InvestmentAnalysisOutput
        
        def __init__(self):
            # Could initialize model here
            # self.model = models.instance("gpt-4", temperature=0.3)
            pass
        
        def forward(self, inputs: InvestmentAnalysisInput) -> InvestmentAnalysisOutput:
            """Analyze investment portfolio."""
            # In real use, would render prompt and call model
            # prompt = self.specification.render_prompt(inputs.dict())
            # response = self.model(prompt)
            
            # Simulate analysis
            risk_score = 5.0 + (inputs.risk_tolerance * 3)
            
            return InvestmentAnalysisOutput(
                summary=f"Portfolio analysis for {len(inputs.portfolio)} holdings",
                risk_score=risk_score,
                recommendations=[
                    {
                        "action": "rebalance",
                        "reasoning": "Current allocation doesn't match risk profile",
                        "priority": "high"
                    },
                    {
                        "action": "diversify",
                        "reasoning": "Too concentrated in single sector",
                        "priority": "medium"
                    }
                ],
                projections={
                    "1_year": {"return": 0.08, "volatility": 0.15},
                    "5_year": {"return": 0.12, "volatility": 0.18}
                },
                warnings=["Market volatility is high"] if risk_score > 7 else None,
                confidence=0.85
            )
    
    # Use the complex operator
    analyzer = InvestmentAnalyzer()
    
    analysis_input = InvestmentAnalysisInput(
        portfolio=[
            {"symbol": "AAPL", "value": 50000, "shares": 300},
            {"symbol": "GOOGL", "value": 30000, "shares": 100}
        ],
        market_conditions={"volatility": 0.25, "trend": 0.05},
        risk_tolerance=0.6,
        analysis_depth="standard"
    )
    
    result = analyzer(analysis_input)
    print("Complex specification result:")
    print_example_output("Risk Score", f"{result.risk_score:.1f}/10")
    print_example_output("Recommendations", len(result.recommendations))
    print_example_output("Confidence", f"{result.confidence:.0%}")
    print("✓ Full validation, structured I/O with operator specs")
    
    # Progression Summary
    print("\n" + "=" * 50)
    print("✅ Specification Progression Summary")
    print("=" * 50)
    
    print("\n📊 When to Use Each Level:\n")
    
    print("Level 1 - No Specs (90%):")
    print("  ✓ Simple transformations")
    print("  ✓ Trust Python's type system")
    print("  ✓ Internal functions")
    print("  Example: text → sentiment")
    
    print("\nLevel 2 - Native Types (8%):")
    print("  ✓ Need basic validation")
    print("  ✓ Simple parameter checking")
    print("  ✓ Standard Python types suffice")
    print("  Example: List[float] → Dict stats")
    
    print("\nLevel 3 - EmberModel (1.5%):")
    print("  ✓ Structured data with constraints")
    print("  ✓ Field-level validation")
    print("  ✓ Reusable data models")
    print("  Example: SearchQuery → List[SearchResult]")
    
    print("\nLevel 4 - Input/Output Specs (0.5%):")
    print("  ✓ Complex business logic")
    print("  ✓ Strict input/output contracts")
    print("  ✓ Cross-field validation")
    print("  ✓ Operator-level type enforcement")
    print("  Example: InvestmentAnalysis")
    
    print("\n🎯 Progressive Disclosure Benefits:")
    print("  1. Start simple - add validation only when needed")
    print("  2. No premature abstraction")
    print("  3. Each level is self-contained")
    print("  4. Easy to upgrade when requirements grow")
    print("  5. Zero overhead for simple cases")
    
    print("\n💡 Best Practice:")
    print("  Always start at Level 1. Only move up when you")
    print("  actually need the additional validation features.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())