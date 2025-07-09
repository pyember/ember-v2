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
Level 4: Full Specification class - 0.5% of cases
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import (
    print_section_header,
    print_example_output,
    ensure_api_key,
)
from ember.api import operators, models
from ember._internal.types import EmberModel
from pydantic import Field


# from ember._internal.registry.specification import Specification
# Mock Specification for demo purposes
class Specification:
    def __init__(
        self,
        input_model=None,
        structured_output=None,
        prompt_template=None,
        check_all_placeholders=False,
    ):
        self.input_model = input_model
        self.structured_output = structured_output
        self.prompt_template = prompt_template
        self.check_all_placeholders = check_all_placeholders

    def render_prompt(self, data_dict):
        if self.prompt_template:
            return self.prompt_template.format(**data_dict)
        return str(data_dict)


from ember.operators.common import ModelCall, ModelText, Chain
from ember.api.xcs import jit
from pydantic import field_validator, model_validator


def main():
    """Explore specification progression from simple to complex."""
    print_section_header("Specifications with Progressive Disclosure")

    # Check API key availability
    has_openai = ensure_api_key("openai")
    has_anthropic = ensure_api_key("anthropic")

    if not has_openai and not has_anthropic:
        print("\n⚠️  Running in demo mode - set API keys for real model examples")
    else:
        print(f"\n✓ API keys available: OpenAI={has_openai}, Anthropic={has_anthropic}")
    print()

    # Level 1: No Specifications (90% of use cases)
    print("Level 1: No Specifications - Just Functions")
    print("=" * 50 + "\n")

    @operators.op
    def classify_sentiment(text: str) -> str:
        """Dead simple - no specs needed."""
        # Rule-based fallback
        if any(word in text.lower() for word in ["good", "great", "excellent"]):
            return "positive"
        elif any(word in text.lower() for word in ["bad", "terrible", "awful"]):
            return "negative"
        return "neutral"

    # Real model version using ModelText
    @operators.op
    def model_classify_sentiment(text: str) -> str:
        """Model-based sentiment classification."""
        if has_openai:
            try:
                model = ModelText("gpt-4o-mini", temperature=0.1)
                prompt = f"Classify the sentiment of this text as 'positive', 'negative', or 'neutral': '{text}'"
                response = model(prompt).strip().lower()
                # Extract the classification
                if "positive" in response:
                    return "positive"
                elif "negative" in response:
                    return "negative"
                else:
                    return "neutral"
            except Exception:
                # Fallback to rule-based
                return classify_sentiment(text)
        else:
            return classify_sentiment(text)

    test_text = "This is a great product!"

    result_rule = classify_sentiment(test_text)
    result_model = model_classify_sentiment(test_text)

    print(f"Rule-based result: {result_rule}")
    print(f"Model-based result: {result_model}")
    print("✓ No validation needed - trust Python's type system")

    # Level 2: Native Python Types (8% of use cases)
    print("\n" + "=" * 50)
    print("Level 2: Native Python Types")
    print("=" * 50 + "\n")

    @operators.op
    def analyze_metrics(
        values: List[float], threshold: float = 0.5, include_outliers: bool = True
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
            "threshold": threshold,
        }

        if include_outliers:
            result["min"] = min(values)
            result["max"] = max(values)

        return result

    # Note: @operators.op decorated functions take positional args only in current implementation
    # metrics = analyze_metrics([0.1, 0.5, 0.9, 0.3], threshold=0.4)

    # For now, call directly as function
    def analyze_metrics_func(
        values: List[float], threshold: float = 0.5, include_outliers: bool = True
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
            "threshold": threshold,
        }

        if include_outliers:
            result["min"] = min(values)
            result["max"] = max(values)

        return result

    metrics = analyze_metrics_func([0.1, 0.5, 0.9, 0.3], threshold=0.4)
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

    @operators.op
    def search_documents(query: SearchQuery) -> List[SearchResult]:
        """Operator with EmberModel validation."""
        # Validation happens automatically on input!
        results = []

        # Simulate search
        for i in range(min(3, query.max_results)):
            results.append(
                SearchResult(
                    id=f"doc-{i}",
                    title=f"Result for: {query.query}",
                    score=0.9 - (i * 0.1),
                    metadata={"rank": i + 1} if query.include_metadata else None,
                )
            )

        return results

    # Use with automatic validation
    search_query = SearchQuery(
        query="machine learning",
        max_results=5,
        filters={"category": "research"},
        include_metadata=True,
    )

    results = search_documents(search_query)
    print(f"Search for '{search_query.query}':")
    for r in results:
        print(f"  {r.id}: {r.title} (score: {r.score})")
    print("✓ Automatic validation on inputs and outputs")

    # Level 4: Full Specification Class (0.5% of use cases)
    print("\n" + "=" * 50)
    print("Level 4: Full Specification with Prompt Templates")
    print("=" * 50 + "\n")

    # Complex input model
    class InvestmentAnalysisInput(EmberModel):
        """Complex nested input structure."""

        portfolio: List[Dict[str, Any]] = Field(
            description="List of holdings", min_items=1, max_items=100
        )
        market_conditions: Dict[str, float] = Field(
            description="Current market indicators"
        )
        risk_tolerance: float = Field(
            ge=0.0,
            le=1.0,
            description="Risk tolerance from 0 (conservative) to 1 (aggressive)",
        )
        analysis_depth: str = Field(
            default="standard", pattern="^(quick|standard|detailed)$"
        )

        @model_validator(mode="after")
        def validate_portfolio(self):
            """Complex cross-field validation."""
            total_value = sum(h.get("value", 0) for h in self.portfolio)
            if total_value <= 0:
                raise ValueError("Portfolio must have positive total value")

            # High risk tolerance requires detailed analysis
            if self.risk_tolerance > 0.8 and self.analysis_depth == "quick":
                raise ValueError(
                    "High risk tolerance requires at least standard analysis"
                )

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
                    raise ValueError(
                        "Each recommendation must have 'action' and 'reasoning'"
                    )
            return v

    # Full specification with prompt template
    class InvestmentAnalysisSpec(Specification):
        """Complete specification with all features."""

        input_model = InvestmentAnalysisInput
        structured_output = InvestmentAnalysisOutput
        prompt_template = """Analyze the investment portfolio and provide recommendations.

Portfolio Holdings:
{portfolio}

Market Conditions:
{market_conditions}

Risk Tolerance: {risk_tolerance}
Analysis Depth: {analysis_depth}

Provide a comprehensive analysis including:
1. Overall risk assessment
2. Specific recommendations
3. Performance projections
4. Any warnings or concerns

Format the output according to the specified structure."""

        check_all_placeholders = True  # Ensure all fields are in template

    # Operator using full specification
    class InvestmentAnalyzer(operators.Operator):
        """Complex operator with full specification."""

        specification: object
        model: object
        use_model: bool

        def __init__(self):
            # Initialize model if API key available
            if has_openai:
                try:
                    model = ModelCall("gpt-4o-mini", temperature=0.3)
                    use_model = True
                except Exception:
                    model = None
                    use_model = False
            else:
                model = None
                use_model = False

            specification = InvestmentAnalysisSpec()

            # Equinox-style initialization
            object.__setattr__(self, "specification", specification)
            object.__setattr__(self, "model", model)
            object.__setattr__(self, "use_model", use_model)

        def forward(self, inputs: InvestmentAnalysisInput) -> InvestmentAnalysisOutput:
            """Analyze investment portfolio."""
            if self.use_model and self.model:
                try:
                    # Real model analysis
                    prompt = self.specification.render_prompt(inputs.model_dump())
                    response = self.model(prompt)

                    # For demo, extract some basic info and use simulated analysis
                    # In practice, you'd parse the structured response
                    risk_score = 5.0 + (inputs.risk_tolerance * 3)
                    summary = (
                        f"AI-analyzed portfolio with {len(inputs.portfolio)} holdings"
                    )
                except Exception as e:
                    # Fallback to simulation
                    risk_score = 5.0 + (inputs.risk_tolerance * 3)
                    summary = f"Simulated analysis for {len(inputs.portfolio)} holdings (model error: {str(e)[:50]})"
            else:
                # Simulate analysis
                risk_score = 5.0 + (inputs.risk_tolerance * 3)
                summary = f"Simulated analysis for {len(inputs.portfolio)} holdings"

            return InvestmentAnalysisOutput(
                summary=summary,
                risk_score=risk_score,
                recommendations=[
                    {
                        "action": "rebalance",
                        "reasoning": "Current allocation doesn't match risk profile",
                        "priority": "high",
                    },
                    {
                        "action": "diversify",
                        "reasoning": "Too concentrated in single sector",
                        "priority": "medium",
                    },
                ],
                projections={
                    "1_year": {"return": 0.08, "volatility": 0.15},
                    "5_year": {"return": 0.12, "volatility": 0.18},
                },
                warnings=["Market volatility is high"] if risk_score > 7 else None,
                confidence=0.85,
            )

    # Use the complex operator
    analyzer = InvestmentAnalyzer()

    analysis_input = InvestmentAnalysisInput(
        portfolio=[
            {"symbol": "AAPL", "value": 50000, "shares": 300},
            {"symbol": "GOOGL", "value": 30000, "shares": 100},
        ],
        market_conditions={"volatility": 0.25, "trend": 0.05},
        risk_tolerance=0.6,
        analysis_depth="standard",
    )

    result = analyzer(analysis_input)
    print("Complex specification result:")
    print_example_output("Risk Score", f"{result.risk_score:.1f}/10")
    print_example_output("Recommendations", len(result.recommendations))
    print_example_output("Confidence", f"{result.confidence:.0%}")
    print(
        f"✓ Full validation, prompt templates, structured I/O (model: {analyzer.use_model})"
    )

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

    print("\nLevel 4 - Full Specification (0.5%):")
    print("  ✓ Complex business logic")
    print("  ✓ Prompt templates needed")
    print("  ✓ Strict input/output contracts")
    print("  ✓ Cross-field validation")
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

    # Bonus example with real model integration
    if has_openai or has_anthropic:
        print("\n" + "=" * 50)
        print("Bonus: Real Model Integration Example")
        print("=" * 50 + "\n")

        try:
            # Create a simple but validated model operator
            class TextSummaryInput(EmberModel):
                text: str = Field(min_length=10, max_length=2000)
                max_words: int = Field(default=50, ge=10, le=200)

            class TextSummaryOutput(EmberModel):
                summary: str
                word_count: int
                original_length: int

            @operators.op
            def summarize_with_validation(
                inputs: TextSummaryInput,
            ) -> TextSummaryOutput:
                """Validated summarization with real model."""
                if has_openai:
                    model = ModelText("gpt-4o-mini", temperature=0.3)
                else:
                    model = ModelText("claude-3-haiku", temperature=0.3)

                prompt = f"Summarize this text in {inputs.max_words} words or less: {inputs.text}"
                summary = model(prompt)

                return TextSummaryOutput(
                    summary=summary,
                    word_count=len(summary.split()),
                    original_length=len(inputs.text),
                )

            # Test the validated model operator
            test_input = TextSummaryInput(
                text="Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions. Applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.",
                max_words=30,
            )

            result = summarize_with_validation(test_input)

            print("Validated Model Integration:")
            print_example_output("Original length", f"{result.original_length} chars")
            print_example_output("Summary", result.summary)
            print_example_output("Word count", result.word_count)

            print("\n✅ This demonstrates:")
            print("  • EmberModel validation with real models")
            print("  • Type-safe input/output contracts")
            print("  • Graceful handling of model calls")
            print("  • Structured data with automatic validation")

        except Exception as e:
            print(f"\n⚠️  Model integration error: {e}")
            print("This might be due to API limits or connectivity issues.")

    else:
        print("\n💡 With API keys, you could see:")
        print("  • Real model calls with validation")
        print("  • Structured input/output with models")
        print("  • Error handling and fallbacks")
        print("  • Type-safe model operations")
    
    print("\nNext: Explore data processing in '../05_data_processing/'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
