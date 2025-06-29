"""Specifications with Progressive Disclosure - Clean Version.

This version demonstrates the same progressive disclosure pattern
but without any pydantic validators leaking through.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from _shared.conditional_execution import conditional_llm
from ember.api import operators, models
from ember.api.types import EmberModel, Field


@conditional_llm()
def main(_simulated_mode=False):
    """Explore specification progression from simple to complex."""
    print_section_header("Specifications with Progressive Disclosure (Clean)")

    # Level 1: No Specifications
    print("Level 1: No Specifications - Just Functions")
    print("=" * 50 + "\n")

    @operators.op
    def classify_sentiment(text: str) -> str:
        """Dead simple - no specs needed."""
        if _simulated_mode:
            if "good" in text.lower() or "great" in text.lower():
                return "positive"
            return "neutral"
        else:
            return models("gpt-4", f"Classify sentiment: {text}").text

    result = classify_sentiment("This is a great product!")
    print(f"Simple function result: {result}")

    # Level 2: EmberModel with Defaults
    print("\n" + "=" * 50)
    print("Level 2: EmberModel with Simple Defaults")
    print("=" * 50 + "\n")

    class SearchQuery(EmberModel):
        """Simple model with defaults - no Field needed!"""

        query: str
        max_results: int = 10
        include_metadata: bool = False

    @operators.op
    def search(query: SearchQuery) -> List[str]:
        """Search with simple validation."""
        results = []
        for i in range(query.max_results):
            results.append(f"Result {i+1} for: {query.query}")
        return results

    search_query = SearchQuery(query="machine learning")
    results = search(search_query)
    print(f"Found {len(results)} results")

    # Level 3: EmberModel with Field Constraints
    print("\n" + "=" * 50)
    print("Level 3: EmberModel with Field Constraints")
    print("=" * 50 + "\n")

    class AdvancedQuery(EmberModel):
        """Model with validation constraints using Field."""

        query: str = Field(min_length=1, max_length=200)
        max_results: int = Field(default=10, ge=1, le=100)
        score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Custom validation can be done in __init__ or as a method
    class ValidatedQuery(EmberModel):
        """Model with custom validation logic."""

        query: str
        filters: Dict[str, Any] = {}

        def __init__(self, **data):
            super().__init__(**data)
            # Custom validation after initialization
            if not self.query.strip():
                raise ValueError("Query cannot be empty")
            if len(self.filters) > 10:
                raise ValueError("Too many filters")

    # Test the models
    try:
        q1 = AdvancedQuery(query="AI", max_results=50)
        print(f"✓ Created query: {q1.query} (max: {q1.max_results})")

        q2 = ValidatedQuery(query="  ", filters={})  # This will fail
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")

    # Level 4: Complex Business Logic
    print("\n" + "=" * 50)
    print("Level 4: Complex Models with Business Logic")
    print("=" * 50 + "\n")

    class InvestmentPortfolio(EmberModel):
        """Complex model with business logic."""

        holdings: List[Dict[str, float]]
        risk_tolerance: float = Field(ge=0.0, le=1.0)

        @property
        def total_value(self) -> float:
            """Computed property for total portfolio value."""
            return sum(h.get("value", 0) for h in self.holdings)

        def validate_allocation(self) -> bool:
            """Business logic validation."""
            if self.total_value <= 0:
                raise ValueError("Portfolio must have positive value")

            # Check concentration risk
            for holding in self.holdings:
                weight = holding.get("value", 0) / self.total_value
                if weight > 0.5 and self.risk_tolerance < 0.8:
                    raise ValueError("Too concentrated for risk tolerance")

            return True

    # Use the complex model
    portfolio = InvestmentPortfolio(
        holdings=[
            {"symbol": "AAPL", "value": 50000},
            {"symbol": "GOOGL", "value": 30000},
        ],
        risk_tolerance=0.6,
    )

    print(f"Portfolio value: ${portfolio.total_value:,.2f}")
    print(f"Risk tolerance: {portfolio.risk_tolerance}")

    try:
        portfolio.validate_allocation()
        print("✓ Portfolio allocation is valid")
    except ValueError as e:
        print(f"✗ Portfolio validation failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("✅ Clean Specification Patterns")
    print("=" * 50)

    print("\n📊 Progressive Disclosure:")
    print("1. Start with plain functions (no validation)")
    print("2. Add EmberModel with simple defaults")
    print("3. Use Field for constraints when needed")
    print("4. Add methods for complex business logic")

    print("\n🎯 Key Benefits:")
    print("- No pydantic imports needed")
    print("- Clean, simple progression")
    print("- Validation only when required")
    print("- Business logic stays in your models")

    return 0


if __name__ == "__main__":
    sys.exit(main())
