"""Rich Specifications - Complex input/output validation with EmberModel.

Difficulty: Intermediate
Time: ~10 minutes

Learning Objectives:
- Define rich structured inputs/outputs with EmberModel
- Use Ember validation features (constraints, validators)
- Create nested and complex data structures
- Build type-safe operators with specifications
- Handle validation errors gracefully

This showcases Ember's powerful specification system that provides:
- Zero-overhead type safety with EmberModel
- Rich validation with constraints
- Nested structures with full validation
- Custom validators for complex logic
- Excellent error messages
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Literal, Any
from datetime import datetime
from enum import Enum

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import operators
from ember.api.types import EmberModel, Field, field_validator, model_validator

# Note: All validation is now native to Ember - no external dependencies!


def main():
    """Explore Ember's rich specification capabilities."""
    print_section_header("Rich Input/Output Specifications")

    # Part 1: Basic Structured Types
    print("Part 1: Basic Structured Types with Validation")
    print("=" * 50 + "\n")

    class UserProfile(EmberModel):
        """User profile with validation constraints."""

        username: str = Field(min_length=3, max_length=20, pattern="^[a-zA-Z0-9_]+$")
        email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        age: int = Field(ge=13, le=120, description="Must be between 13 and 120")
        bio: Optional[str] = Field(None, max_length=500)
        verified: bool = Field(default=False)
        created_at: datetime = Field(default_factory=datetime.now)

    # Test validation
    print("Creating valid user profile:")
    try:
        user = UserProfile(
            username="john_doe",
            email="john@example.com",
            age=25,
            bio="Software developer",
        )
        print_example_output("Username", user.username)
        print_example_output("Email", user.email)
        print_example_output("Verified", user.verified)
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting validation (invalid username):")
    try:
        invalid_user = UserProfile(
            username="a", email="john@example.com", age=25  # Too short
        )
    except Exception as e:
        print(f"✓ Validation caught error: {str(e)[:80]}...")

    # Part 2: Nested Structures
    print("\n" + "=" * 50)
    print("Part 2: Nested Structures with Complex Validation")
    print("=" * 50 + "\n")

    class Address(EmberModel):
        """Nested address structure."""

        street: str
        city: str
        state: str = Field(min_length=2, max_length=2)
        zip_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")
        country: str = "US"

    class ContactInfo(EmberModel):
        """Contact information with multiple channels."""

        primary_email: str
        secondary_email: Optional[str] = None
        phone: Optional[str] = Field(None, pattern=r"^\+?1?\d{10,14}$")
        address: Optional[Address] = None
        preferred_contact: Literal["email", "phone", "mail"] = "email"

        @field_validator("secondary_email")
        def validate_secondary_email(cls, v, info):
            """Ensure secondary email is different from primary."""
            # Access other fields via info.data
            if v and info.data.get("primary_email") and v == info.data["primary_email"]:
                raise ValueError("Secondary email must be different from primary")
            return v

    class Customer(EmberModel):
        """Complete customer record with nested structures."""

        id: str
        profile: UserProfile
        contact: ContactInfo
        tags: List[str] = Field(default_factory=list, max_items=10)
        metadata: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)
        status: Literal["active", "inactive", "suspended"] = "active"

        @model_validator()
        def validate_customer(self):
            """Cross-field validation."""
            if self.status == "suspended" and "vip" in self.tags:
                raise ValueError("VIP customers cannot be suspended")
            return self

    # Create complex nested structure
    print("Creating nested customer structure:")
    customer = Customer(
        id="CUST-12345",
        profile=UserProfile(username="alice_smith", email="alice@company.com", age=30),
        contact=ContactInfo(
            primary_email="alice@company.com",
            phone="+14155551234",
            address=Address(
                street="123 Main St", city="San Francisco", state="CA", zip_code="94105"
            ),
        ),
        tags=["premium", "early_adopter"],
        metadata={"signup_source": "organic", "ltv": 1250.50},
    )

    print_example_output("Customer ID", customer.id)
    print_example_output("Username", customer.profile.username)
    print_example_output("City", customer.contact.address.city)
    print_example_output("Tags", customer.tags)

    # Part 3: Operators with Rich Types
    print("\n" + "=" * 50)
    print("Part 3: Operators with Rich Type Validation")
    print("=" * 50 + "\n")

    # Input specification
    class AnalysisRequest(EmberModel):
        """Request for document analysis."""

        document: str = Field(min_length=10, max_length=10000)
        analysis_type: Literal["sentiment", "summary", "entities", "all"] = "all"
        language: str = Field(default="en", pattern="^[a-z]{2}$")
        options: Dict[str, Any] = Field(default_factory=dict)

        @field_validator("document")
        def clean_document(cls, v):
            """Clean and validate document."""
            # Remove excessive whitespace
            v = " ".join(v.split())
            if len(v.split()) < 5:
                raise ValueError("Document must contain at least 5 words")
            return v

    # Output specification
    class AnalysisResult(EmberModel):
        """Structured analysis results."""

        request_id: str
        timestamp: datetime
        document_stats: Dict[str, int]

        # Conditional fields based on analysis type
        sentiment: Optional[Dict[str, float]] = None
        summary: Optional[str] = Field(None, max_length=500)
        entities: Optional[List[Dict[str, str]]] = None

        confidence: float = Field(ge=0.0, le=1.0)
        processing_time_ms: int = Field(ge=0)

        @model_validator()
        def validate_results(self):
            """Ensure at least one analysis result is present."""
            if not any([self.sentiment, self.summary, self.entities]):
                raise ValueError("At least one analysis result must be present")
            return self

    # Modern Ember pattern: Just use EmberModel types directly
    # No need for Specification classes anymore!

    # Create operator with rich types
    @operators.op
    def analyze_document(inputs: AnalysisRequest) -> AnalysisResult:
        """Analyze document with automatic validation through types."""
        import time

        start_time = time.time()

        # Simulate analysis
        doc_stats = {
            "characters": len(inputs.document),
            "words": len(inputs.document.split()),
            "sentences": inputs.document.count(".")
            + inputs.document.count("!")
            + inputs.document.count("?"),
        }

        # Build results based on analysis type
        result_data = {
            "request_id": f"REQ-{hash(inputs.document) % 10000:04d}",
            "timestamp": datetime.now(),
            "document_stats": doc_stats,
            "confidence": 0.85,
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }

        if inputs.analysis_type in ["sentiment", "all"]:
            result_data["sentiment"] = {
                "positive": 0.6,
                "negative": 0.1,
                "neutral": 0.3,
            }

        if inputs.analysis_type in ["summary", "all"]:
            result_data["summary"] = (
                f"Summary of document about {inputs.document.split()[0]}..."
            )

        if inputs.analysis_type in ["entities", "all"]:
            result_data["entities"] = [
                {"text": "Example Entity", "type": "ORGANIZATION"},
                {"text": "John Doe", "type": "PERSON"},
            ]

        return AnalysisResult(**result_data)

    # Use the operator directly - no class needed!

    print("Document analysis with rich specifications:")
    request = AnalysisRequest(
        document="This is a sample document about Ember's powerful specification system. It provides type safety and validation.",
        analysis_type="all",
    )

    result = analyze_document(request)
    print_example_output("Request ID", result.request_id)
    print_example_output("Word count", result.document_stats["words"])
    print_example_output("Sentiment", result.sentiment)
    print_example_output("Confidence", f"{result.confidence:.1%}")

    # Part 4: Advanced Validation Patterns
    print("\n" + "=" * 50)
    print("Part 4: Advanced Validation Patterns")
    print("=" * 50 + "\n")

    class FinancialTransaction(EmberModel):
        """Complex financial transaction with business rules."""

        id: str
        type: Literal["deposit", "withdrawal", "transfer"]
        amount: float = Field(gt=0, le=1_000_000)
        currency: str = Field(pattern="^[A-Z]{3}$")

        source_account: str
        destination_account: Optional[str] = None

        metadata: Dict[str, Any] = Field(default_factory=dict)
        tags: List[str] = Field(default_factory=list, max_items=5)

        created_at: datetime = Field(default_factory=datetime.now)
        status: Literal["pending", "completed", "failed"] = "pending"

        @field_validator("amount")
        def validate_amount_precision(cls, v):
            """Ensure amount has at most 2 decimal places."""
            if round(v, 2) != v:
                raise ValueError("Amount must have at most 2 decimal places")
            return v

        @model_validator()
        def validate_transaction_logic(self):
            """Complex business rule validation."""
            # Transfers must have destination
            if self.type == "transfer" and not self.destination_account:
                raise ValueError("Transfer must have destination account")

            # Deposits don't need destination
            if self.type == "deposit" and self.destination_account:
                raise ValueError("Deposits should not have destination account")

            # Large transactions need approval tag
            if self.amount > 10_000 and "approved" not in self.tags:
                raise ValueError("Transactions over $10,000 require approval tag")

            return self

    print("Testing complex validation rules:")

    # Valid transaction
    try:
        txn = FinancialTransaction(
            id="TXN-001",
            type="transfer",
            amount=5000.00,
            currency="USD",
            source_account="ACC-123",
            destination_account="ACC-456",
            tags=["international"],
        )
        print("✓ Valid transaction created")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Invalid transaction (missing approval)
    try:
        large_txn = FinancialTransaction(
            id="TXN-002",
            type="withdrawal",
            amount=50_000.00,
            currency="USD",
            source_account="ACC-789",
        )
    except Exception as e:
        print(f"✓ Validation caught: {str(e)[:60]}...")

    # Part 5: Error Handling and User Feedback
    print("\n" + "=" * 50)
    print("Part 5: Rich Error Messages")
    print("=" * 50 + "\n")

    print("EmberModel provides detailed validation errors:")

    try:
        bad_customer = Customer(
            id="",  # Empty ID
            profile=UserProfile(
                username="x",  # Too short
                email="not-an-email",  # Invalid format
                age=200,  # Too old
            ),
            contact=ContactInfo(primary_email="also-not-an-email"),
        )
    except Exception as e:
        print("Validation errors (truncated):")
        error_lines = str(e).split("\n")[:8]
        for line in error_lines:
            print(f"  {line}")
        print("  ...")

    # Summary
    print("\n" + "=" * 50)
    print("✅ Rich Type Validation Summary")
    print("=" * 50)

    print("\n🎯 Key Capabilities:")
    print("  • Type-safe structures with EmberModel")
    print("  • Field-level constraints (min/max, patterns, etc.)")
    print("  • Cross-field validation with @model_validator")
    print("  • Nested structures with full validation")
    print("  • Conditional fields and union types")
    print("  • Custom validators for business logic")
    print("  • Excellent error messages with context")

    print("\n💡 Best Practices:")
    print("  1. Use EmberModel for all structured data")
    print("  2. Add constraints at the field level")
    print("  3. Use validators for complex logic")
    print("  4. Provide helpful descriptions")
    print("  5. Test edge cases thoroughly")

    print("\n📚 Example Pattern:")
    print("```python")
    print("class MyInput(EmberModel):")
    print("    text: str = Field(min_length=1, max_length=1000)")
    print("    options: Dict[str, Any] = Field(default_factory=dict)")
    print("    ")
    print("    @field_validator('text')")
    print("    def clean_text(cls, v):")
    print("        return v.strip()")
    print("")
    print("def my_operator(inputs: MyInput) -> MyOutput:")
    print("    # Automatic validation through type annotations!")
    print("    return MyOutput(...)")
    print("```")

    return 0


if __name__ == "__main__":
    sys.exit(main())
