"""Ember validators for data validation.

This module provides Ember-native validation decorators that abstract away
the underlying implementation details while maintaining full functionality.

Design Principles:
    1. Hide implementation details - users shouldn't know about Pydantic
    2. Provide a clean, intuitive API that follows Python conventions
    3. Enable the same validation capabilities with better ergonomics
    4. Make the common case simple, advanced cases possible

The validators are designed to work seamlessly with EmberModel types,
providing automatic validation without exposing framework internals.
"""

from typing import Callable, Optional, Type, TypeVar, Union

# Import pydantic validators internally
from pydantic import (
    field_validator as _pydantic_field_validator,
)
from pydantic import (
    model_validator as _pydantic_model_validator,
)

T = TypeVar("T")


def field_validator(
    *fields: str,
    check_fields: Optional[bool] = None,
    mode: str = "after",
) -> Callable[[Callable], Callable]:
    """Validate individual fields in an EmberModel.

    This decorator creates validators that run on specific fields during
    model initialization. The validator function receives the field value
    and should return the processed value or raise a ValueError.

    Args:
        *fields: Field names to validate
        check_fields: Whether to check that fields exist on the model
        mode: When to run validation - "before", "after", or "wrap"

    Returns:
        Decorator that registers the field validator

    Examples:
        class UserProfile(EmberModel):
            username: str
            email: str

            @field_validator("username")
            def clean_username(cls, value: str) -> str:
                # Remove whitespace and validate
                value = value.strip()
                if len(value) < 3:
                    raise ValueError(
                        "Username must be at least 3 characters")
                return value.lower()

            @field_validator("email")
            def validate_email(cls, value: str, info) -> str:
                # Can access other fields via info.data if needed
                if "@" not in value:
                    raise ValueError("Invalid email format")
                return value.lower()
    """
    return _pydantic_field_validator(*fields, mode=mode, check_fields=check_fields)


def model_validator(
    *,
    mode: str = "after",
) -> Callable[[Callable], Callable]:
    """Validate the entire model with cross-field validation.

    This decorator creates validators that run after all fields are set,
    allowing you to validate relationships between fields. The validator
    receives the model instance and should return it (potentially modified)
    or raise a ValueError.

    Args:
        mode: When to run validation - "after" (default) runs after all
            fields are set

    Returns:
        Decorator that registers the model validator

    Examples:
        class Transaction(EmberModel):
            amount: float
            source_account: str
            destination_account: Optional[str] = None
            transaction_type: str

            @model_validator()
            def validate_transaction(self) -> "Transaction":
                # Cross-field validation
                if self.transaction_type == "transfer" and not self.destination_account:
                    raise ValueError("Transfers require a destination account")

                if self.amount > 10000 and self.transaction_type == "withdrawal":
                    raise ValueError("Large withdrawals require additional verification")

                return self
    """

    def decorator(func: Callable) -> Callable:
        # For model validators, we want to maintain the self-based signature
        # This is more Pythonic than Pydantic's approach
        if mode == "after":
            # The function already expects self, so we can use it directly
            return _pydantic_model_validator(mode=mode)(func)
        else:
            # For other modes, we might need different transformations
            # For now, we only support "after" mode as it's the most common
            raise ValueError(f"Unsupported validation mode: {mode}")

    return decorator


def validator(
    *fields: str,
    always: bool = False,
    check_fields: Optional[bool] = None,
) -> Callable[[Callable], Callable]:
    """Create a validator for one or more fields (legacy compatibility).

    This is provided for backwards compatibility. New code should use
    field_validator or model_validator for clearer intent.

    Args:
        *fields: Field names to validate
        always: Whether to run even if the field is not provided
        check_fields: Whether to check that fields exist

    Returns:
        Decorator that registers the validator
    """
    # Redirect to field_validator for compatibility
    return field_validator(*fields, check_fields=check_fields)


# Validation helpers that provide common validation patterns
class ValidationHelpers:
    """Common validation patterns for EmberModel fields."""

    @staticmethod
    def email_validator(field_name: str = "email") -> Callable[[Type[T]], Type[T]]:
        """Create an email validator for a field.

        Args:
            field_name: Name of the field to validate. Defaults to "email".

        Returns:
            A decorator function that adds email validation to a class.

        Raises:
            ValueError: If the email format is invalid (missing @ or domain).
        """

        def decorator(cls: Type[T]) -> Type[T]:
            @field_validator(field_name)
            def validate_email(cls, value: str) -> str:
                value = value.strip().lower()
                if "@" not in value or "." not in value.split("@")[1]:
                    raise ValueError("Invalid email format")
                return value

            # Attach the validator to the class
            setattr(cls, f"_validate_{field_name}", validate_email)
            return cls

        return decorator

    @staticmethod
    def range_validator(
        field_name: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Create a range validator for numeric fields.

        Args:
            field_name: Name of the numeric field to validate.
            min_value: Minimum allowed value (inclusive). None means no minimum.
            max_value: Maximum allowed value (inclusive). None means no maximum.

        Returns:
            A decorator function that adds range validation to a class.

        Raises:
            ValueError: If value is outside the specified range.
        """

        def decorator(cls: Type[T]) -> Type[T]:
            @field_validator(field_name)
            def validate_range(cls, value: Union[int, float]) -> Union[int, float]:
                if min_value is not None and value < min_value:
                    raise ValueError(f"{field_name} must be at least {min_value}")
                if max_value is not None and value > max_value:
                    raise ValueError(f"{field_name} must be at most {max_value}")
                return value

            setattr(cls, f"_validate_{field_name}_range", validate_range)
            return cls

        return decorator

    @staticmethod
    def length_validator(
        field_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Create a length validator for string fields.

        Args:
            field_name: Name of the string field to validate.
            min_length: Minimum allowed length. None means no minimum.
            max_length: Maximum allowed length. None means no maximum.

        Returns:
            A decorator function that adds length validation to a class.

        Raises:
            ValueError: If string length is outside the specified range.
        """

        def decorator(cls: Type[T]) -> Type[T]:
            @field_validator(field_name)
            def validate_length(cls, value: str) -> str:
                if min_length is not None and len(value) < min_length:
                    raise ValueError(f"{field_name} must be at least {min_length} characters")
                if max_length is not None and len(value) > max_length:
                    raise ValueError(f"{field_name} must be at most {max_length} characters")
                return value

            setattr(cls, f"_validate_{field_name}_length", validate_length)
            return cls

        return decorator


# Export the public API
__all__ = [
    "field_validator",
    "model_validator",
    "validator",  # Legacy compatibility
    "ValidationHelpers",
]
