"""
Runtime type checking utilities.

This module provides functions for validating types at runtime,
which can be used for testing and validation purposes.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints)

T = TypeVar("T")


def validate_type(value: Any, expected_type: Type[T]) -> bool:
    """
    Validate that a value conforms to an expected type.

    This function checks if the given value matches the expected type,
    taking into account generic types, type variables, and protocols.

    Args:
        value: The value to check
        expected_type: The type to check against

    Returns:
        True if the value matches the expected type, False otherwise
    """
    # Handle None for Optional types
    if value is None:
        args = get_args(expected_type)
        return type(None) in args if args else expected_type is type(None)

    # Get the origin type (for generics)
    origin = get_origin(expected_type)
    if origin is not None:
        # Handle Union types (including Optional which is Union[T, None])
        if origin is Union:
            return any(validate_type(value, arg) for arg in get_args(expected_type))

        # Handle generic types like List, Dict, etc.
        if origin is list:
            elem_type = get_args(expected_type)[0]
            return isinstance(value, list) and all(
                validate_type(elem, elem_type) for elem in value
            )

        elif origin is dict:
            key_type, val_type = get_args(expected_type)
            return isinstance(value, dict) and all(
                validate_type(k, key_type) and validate_type(v, val_type)
                for k, v in value.items()
            )

        elif origin is tuple:
            args = get_args(expected_type)
            if not isinstance(value, tuple) or len(value) != len(args):
                return False
            return all(validate_type(v, t) for v, t in zip(value, args))

        # For other generic types, check the origin type and then validate type arguments if possible
        if not isinstance(value, origin):
            return False

        # Check generic type parameters if available
        if hasattr(value, "__orig_class__"):
            value_type_args = get_args(value.__orig_class__)
            expected_type_args = get_args(expected_type)

            # Skip validation if we don't have enough information
            if (
                not value_type_args
                or not expected_type_args
                or len(value_type_args) != len(expected_type_args)
            ):
                return True

            # For generic type parameters, we need to compare exact types, not use validate_type
            # because type parameters themselves are types, not values
            return all(t1 == t2 for t1, t2 in zip(value_type_args, expected_type_args))

        return True

    # Handle non-generic or primitive types
    return isinstance(value, expected_type)


def validate_instance_attrs(obj: Any, cls: Type) -> Dict[str, Any]:
    """
    Validate that an object's attributes match the expected types.

    This function checks if the attributes of the given object match
    the type hints defined in its class.

    Args:
        obj: The object to check
        cls: The class to check against

    Returns:
        A dictionary of validation errors, empty if no errors
    """
    errors: Dict[str, List[str]] = {}
    try:
        type_hints = get_type_hints(cls)
    except (TypeError, AttributeError):
        # Can't get type hints for this class
        return errors

    for attr_name, expected_type in type_hints.items():
        if not hasattr(obj, attr_name):
            errors.setdefault(attr_name, []).append("Missing attribute")
            continue

        attr_value = getattr(obj, attr_name)
        try:
            # Safely try to validate the type
            if not isinstance(attr_value, expected_type):
                # For primitive types, we can use isinstance
                errors.setdefault(attr_name, []).append(
                    f"Expected {expected_type}, got {type(attr_value)}"
                )
        except TypeError:
            # For generic types, we fall back to a simple type check
            if type(attr_value).__name__ != getattr(
                expected_type, "__name__", str(expected_type)
            ):
                errors.setdefault(attr_name, []).append(
                    f"Expected {expected_type}, got {type(attr_value)}"
                )

    return errors


def type_check(obj: Any, expected_type: Optional[Type] = None) -> bool:
    """
    Check if an object matches an expected type.

    This function serves as a convenience wrapper around validate_type
    and validate_instance_attrs.

    Args:
        obj: The object to check
        expected_type: The type to check against

    Returns:
        True if the object matches the expected type, False otherwise
    """

    if expected_type is None:
        expected_type = type(obj)
    try:
        # First check the type itself, but handle potential errors
        # with generic types
        if not isinstance(obj, expected_type):
            return False
    except TypeError:
        # For generic types that can't be used with isinstance
        if not obj.__class__.__name__ == getattr(expected_type, "__name__", None):
            return False

    # Then check the attributes
    errors = validate_instance_attrs(obj, expected_type)
    if errors:
        return False

    return True
