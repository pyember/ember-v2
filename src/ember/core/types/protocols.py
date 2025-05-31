"""Core protocols for Ember's type system.

Defines minimal interface contracts following SOLID principles.
Each protocol has a single responsibility with focused interfaces.
"""

from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable)


class TypeInfo:
    """Runtime type metadata container."""

    def __init__(
        self,
        origin_type: Type,
        type_args: Optional[tuple] = None,
        is_optional: bool = False,
        is_container: bool = False) -> None:
        """Initialize TypeInfo.

        Args:
            origin_type: Base type (e.g., dict, list, str).
            type_args: Type arguments for generics.
            is_optional: Whether type is Optional.
            is_container: Whether type is a container.
        """
        self.origin_type = origin_type
        self.type_args = type_args or ()
        self.is_optional = is_optional
        self.is_container = is_container


T = TypeVar("T")


@runtime_checkable
class TypedProtocol(Protocol):
    """Protocol for objects with inspectable type information."""

    def get_type_info(self) -> TypeInfo:
        """Return type metadata."""
        ...


@runtime_checkable
class Serializable(Protocol, Generic[T]):
    """Protocol for serialization and deserialization."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...

    def to_json(self) -> str:
        """Convert to JSON string."""
        ...

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create instance from dictionary."""
        ...

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create instance from JSON string."""
        ...


# Legacy aliases for backward compatibility
EmberTyped = TypedProtocol
EmberSerializable = Serializable
