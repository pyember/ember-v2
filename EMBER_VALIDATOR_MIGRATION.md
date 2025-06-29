# Ember Validator Migration Guide

This guide helps you migrate from direct Pydantic usage to Ember's native validation API.

## Overview

Ember now provides native validation decorators that hide implementation details while maintaining full functionality. This creates a cleaner, more maintainable API that follows Python conventions.

## Key Changes

### 1. Import Changes

**Before:**
```python
from pydantic import field_validator, model_validator
from ember.api.types import EmberModel, Field
```

**After:**
```python
from ember.api.types import EmberModel, Field, field_validator, model_validator
# Or import validators separately:
from ember.api import validators
```

### 2. Field Validator Signature

Field validators in Ember maintain Pydantic compatibility while providing clear documentation:

```python
@field_validator("username")
def validate_username(cls, v):
    # Simple validation
    return v.lower()

# For cross-field validation, use info parameter:
@field_validator("secondary_email")
def validate_secondary(cls, v, info):
    # Access other fields via info.data
    if v and info.data.get("primary_email") == v:
        raise ValueError("Must be different from primary")
    return v
```

### 3. Model Validator

**Before:**
```python
@model_validator(mode="after")
def validate_model(self):
    # Validation logic
    return self
```

**After:**
```python
@model_validator()  # mode="after" is the default
def validate_model(self):
    # Same validation logic
    return self
```

## Complete Example

```python
from ember.api.types import EmberModel, Field, field_validator, model_validator
from typing import Optional

class UserProfile(EmberModel):
    """User profile with Ember-native validation."""
    username: str = Field(min_length=3, max_length=20)
    email: str
    age: int = Field(ge=13, le=120)
    bio: Optional[str] = None
    
    @field_validator("username")
    def clean_username(cls, value: str) -> str:
        """Normalize username."""
        value = value.strip().lower()
        if not value.replace("_", "").isalnum():
            raise ValueError("Username must be alphanumeric")
        return value
    
    @field_validator("email")
    def validate_email(cls, value: str) -> str:
        """Basic email validation."""
        value = value.strip().lower()
        if "@" not in value:
            raise ValueError("Invalid email format")
        return value
    
    @model_validator()
    def validate_profile(self) -> "UserProfile":
        """Cross-field validation."""
        if self.username == self.email.split("@")[0]:
            raise ValueError("Username should be different from email prefix")
        
        if self.bio and len(self.bio) < 10:
            raise ValueError("Bio must be at least 10 characters if provided")
            
        return self
```

## Validation Helpers

Ember also provides common validation patterns:

```python
from ember.api.validators import ValidationHelpers

@ValidationHelpers.email_validator("contact_email")
@ValidationHelpers.range_validator("age", min_value=18, max_value=100)
@ValidationHelpers.length_validator("bio", min_length=10, max_length=500)
class User(EmberModel):
    contact_email: str
    age: int
    bio: str
```

## Benefits of Ember Validators

1. **Cleaner API**: More Pythonic signatures without framework-specific parameters
2. **Future-proof**: Implementation can change without breaking user code
3. **Better IDE support**: Simplified signatures improve autocomplete
4. **Consistent style**: Follows Ember's design principles

## Advanced Usage

For complex validation scenarios, you can still access all functionality:

```python
# Custom validator with multiple fields
@field_validator("price", "discount_price")
def validate_prices(cls, v):
    if v < 0:
        raise ValueError("Price cannot be negative")
    return round(v, 2)

# Conditional validation
@model_validator()
def validate_conditionally(self):
    if self.status == "published" and not self.content:
        raise ValueError("Published items must have content")
    return self
```

## Migration Checklist

1. ✅ Update imports to use Ember's validators
2. ✅ Use `cls, v` signature for simple field validators
3. ✅ Use `cls, v, info` when you need access to other fields
4. ✅ Remove `mode="after"` from model validators (it's the default)
5. ✅ Move complex cross-field validation to model validators
6. ✅ Test that validation still works as expected

## Summary

Ember's validation API provides the same power as Pydantic with a cleaner, more maintainable interface. The migration is straightforward and results in code that's easier to understand and maintain.