"""Patch for pyasn1 compatibility issues.

This module patches pyasn1 to fix the SingleValueConstraint tuple concatenation
error that occurs with certain versions of google-generativeai and its dependencies.

This is a targeted fix for the specific error:
"TypeError: can only concatenate tuple (not "SingleValueConstraint") to tuple"

Architecture Notes:
    This module implements a surgical monkey patch to fix a version incompatibility
    between pyasn1 and google-generativeai. The issue arises from pyasn1's
    SingleValueConstraint class not properly implementing tuple concatenation
    operators, causing failures when google-generativeai tries to define ASN.1
    schema constraints.

Design Rationale:
    Why patch instead of fixing upstream?
    1. **Immediate Resolution**: Users need working code now, not in 6 months
    2. **Version Constraints**: Many dependencies pin specific pyasn1 versions
    3. **Minimal Risk**: Patch only affects specific failing operations
    4. **Auto-Removal**: Can be removed when upstream fixes propagate

    Why use import hooks?
    - Ensures patch applies before any code uses pyasn1
    - Works regardless of import order in user code
    - Zero performance impact after first import
    - Transparent to all downstream code

Implementation Strategy:
    1. Subclass SingleValueConstraint with proper __add__/__radd__ methods
    2. Replace class in constraint module before any usage
    3. Fix Boolean class that triggers the error
    4. Use import hooks for reliable early patching

Trade-offs:
    - We do realize that monkey patching is generally discouraged but it's justified here by:
      * Blocking bug preventing basic functionality
      * Surgical precision of the fix
      * Spec of clear removal criteria (upstream fix)
    - Import hook adds some complexity but ensures reliability
    - Debug logging instead of failures maintains stability

Removal Criteria:
    This patch should be removed when:
    1. pyasn1 >= 0.6.0 includes the fix
    2. google-generativeai updates its constraints
    3. Minimum 6 months have passed for ecosystem updates
"""

import sys


def patch_pyasn1():
    """Apply compatibility patches to pyasn1 before it's used by other modules."""
    try:
        # Import pyasn1 modules that need patching
        from pyasn1.type import constraint, univ

        # Store original SingleValueConstraint class
        _original_svc = constraint.SingleValueConstraint

        class PatchedSingleValueConstraint(_original_svc):
            """Patched SingleValueConstraint that properly handles tuple operations."""

            def __add__(self, other):
                """Override addition to handle tuple concatenation properly."""
                if isinstance(other, tuple):
                    # Convert self to tuple for concatenation
                    return self._values + other
                elif isinstance(other, PatchedSingleValueConstraint):
                    # Combine values from both constraints
                    return PatchedSingleValueConstraint(*(self._values + other._values))
                else:
                    # Fallback to original behavior
                    return super().__add__(other)

            def __radd__(self, other):
                """Override reverse addition for tuple + constraint operations."""
                if isinstance(other, tuple):
                    return other + self._values
                return NotImplemented

        # Replace the original class
        constraint.SingleValueConstraint = PatchedSingleValueConstraint

        # Patch the Boolean class definition that causes the error
        if hasattr(univ, "Boolean"):
            try:
                # Create a new Boolean class with proper constraint handling
                class PatchedBoolean(univ.Integer):
                    """Patched Boolean class with fixed constraint definition."""

                    tagSet = univ.Boolean.tagSet
                    namedValues = univ.Boolean.namedValues

                    # Fix the constraint definition
                    subtypeSpec = univ.Integer.subtypeSpec + PatchedSingleValueConstraint(0, 1)

                    # Copy other attributes
                    encoding = "us-ascii"

                # Replace the original Boolean class
                univ.Boolean = PatchedBoolean
            except Exception:
                # If patching Boolean fails, continue anyway
                pass

        return True

    except Exception as e:
        # Log but don't fail - the patch is optional
        import logging

        logging.debug(f"pyasn1 patch failed (non-critical): {e}")
        return False


def ensure_pyasn1_compatibility():
    """Ensure pyasn1 is compatible with google-generativeai.

    This function should be called early in the application initialization,
    before any imports that might use pyasn1 (like google.generativeai).
    """
    # Check if pyasn1 is already imported
    if "pyasn1" in sys.modules:
        # If already imported, try to patch it anyway
        patch_pyasn1()
    else:
        # If not imported yet, set up an import hook
        import builtins

        _original_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            """Import hook that patches pyasn1 when it's first imported."""
            module = _original_import(name, *args, **kwargs)

            # When pyasn1.type.univ is imported, apply our patches
            if name == "pyasn1.type.univ" or name.startswith("pyasn1.type.univ"):
                patch_pyasn1()
                # Restore original import to avoid overhead
                builtins.__import__ = _original_import

            return module

        # Install our import hook
        builtins.__import__ = _patched_import


# Auto-apply the patch when this module is imported
ensure_pyasn1_compatibility()
