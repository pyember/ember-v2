"""Module system patch to allow natural __init__ patterns.

This module patches the Module class to allow setting attributes in __init__
without requiring field annotations. This eliminates the leaky abstraction
where users need to know about Equinox's requirements.

The solution follows the principle: "Make simple things simple."
Users should write natural Python code, and the framework should adapt.
"""

from typing import Any, Dict

import equinox as eqx
import jax


class _InitializationContext:
    """Context manager for initialization mode."""

    _initializing: Dict[int, bool] = {}

    @classmethod
    def is_initializing(cls, obj_id: int) -> bool:
        return cls._initializing.get(obj_id, False)

    @classmethod
    def enter(cls, obj_id: int):
        cls._initializing[obj_id] = True

    @classmethod
    def exit(cls, obj_id: int):
        cls._initializing.pop(obj_id, None)


def _create_dynamic_field(value: Any) -> Any:
    """Create appropriate field based on value type."""
    # JAX arrays are dynamic (learnable)
    if isinstance(value, (jax.Array, jax.numpy.ndarray)):
        return value  # Dynamic by default in equinox
    # Everything else is static
    return eqx.field(static=True, default=value)


class ModuleMeta(type(eqx.Module)):
    """Enhanced metaclass that handles dynamic field creation."""

    def __call__(cls, *args, **kwargs):
        """Intercept instance creation to enable initialization mode."""
        # Create instance without calling __init__
        instance = cls.__new__(cls)

        # Enter initialization mode
        obj_id = id(instance)
        _InitializationContext.enter(obj_id)

        try:
            # Get the original __init__ from the class hierarchy
            init_func = None
            for base in cls.__mro__:
                if "__init__" in base.__dict__:
                    init_func = base.__dict__["__init__"]
                    break

            if init_func:
                # Collect attributes that will be set
                # This is a simple approach - could be enhanced with AST analysis
                instance.__dict__["_pending_fields"] = {}

                # Call original __init__
                init_func(instance, *args, **kwargs)

                # Now create the actual instance with collected fields
                if hasattr(instance, "_pending_fields"):
                    fields = instance._pending_fields
                    del instance.__dict__["_pending_fields"]

                    # Add field annotations dynamically
                    if fields:
                        # Create a new class with proper annotations
                        annotations = getattr(cls, "__annotations__", {}).copy()
                        for name, value in fields.items():
                            if name not in annotations:
                                # Infer type from value
                                annotations[name] = type(value)

                        # Create dynamic class with annotations
                        dynamic_cls = type(cls.__name__, (cls,), {"__annotations__": annotations})

                        # Create proper instance with equinox
                        instance = object.__new__(dynamic_cls)
                        eqx.Module.__init__(instance)

                        # Set all fields properly
                        for name, value in fields.items():
                            object.__setattr__(instance, name, value)

            return instance

        finally:
            _InitializationContext.exit(obj_id)


class Module(eqx.Module):
    """Enhanced Module that allows natural __init__ patterns."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to allow setting during initialization."""
        obj_id = id(self)

        # During initialization, collect fields instead of setting them
        if _InitializationContext.is_initializing(obj_id):
            if hasattr(self, "_pending_fields"):
                self._pending_fields[name] = value
            else:
                # Fallback to normal behavior
                super().__setattr__(name, value)
        else:
            # Normal operation - use equinox's setattr
            super().__setattr__(name, value)


def patch_module_class():
    """Monkey-patch the existing Module class to support natural init patterns.

    This is a pragmatic solution that modifies the Module class in-place
    to support setting attributes in __init__ without annotations.
    """
    from ember._internal import module as ember_module

    # Save original Module class
    OriginalModule = ember_module.Module

    class PatchedModule(OriginalModule):
        """Patched Module class that allows natural __init__ patterns."""

        def __init_subclass__(cls, **kwargs):
            """Hook into subclass creation to enable proper initialization."""
            super().__init_subclass__(**kwargs)

            # Wrap the __init__ method if it exists
            if hasattr(cls, "__init__"):
                original_init = cls.__init__

                def wrapped_init(self, *args, **kwargs):
                    # Temporarily enable attribute setting
                    temp_annotations = {}

                    # Create a temporary setattr that collects attributes
                    original_setattr = self.__class__.__setattr__
                    pending_attrs = {}

                    def collecting_setattr(obj, name, value):
                        if name.startswith("_"):
                            # Private attributes bypass collection
                            object.__setattr__(obj, name, value)
                        else:
                            pending_attrs[name] = value
                            # Infer type annotation
                            if isinstance(value, (jax.Array, jax.numpy.ndarray)):
                                temp_annotations[name] = jax.Array
                            else:
                                temp_annotations[name] = type(value)

                    # Temporarily replace setattr
                    self.__class__.__setattr__ = collecting_setattr

                    try:
                        # Call original init to collect attributes
                        original_init(self, *args, **kwargs)

                        # Add collected annotations to class
                        if temp_annotations:
                            if not hasattr(self.__class__, "__annotations__"):
                                self.__class__.__annotations__ = {}
                            self.__class__.__annotations__.update(temp_annotations)

                        # Restore setattr
                        self.__class__.__setattr__ = original_setattr

                        # Now set all collected attributes properly
                        for name, value in pending_attrs.items():
                            # Set through equinox's mechanism
                            if isinstance(value, (jax.Array, jax.numpy.ndarray)):
                                # Dynamic field
                                object.__setattr__(self, name, value)
                            else:
                                # Static field
                                object.__setattr__(
                                    self, name, eqx.field(static=True, default=value)
                                )

                    finally:
                        # Ensure setattr is restored
                        self.__class__.__setattr__ = original_setattr

                cls.__init__ = wrapped_init

    # Replace the Module class
    ember_module.Module = PatchedModule

    # Also update it in the module's __all__
    if hasattr(ember_module, "__all__") and "Module" in ember_module.__all__:
        # Force reload of the symbol
        ember_module.__dict__["Module"] = PatchedModule

    return PatchedModule


# Auto-patch on import
patch_module_class()
