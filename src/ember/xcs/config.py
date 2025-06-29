"""Configuration for advanced XCS users.

Following Jobs' principle: Make opinionated choices, but allow override
when truly needed. No complexity exposed - just simple on/off switches.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Simple configuration for advanced XCS users.

    Most users should NOT use this. The defaults are carefully chosen
    to work well for 90% of use cases. Only use this if you have
    specific requirements.

    Examples:
        # Disable caching for sensitive data
        @jit(config=Config(cache=False))
        def process_private_data(data):
            return model(data)

        # Force profiling for performance analysis
        @jit(config=Config(profile=True))
        def expensive_operation(data):
            return complex_model(data)

        # Limit parallelism for resource-constrained environments
        @jit(config=Config(max_workers=2))
        def process_batch(items):
            return [process(item) for item in items]

    Note: This is the ONLY configuration object. We don't expose
    schedulers, strategies, or execution modes. The system makes
    those choices automatically.
    """

    # Simple on/off switches
    parallel: bool = True  # Enable parallel execution
    cache: bool = True  # Enable result caching
    profile: bool = False  # Force profiling for this function

    # Resource limits
    max_workers: Optional[int] = None  # Limit parallel workers (None = auto)
    max_memory_mb: Optional[int] = None  # Memory limit in MB (None = unlimited)

    # No scheduler configuration!
    # No strategy selection!
    # No execution modes!
    # No IR manipulation!
    # Just simple, practical options.

    def __post_init__(self):
        """Validate configuration."""
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if self.max_memory_mb is not None and self.max_memory_mb < 1:
            raise ValueError("max_memory_mb must be at least 1")

        # That's it. No complex validation needed because
        # we keep the options simple.


# Presets for common scenarios (following Jobs - opinionated defaults)
class Presets:
    """Common configuration presets.

    Use these instead of creating custom configs when possible.
    """

    # For processing sensitive data
    SECURE = Config(
        cache=False,  # Don't cache sensitive data
        profile=False,  # Don't profile sensitive operations
    )

    # For debugging performance issues
    DEBUG = Config(
        profile=True,  # Always profile
        parallel=True,  # Keep parallelism for realistic timing
    )

    # For resource-constrained environments (e.g., Lambda)
    LIGHTWEIGHT = Config(
        max_workers=2,  # Limit parallelism
        cache=True,  # Cache is even more important with limited resources
    )

    # For single-threaded execution (testing, debugging)
    SERIAL = Config(parallel=False, cache=True)  # Disable all parallelism  # Keep caching


# Export only what users need
__all__ = ["Config", "Presets"]
