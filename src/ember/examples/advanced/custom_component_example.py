"""Example of creating custom components with the new context system.

This example demonstrates how to create custom components
that integrate with the context system.
"""

from typing import Any, Dict, List, Optional, Type

from ember.core.context import Component, Registry
from ember.core.context.config import ConfigComponent


class CacheComponent(Component):
    """Example custom component for caching.

    This component demonstrates how to create a custom component
    that works with the context system.
    """

    def __init__(self, registry: Optional[Registry] = None):
        """Initialize with registry.

        Args:
            registry: Registry to use (current thread's if None)
        """
        super().__init__(registry)
        self._cache: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    def _register(self) -> None:
        """Register in registry as 'cache'."""
        self._registry.register("cache", self)

    def _initialize(self) -> None:
        """Initialize cache component.

        In this case, we'll load cache configuration from the config component.
        """
        # Check if there's a config component
        config = self._registry.get("config")
        if config:
            # Get cache configuration
            cache_config = config.get_config("cache")

            # Apply any configuration options
            self._max_size = cache_config.get("max_size", 1000)
            if "initial_values" in cache_config:
                self._cache.update(cache_config["initial_values"])

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        self._ensure_initialized()

        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        else:
            self._misses += 1
            return default

    def set(self, key: str, value: Any) -> None:
        """Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._ensure_initialized()

        # Basic cache management - remove oldest if too many items
        if len(self._cache) >= self._max_size:
            # Remove first item (oldest)
            if self._cache:
                del self._cache[next(iter(self._cache))]

        self._cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_ratio": self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0
            else 0,
        }


class LoggerComponent(Component):
    """Example custom logging component.

    This component demonstrates dependency injection with
    the context system.
    """

    def __init__(self, registry: Optional[Registry] = None):
        """Initialize with registry.

        Args:
            registry: Registry to use (current thread's if None)
        """
        super().__init__(registry)
        self._logs: List[str] = []
        self._level = "INFO"

    def _register(self) -> None:
        """Register in registry as 'logger'."""
        self._registry.register("logger", self)

    def _initialize(self) -> None:
        """Initialize logger component.

        Loads configuration from config component.
        """
        # Check if there's a config component
        config = self._registry.get("config")
        if config:
            # Get logger configuration
            logger_config = config.get_config("logger")

            # Apply any configuration options
            self._level = logger_config.get("level", "INFO").upper()

    def log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level
            message: Message to log
        """
        self._ensure_initialized()

        level = level.upper()
        if self._should_log(level):
            log_entry = f"[{level}] {message}"
            self._logs.append(log_entry)
            print(log_entry)

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged at this level.

        Args:
            level: Log level to check

        Returns:
            True if message should be logged
        """
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

        message_level = levels.get(level.upper(), 0)
        current_level = levels.get(self._level, 1)

        return message_level >= current_level

    def debug(self, message: str) -> None:
        """Log debug message.

        Args:
            message: Debug message
        """
        self.log("DEBUG", message)

    def info(self, message: str) -> None:
        """Log info message.

        Args:
            message: Info message
        """
        self.log("INFO", message)

    def warning(self, message: str) -> None:
        """Log warning message.

        Args:
            message: Warning message
        """
        self.log("WARNING", message)

    def error(self, message: str) -> None:
        """Log error message.

        Args:
            message: Error message
        """
        self.log("ERROR", message)

    def critical(self, message: str) -> None:
        """Log critical message.

        Args:
            message: Critical message
        """
        self.log("CRITICAL", message)

    def get_logs(self) -> List[str]:
        """Get all logs.

        Returns:
            List of log entries
        """
        return self._logs.copy()


def demonstrate_custom_components() -> None:
    """Demonstrate custom components working together."""
    # Clear any existing registry
    Registry.clear()

    # Create configuration
    config = ConfigComponent(
        config_data={
            "cache": {"max_size": 100, "initial_values": {"greeting": "Hello, world!"}},
            "logger": {"level": "DEBUG"},
        }
    )

    # Create custom components
    cache = CacheComponent()
    logger = LoggerComponent()

    # Use logger
    logger.debug("Initializing application")
    logger.info("Application started")

    # Use cache
    greeting = cache.get("greeting")
    logger.info(f"Greeting from cache: {greeting}")

    # Add more items to cache
    cache.set("answer", 42)
    logger.debug("Added answer to cache")

    # Get cache stats
    stats = cache.get_stats()
    logger.info(f"Cache stats: {stats}")

    # Get non-existent item
    value = cache.get("nonexistent")
    logger.warning(f"Attempted to access nonexistent key, got: {value}")

    # Update stats
    stats = cache.get_stats()
    logger.info(f"Updated cache stats: {stats}")

    # Demonstrate component discovery
    registry = Registry.current()
    components = registry.keys()
    logger.info(f"Registered components: {components}")

    # Log final message
    logger.info("Custom component demonstration completed")

    # Print all logs
    print("\nAll logs:")
    for log in logger.get_logs():
        print(f"  {log}")


if __name__ == "__main__":
    demonstrate_custom_components()
