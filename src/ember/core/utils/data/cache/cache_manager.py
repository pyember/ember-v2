"""Multi-level caching for datasets.

Provides memory and disk caching to improve performance for
frequently accessed datasets.
"""

import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatasetCache(Generic[T]):
    """Thread-safe LRU cache with optional disk persistence.

    Features:
    - Thread-safe operations for concurrent access
    - LRU eviction policy for memory efficiency
    - Optional disk persistence for long-term caching
    - TTL-based expiration for cache invalidation

    Memory usage is controlled through capacity limit, while
    disk cache uses TTL for expiration.
    """

    def __init__(
        self,
        capacity: int = 100,
        disk_cache_dir: Optional[str] = None,
        default_ttl: int = 3600,  # 1 hour in seconds
    ):
        """Initialize cache with configuration.

        Args:
            capacity: Maximum number of items in memory cache
            disk_cache_dir: Optional directory for disk caching
            default_ttl: Default time-to-live in seconds
        """
        self._capacity = max(1, capacity)
        self._default_ttl = default_ttl

        # Memory cache with OrderedDict for LRU semantics
        self._memory_cache: Dict[str, Dict[str, Any]] = OrderedDict()

        # Disk cache directory
        self._disk_cache_dir = disk_cache_dir
        if disk_cache_dir and not os.path.exists(disk_cache_dir):
            try:
                os.makedirs(disk_cache_dir, exist_ok=True)
            except OSError as e:
                logger.warning(f"Failed to create disk cache directory: {e}")
                self._disk_cache_dir = None

        # Thread safety
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[T]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        # Memory cache lookup (thread-safe)
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]

                # Check expiration
                if time.time() < entry.get("expiry", 0):
                    # Move to end for LRU
                    self._memory_cache.move_to_end(key)
                    return cast(T, entry.get("value"))
                else:
                    # Expired, remove from memory
                    del self._memory_cache[key]

        # Disk cache lookup
        if self._disk_cache_dir:
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, "rb") as f:
                        entry = pickle.load(f)

                        # Check expiration
                        if time.time() < entry.get("expiry", 0):
                            # Add to memory cache
                            self.set(
                                key,
                                entry.get("value"),
                                entry.get("ttl", self._default_ttl))
                            return cast(T, entry.get("value"))
                        else:
                            # Expired, remove disk file
                            try:
                                os.remove(disk_path)
                            except OSError:
                                pass
                except (pickle.PickleError, OSError) as e:
                    logger.warning(f"Failed to load from disk cache: {e}")

        return None

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
        """
        ttl_seconds = ttl if ttl is not None else self._default_ttl
        expiry = time.time() + ttl_seconds

        # Update memory cache
        with self._lock:
            # Evict if at capacity
            if (
                len(self._memory_cache) >= self._capacity
                and key not in self._memory_cache
            ):
                # Remove oldest item (first in OrderedDict)
                try:
                    self._memory_cache.popitem(last=False)
                except KeyError:
                    pass

            # Add to memory cache
            self._memory_cache[key] = {
                "value": value,
                "expiry": expiry,
                "ttl": ttl_seconds,
            }

        # Update disk cache
        if self._disk_cache_dir:
            disk_path = self._get_disk_path(key)
            try:
                with open(disk_path, "wb") as f:
                    pickle.dump(
                        {
                            "value": value,
                            "expiry": expiry,
                            "ttl": ttl_seconds,
                        },
                        f)
            except (pickle.PickleError, OSError) as e:
                logger.warning(f"Failed to write to disk cache: {e}")

    def delete(self, key: str) -> bool:
        """Delete item from cache.

        Args:
            key: Cache key

        Returns:
            True if item was deleted, False otherwise
        """
        deleted = False

        # Remove from memory cache
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
                deleted = True

        # Remove from disk cache
        if self._disk_cache_dir:
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                try:
                    os.remove(disk_path)
                    deleted = True
                except OSError as e:
                    logger.warning(f"Failed to delete from disk cache: {e}")

        return deleted

    def clear(self) -> None:
        """Clear all items from cache."""
        # Clear memory cache
        with self._lock:
            self._memory_cache.clear()

        # Clear disk cache
        if self._disk_cache_dir:
            try:
                for filename in os.listdir(self._disk_cache_dir):
                    if filename.endswith(".cache"):
                        try:
                            os.remove(os.path.join(self._disk_cache_dir, filename))
                        except OSError:
                            pass
            except OSError as e:
                logger.warning(f"Failed to clear disk cache: {e}")

    def _get_disk_path(self, key: str) -> str:
        """Get disk path for cache key.

        Args:
            key: Cache key

        Returns:
            Path to disk cache file
        """
        # Make key safe for filesystem
        safe_key = "".join(c if c.isalnum() else "_" for c in key)

        # Use cache_dir and key hash
        filename = f"{safe_key}_{hash(key)}.cache"

        return os.path.join(self._disk_cache_dir, filename)

    def exists(self, key: str) -> bool:
        """Check if key exists in cache and is not expired.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        # Check memory cache
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if time.time() < entry.get("expiry", 0):
                    return True
                else:
                    # Expired, remove
                    del self._memory_cache[key]

        # Check disk cache
        if self._disk_cache_dir:
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, "rb") as f:
                        entry = pickle.load(f)
                        return time.time() < entry.get("expiry", 0)
                except (pickle.PickleError, OSError):
                    pass

        return False


# Thread-local default cache
_thread_local = threading.local()


def get_default_cache() -> DatasetCache:
    """Get thread-local default cache.

    This provides isolated caching for each thread.

    Returns:
        Default cache for current thread
    """
    if not hasattr(_thread_local, "default_cache"):
        # Create thread-local cache
        # Get cache dir from environment or use memory-only
        cache_dir = os.environ.get("EMBER_DATA_CACHE_DIR")
        _thread_local.default_cache = DatasetCache(disk_cache_dir=cache_dir)

    return _thread_local.default_cache


def cached(
    ttl_seconds: int = 3600,
    key_fn: Optional[Callable[..., str]] = None,
    cache: Optional[DatasetCache] = None):
    """Decorator for caching function results.

    Args:
        ttl_seconds: Cache TTL in seconds
        key_fn: Optional function to generate cache key
        cache: Optional cache instance (uses default if None)

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get cache
            cache_instance = cache or get_default_cache()

            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default key based on function and arguments
                arg_key = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__module__}.{func.__name__}:{hash(arg_key)}"

            # Check cache
            cached_value = cache_instance.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Cache result
            cache_instance.set(cache_key, result, ttl=ttl_seconds)

            return result

        return wrapper

    return decorator
