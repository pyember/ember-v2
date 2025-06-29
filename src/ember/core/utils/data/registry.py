"""Data registry for dataset management."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ember._internal.context import EmberContext


class DataRegistry:
    """Registry for dataset loaders."""

    def __init__(self, context: Optional[EmberContext] = None):
        """Initialize registry.

        Args:
            context: Optional EmberContext.
        """
        self._context = context
        self._sources: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def load(self, name: str, **kwargs) -> Any:
        """Load dataset.

        Args:
            name: Dataset name.
            **kwargs: Dataset parameters.

        Returns:
            Dataset iterator.
        """
        # Use the data API directly for now
        from ember.api import data

        return data.stream(name, **kwargs)

    def register(self, name: str, source: Any) -> None:
        """Register data source.

        Args:
            name: Source name.
            source: Data source.
        """
        with self._lock:
            self._sources[name] = source
