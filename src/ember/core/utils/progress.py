"""Progress reporting for long-running operations.

Clean status indicators with emoji support and quiet mode for CI.
"""

import sys
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import time


class ProgressReporter:
    """Progress reporter for operations.

    Example:
        >>> reporter = ProgressReporter(quiet=False)
        >>> reporter.discovery_start()
        >>> reporter.discovery_complete(25)
    """
    
    def __init__(self, quiet: bool = False, use_emoji: bool = True):
        """Initialize progress reporter.

        Args:
            quiet: If True, suppress all output
            use_emoji: If True, use emoji indicators (auto-detects support)
        """
        self.quiet = quiet or os.environ.get("EMBER_QUIET", "").lower() in ("1", "true", "yes")
        self.use_emoji = use_emoji and self._supports_emoji()
        self._start_times: Dict[str, float] = {}
    
    def _supports_emoji(self) -> bool:
        """Check if the terminal supports emoji."""
        # Disable emoji in CI environments
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            return False
        
        # Check if stdout is a TTY
        if not sys.stdout.isatty():
            return False
        
        # Check terminal encoding
        try:
            encoding = sys.stdout.encoding or "ascii"
            "🔍".encode(encoding)
            return True
        except (UnicodeEncodeError, AttributeError):
            return False
    
    def _print(self, message: str, end: str = "\n") -> None:
        """Print a message if not in quiet mode."""
        if not self.quiet:
            print(message, end=end, flush=True)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
    
    # Model Discovery Operations
    
    def discovery_start(self) -> None:
        """Signal the start of model discovery."""
        self._start_times["discovery"] = time.time()
        if self.use_emoji:
            self._print("🔍 Discovering available models...")
        else:
            self._print("Discovering available models...")
    
    def discovery_complete(self, count: int) -> None:
        """Signal completion of model discovery."""
        duration = time.time() - self._start_times.get("discovery", time.time())
        if self.use_emoji:
            self._print(f"✅ Found {count} models ({self._format_duration(duration)})")
        else:
            self._print(f"Found {count} models ({self._format_duration(duration)})")
    
    def discovery_error(self, error: str) -> None:
        """Signal an error during model discovery."""
        if self.use_emoji:
            self._print(f"❌ Model discovery failed: {error}")
        else:
            self._print(f"ERROR: Model discovery failed: {error}")
    
    # Data Loading Operations
    
    def loading_start(self, dataset_name: str) -> None:
        """Signal the start of dataset loading."""
        self._start_times[f"loading_{dataset_name}"] = time.time()
        if self.use_emoji:
            self._print(f"📊 Loading dataset '{dataset_name}'...")
        else:
            self._print(f"Loading dataset '{dataset_name}'...")
    
    def loading_complete(self, dataset_name: str, size: Optional[int] = None) -> None:
        """Signal completion of dataset loading."""
        duration = time.time() - self._start_times.get(f"loading_{dataset_name}", time.time())
        size_str = f" ({size} examples)" if size is not None else ""
        if self.use_emoji:
            self._print(f"✅ Loaded '{dataset_name}'{size_str} ({self._format_duration(duration)})")
        else:
            self._print(f"Loaded '{dataset_name}'{size_str} ({self._format_duration(duration)})")
    
    def loading_error(self, dataset_name: str, error: str) -> None:
        """Signal an error during dataset loading."""
        if self.use_emoji:
            self._print(f"❌ Failed to load '{dataset_name}': {error}")
        else:
            self._print(f"ERROR: Failed to load '{dataset_name}': {error}")
    
    # Execution Operations
    
    def execution_start(self, operation: str) -> None:
        """Signal the start of an execution operation."""
        self._start_times[f"execution_{operation}"] = time.time()
        if self.use_emoji:
            self._print(f"⚡ Executing {operation}...")
        else:
            self._print(f"Executing {operation}...")
    
    def execution_complete(self, operation: str) -> None:
        """Signal completion of an execution operation."""
        duration = time.time() - self._start_times.get(f"execution_{operation}", time.time())
        if self.use_emoji:
            self._print(f"✅ Completed {operation} ({self._format_duration(duration)})")
        else:
            self._print(f"Completed {operation} ({self._format_duration(duration)})")
    
    def execution_error(self, operation: str, error: str) -> None:
        """Signal an error during execution."""
        if self.use_emoji:
            self._print(f"❌ {operation} failed: {error}")
        else:
            self._print(f"ERROR: {operation} failed: {error}")
    
    # General Operations
    
    def info(self, message: str) -> None:
        """Display an informational message."""
        if self.use_emoji:
            self._print(f"ℹ️  {message}")
        else:
            self._print(f"INFO: {message}")
    
    def warning(self, message: str) -> None:
        """Display a warning message."""
        if self.use_emoji:
            self._print(f"⚠️  {message}")
        else:
            self._print(f"WARNING: {message}")
    
    def success(self, message: str) -> None:
        """Display a success message."""
        if self.use_emoji:
            self._print(f"✅ {message}")
        else:
            self._print(f"SUCCESS: {message}")
    
    def step(self, step_num: int, total: int, message: str) -> None:
        """Display a step in a multi-step process."""
        if self.use_emoji:
            self._print(f"  [{step_num}/{total}] {message}")
        else:
            self._print(f"  Step {step_num}/{total}: {message}")
    
    @contextmanager
    def timed_operation(self, operation: str):
        """
        Context manager for timing operations.
        
        Example:
            with reporter.timed_operation("model inference"):
                result = model(prompt)
        """
        start_time = time.time()
        try:
            if self.use_emoji:
                self._print(f"⏱️  Starting {operation}...", end="")
            else:
                self._print(f"Starting {operation}...", end="")
            
            yield
            
            duration = time.time() - start_time
            self._print(f" done ({self._format_duration(duration)})")
            
        except Exception as e:
            duration = time.time() - start_time
            self._print(f" failed ({self._format_duration(duration)})")
            raise


# Global instance for convenience
_default_reporter: Optional[ProgressReporter] = None


def get_default_reporter(quiet: Optional[bool] = None) -> ProgressReporter:
    """
    Get the default progress reporter instance.
    
    Args:
        quiet: Override quiet mode setting
        
    Returns:
        The default ProgressReporter instance
    """
    global _default_reporter
    
    if _default_reporter is None or quiet is not None:
        _default_reporter = ProgressReporter(quiet=quiet if quiet is not None else False)
    
    return _default_reporter


# Convenience functions using the default reporter

def discovery_start() -> None:
    """Signal the start of model discovery using the default reporter."""
    get_default_reporter().discovery_start()


def discovery_complete(count: int) -> None:
    """Signal completion of model discovery using the default reporter."""
    get_default_reporter().discovery_complete(count)


def loading_start(dataset_name: str) -> None:
    """Signal the start of dataset loading using the default reporter."""
    get_default_reporter().loading_start(dataset_name)


def loading_complete(dataset_name: str, size: Optional[int] = None) -> None:
    """Signal completion of dataset loading using the default reporter."""
    get_default_reporter().loading_complete(dataset_name, size)


def info(message: str) -> None:
    """Display an informational message using the default reporter."""
    get_default_reporter().info(message)


def warning(message: str) -> None:
    """Display a warning message using the default reporter."""
    get_default_reporter().warning(message)


def success(message: str) -> None:
    """Display a success message using the default reporter."""
    get_default_reporter().success(message)