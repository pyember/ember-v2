"""
Clean Output Formatting for Ember

Provides utilities for formatting output in a clean, professional manner
for Ember examples and CLI tools.

This module offers:
- Formatted headers and sections
- Clean table output for results
- Model listings with grouping
- Performance metrics formatting
- Color support with NO_COLOR respect
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


# Check if colors should be used
def _should_use_colors() -> bool:
    """Check if terminal supports and wants colors."""
    # Respect NO_COLOR environment variable
    if os.environ.get("NO_COLOR"):
        return False
    
    # Check if stdout is a TTY
    if not sys.stdout.isatty():
        return False
    
    # Check if in CI environment
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return False
    
    return True


USE_COLORS = _should_use_colors()


# Color codes
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[1;34m' if USE_COLORS else ''  # Bold blue
    SUCCESS = '\033[32m' if USE_COLORS else ''    # Green
    WARNING = '\033[33m' if USE_COLORS else ''    # Yellow
    ERROR = '\033[31m' if USE_COLORS else ''      # Red
    INFO = '\033[36m' if USE_COLORS else ''       # Cyan
    BOLD = '\033[1m' if USE_COLORS else ''        # Bold
    DIM = '\033[2m' if USE_COLORS else ''         # Dim
    RESET = '\033[0m' if USE_COLORS else ''       # Reset


def print_header(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Header text
        width: Total width of the header line
    """
    # Calculate padding
    padding = (width - len(title) - 2) // 2
    line = "=" * width
    
    print()
    print(f"{Colors.HEADER}{line}{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * padding} {title} {'=' * (width - padding - len(title) - 2)}{Colors.RESET}")
    print(f"{Colors.HEADER}{line}{Colors.RESET}")
    print()


def print_subheader(title: str, width: int = 60) -> None:
    """
    Print a formatted subsection header.
    
    Args:
        title: Subheader text
        width: Total width of the header line
    """
    print()
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * len(title)}{Colors.RESET}")


def print_summary(results: Dict[str, Any], title: str = "Summary") -> None:
    """
    Print a clean summary table of results.
    
    Args:
        results: Dictionary of key-value pairs to display
        title: Title for the summary section
    """
    if not results:
        return
    
    print_subheader(title)
    
    # Find the longest key for alignment
    max_key_len = max(len(str(k)) for k in results.keys())
    
    # Print each key-value pair
    for key, value in results.items():
        # Format the value based on type
        if isinstance(value, float):
            value_str = f"{value:.2f}"
        elif isinstance(value, bool):
            value_str = f"{Colors.SUCCESS}✓{Colors.RESET}" if value else f"{Colors.ERROR}✗{Colors.RESET}"
        else:
            value_str = str(value)
        
        print(f"  {str(key).ljust(max_key_len)} : {value_str}")
    
    print()


def print_models(models: List[str], group_by_provider: bool = True) -> None:
    """
    Print a formatted list of models.
    
    Args:
        models: List of model identifiers
        group_by_provider: Whether to group models by provider
    """
    if not models:
        print(f"{Colors.WARNING}No models found.{Colors.RESET}")
        return
    
    if group_by_provider and any(":" in m for m in models):
        # Group by provider
        by_provider = defaultdict(list)
        for model in models:
            if ":" in model:
                provider, model_name = model.split(":", 1)
                by_provider[provider].append(model_name)
            else:
                by_provider["unknown"].append(model)
        
        # Print grouped
        for provider in sorted(by_provider.keys()):
            model_list = by_provider[provider]
            print(f"\n{Colors.BOLD}{provider.upper()}{Colors.RESET} ({len(model_list)} models)")
            for model in sorted(model_list):
                print(f"  • {model}")
    else:
        # Print flat list
        print(f"\n{Colors.BOLD}Available Models{Colors.RESET} ({len(models)} total)")
        for model in sorted(models):
            print(f"  • {model}")
    
    print()


def print_metrics(metrics: Dict[str, Any], title: str = "Performance Metrics") -> None:
    """
    Print formatted performance metrics.
    
    Args:
        metrics: Dictionary of metric names to values
        title: Title for the metrics section
    """
    if not metrics:
        return
    
    print_subheader(title)
    
    # Group metrics by type
    timing_metrics = {}
    count_metrics = {}
    other_metrics = {}
    
    for key, value in metrics.items():
        if any(x in key.lower() for x in ["time", "duration", "latency", "_ms", "_sec"]):
            timing_metrics[key] = value
        elif any(x in key.lower() for x in ["count", "total", "num_", "n_"]):
            count_metrics[key] = value
        else:
            other_metrics[key] = value
    
    # Print timing metrics
    if timing_metrics:
        print(f"  {Colors.INFO}Timing:{Colors.RESET}")
        max_key_len = max(len(k) for k in timing_metrics.keys())
        for key, value in sorted(timing_metrics.items()):
            if isinstance(value, (int, float)):
                # Format time values
                if value < 1:
                    value_str = f"{value*1000:.1f}ms"
                elif value < 60:
                    value_str = f"{value:.2f}s"
                else:
                    value_str = f"{value/60:.1f}min"
            else:
                value_str = str(value)
            print(f"    {key.ljust(max_key_len)} : {value_str}")
    
    # Print count metrics
    if count_metrics:
        print(f"\n  {Colors.INFO}Counts:{Colors.RESET}")
        max_key_len = max(len(k) for k in count_metrics.keys())
        for key, value in sorted(count_metrics.items()):
            print(f"    {key.ljust(max_key_len)} : {value:,}")
    
    # Print other metrics
    if other_metrics:
        print(f"\n  {Colors.INFO}Other:{Colors.RESET}")
        max_key_len = max(len(k) for k in other_metrics.keys())
        for key, value in sorted(other_metrics.items()):
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            print(f"    {key.ljust(max_key_len)} : {value_str}")
    
    print()


def print_table(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    title: Optional[str] = None
) -> None:
    """
    Print data in a clean table format.
    
    Args:
        data: List of dictionaries representing rows
        columns: List of column names to display (None = all columns)
        title: Optional title for the table
    """
    if not data:
        print(f"{Colors.WARNING}No data to display.{Colors.RESET}")
        return
    
    if title:
        print_subheader(title)
    
    # Determine columns
    if columns is None:
        columns = list(data[0].keys())
    
    # Calculate column widths
    col_widths = {}
    for col in columns:
        # Start with column name length
        col_widths[col] = len(col)
        # Check all values
        for row in data:
            if col in row:
                val_len = len(str(row[col]))
                col_widths[col] = max(col_widths[col], val_len)
    
    # Print header
    header_parts = []
    for col in columns:
        header_parts.append(col.ljust(col_widths[col]))
    print(f"  {Colors.BOLD}{' | '.join(header_parts)}{Colors.RESET}")
    
    # Print separator
    sep_parts = []
    for col in columns:
        sep_parts.append("-" * col_widths[col])
    print(f"  {Colors.DIM}{'-|-'.join(sep_parts)}{Colors.RESET}")
    
    # Print rows
    for row in data:
        row_parts = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            row_parts.append(value_str.ljust(col_widths[col]))
        print(f"  {' | '.join(row_parts)}")
    
    print()


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """
    Print a simple progress indicator.
    
    Args:
        current: Current item number
        total: Total number of items
        prefix: Prefix text for the progress
    """
    percent = (current / total) * 100 if total > 0 else 0
    bar_length = 40
    filled_length = int(bar_length * current // total)
    
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    
    # Use carriage return to update the same line
    print(f"\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
    
    # Print newline when complete
    if current >= total:
        print()


def print_error(message: str, details: Optional[str] = None) -> None:
    """
    Print an error message with optional details.
    
    Args:
        message: Main error message
        details: Optional detailed error information
    """
    print(f"{Colors.ERROR}Error: {message}{Colors.RESET}")
    if details:
        print(f"{Colors.DIM}Details: {details}{Colors.RESET}")


def print_warning(message: str) -> None:
    """
    Print a warning message.
    
    Args:
        message: Warning message
    """
    print(f"{Colors.WARNING}Warning: {message}{Colors.RESET}")


def print_success(message: str) -> None:
    """
    Print a success message.
    
    Args:
        message: Success message
    """
    print(f"{Colors.SUCCESS}✓ {message}{Colors.RESET}")


def print_info(message: str) -> None:
    """
    Print an informational message.
    
    Args:
        message: Information message
    """
    print(f"{Colors.INFO}ℹ {message}{Colors.RESET}")