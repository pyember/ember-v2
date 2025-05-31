"""Verbosity control for output levels.

Centralized management of output detail with environment variable support.
"""

import os
import argparse
from enum import IntEnum
from typing import Optional, Any
from contextlib import contextmanager

from ember.core.utils.logging import configure_logging
from ember.core.utils.progress import get_default_reporter


class VerbosityLevel(IntEnum):
    """Verbosity levels for output control.
    
    Attributes:
        QUIET: Essential output only
        NORMAL: Default clean output
        VERBOSE: Additional details
        DEBUG: All debug information
    """
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


# Global verbosity level
_verbosity_level: VerbosityLevel = VerbosityLevel.NORMAL


def get_verbosity() -> VerbosityLevel:
    """Get the current global verbosity level."""
    return _verbosity_level


def set_verbosity(level: VerbosityLevel) -> None:
    """
    Set the global verbosity level.
    
    Args:
        level: The verbosity level to set
    """
    global _verbosity_level
    _verbosity_level = level
    
    # Configure logging based on verbosity
    if level == VerbosityLevel.QUIET:
        configure_logging(verbose=False)
        # Set progress reporter to quiet mode
        reporter = get_default_reporter(quiet=True)
    elif level == VerbosityLevel.VERBOSE:
        configure_logging(verbose=True)
        reporter = get_default_reporter(quiet=False)
    elif level == VerbosityLevel.DEBUG:
        import logging
        # Set all Ember loggers to DEBUG
        ember_logger = logging.getLogger("ember")
        ember_logger.setLevel(logging.DEBUG)
        configure_logging(verbose=True)
        reporter = get_default_reporter(quiet=False)
    else:  # NORMAL
        configure_logging(verbose=False)
        reporter = get_default_reporter(quiet=False)


def add_verbosity_args(parser: argparse.ArgumentParser) -> None:
    """
    Add standard verbosity arguments to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to
    
    Example:
        parser = argparse.ArgumentParser()
        add_verbosity_args(parser)
        args = parser.parse_args()
        apply_verbosity_args(args)
    """
    verbosity_group = parser.add_mutually_exclusive_group()
    
    verbosity_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - suppress non-essential output"
    )
    
    verbosity_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode - show additional information"
    )
    
    verbosity_group.add_argument(
        "-vv", "--debug",
        action="store_true",
        help="Debug mode - show all debugging information"
    )


def apply_verbosity_args(args: argparse.Namespace) -> None:
    """
    Apply verbosity settings from parsed arguments.
    
    Args:
        args: Parsed arguments containing verbosity flags
    """
    if hasattr(args, "quiet") and args.quiet:
        set_verbosity(VerbosityLevel.QUIET)
    elif hasattr(args, "debug") and args.debug:
        set_verbosity(VerbosityLevel.DEBUG)
    elif hasattr(args, "verbose") and args.verbose:
        set_verbosity(VerbosityLevel.VERBOSE)
    else:
        # Check environment variable
        env_verbosity = os.environ.get("EMBER_VERBOSITY", "").lower()
        if env_verbosity == "quiet":
            set_verbosity(VerbosityLevel.QUIET)
        elif env_verbosity == "verbose":
            set_verbosity(VerbosityLevel.VERBOSE)
        elif env_verbosity == "debug":
            set_verbosity(VerbosityLevel.DEBUG)
        else:
            set_verbosity(VerbosityLevel.NORMAL)


def should_print(level: VerbosityLevel) -> bool:
    """
    Check if output should be printed at the given level.
    
    Args:
        level: The verbosity level of the output
        
    Returns:
        True if the output should be printed
    """
    return _verbosity_level >= level


def print_at_level(message: str, level: VerbosityLevel) -> None:
    """
    Print a message only if the current verbosity level allows it.
    
    Args:
        message: Message to print
        level: Minimum verbosity level required
    """
    if should_print(level):
        print(message)


def vprint(message: str) -> None:
    """Print a message in verbose mode."""
    print_at_level(message, VerbosityLevel.VERBOSE)


def dprint(message: str) -> None:
    """Print a message in debug mode."""
    print_at_level(message, VerbosityLevel.DEBUG)


@contextmanager
def temporary_verbosity(level: VerbosityLevel):
    """
    Context manager for temporary verbosity changes.
    
    Args:
        level: Verbosity level to use temporarily
        
    Example:
        with temporary_verbosity(VerbosityLevel.DEBUG):
            # Debug output enabled here
            run_complex_operation()
    """
    old_level = get_verbosity()
    set_verbosity(level)
    try:
        yield
    finally:
        set_verbosity(old_level)


class VerbosityManager:
    """
    Manager class for handling verbosity in a structured way.
    
    Useful for classes that need to manage their own verbosity
    settings independently.
    """
    
    def __init__(self, default_level: VerbosityLevel = VerbosityLevel.NORMAL):
        """
        Initialize the verbosity manager.
        
        Args:
            default_level: Default verbosity level
        """
        self.level = default_level
    
    def should_print(self, level: VerbosityLevel) -> bool:
        """Check if output should be printed at the given level."""
        return self.level >= level
    
    def print(self, message: str, level: VerbosityLevel = VerbosityLevel.NORMAL) -> None:
        """Print a message if verbosity allows."""
        if self.should_print(level):
            print(message)
    
    def info(self, message: str) -> None:
        """Print an info message (normal verbosity)."""
        self.print(message, VerbosityLevel.NORMAL)
    
    def verbose(self, message: str) -> None:
        """Print a verbose message."""
        self.print(message, VerbosityLevel.VERBOSE)
    
    def debug(self, message: str) -> None:
        """Print a debug message."""
        self.print(message, VerbosityLevel.DEBUG)
    
    @contextmanager
    def at_level(self, level: VerbosityLevel):
        """Temporarily change verbosity level."""
        old_level = self.level
        self.level = level
        try:
            yield
        finally:
            self.level = old_level


# Convenience functions for common patterns

def create_argument_parser(description: str, **kwargs) -> argparse.ArgumentParser:
    """
    Create an ArgumentParser with standard verbosity arguments.
    
    Args:
        description: Program description
        **kwargs: Additional ArgumentParser arguments
        
    Returns:
        ArgumentParser with verbosity arguments added
    """
    parser = argparse.ArgumentParser(description=description, **kwargs)
    add_verbosity_args(parser)
    return parser


def setup_verbosity_from_args(args: Optional[Any] = None) -> VerbosityLevel:
    """
    Set up verbosity from command line arguments.
    
    Args:
        args: Pre-parsed arguments (None to parse sys.argv)
        
    Returns:
        The configured verbosity level
    """
    if args is None:
        # Create a minimal parser just for verbosity
        parser = argparse.ArgumentParser(add_help=False)
        add_verbosity_args(parser)
        args, _ = parser.parse_known_args()
    
    apply_verbosity_args(args)
    return get_verbosity()