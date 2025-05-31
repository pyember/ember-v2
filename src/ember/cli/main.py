#!/usr/bin/env python3
"""Ember CLI - Command-line interface for Ember AI framework."""

import sys
import argparse
from typing import Optional, List

from ember.core.utils.output import print_header, print_error, print_success
from ember.core.utils.verbosity import VerbosityLevel, set_verbosity
from ember.core.utils.logging import configure_logging

from ember.cli import __version__
from ember.cli.commands import (
    model_command,
    invoke_command,
    eval_command,
    project_command,
    config_command,
    version_command)


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        prog="ember",
        description="Ember AI - Command-line interface for building compound AI systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ember model list                    List available models
  ember invoke gpt-4 "Hello world"    Invoke a model with a prompt
  ember eval run mmlu                 Run MMLU evaluation
  ember project new my-app            Create a new Ember project
  
For more help on a specific command:
  ember <command> --help
"""
    )
    
    # Global options
    global_group = parser.add_argument_group("global options")
    global_group.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    global_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    global_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show additional information"
    )
    global_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logs"
    )
    global_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    global_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
        metavar="<command>"
    )
    
    # Register commands
    model_command.register(subparsers)
    invoke_command.register(subparsers)
    eval_command.register(subparsers)
    project_command.register(subparsers)
    config_command.register(subparsers)
    version_command.register(subparsers)
    
    return parser


def setup_environment(args: argparse.Namespace) -> None:
    """Set up the environment based on global options."""
    # Set verbosity level
    if args.quiet:
        set_verbosity(VerbosityLevel.QUIET)
    elif args.debug:
        set_verbosity(VerbosityLevel.DEBUG)
    elif args.verbose:
        set_verbosity(VerbosityLevel.VERBOSE)
    else:
        set_verbosity(VerbosityLevel.NORMAL)
    
    # Configure logging
    configure_logging(verbose=args.verbose or args.debug)
    
    # Disable colors if requested
    if args.no_color:
        import os
        os.environ["NO_COLOR"] = "1"
    
    # Set JSON output mode
    if args.json:
        import os
        os.environ["EMBER_OUTPUT_FORMAT"] = "json"


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the Ember CLI."""
    try:
        # Create parser
        parser = create_parser()
        
        # Parse arguments
        args = parser.parse_args(argv)
        
        # Show help if no command specified
        if not args.command:
            parser.print_help()
            return 0
        
        # Set up environment
        setup_environment(args)
        
        # Execute command
        return args.func(args)
        
    except KeyboardInterrupt:
        print_error("\nOperation cancelled by user")
        return 130
    except Exception as e:
        if args.debug:
            # In debug mode, show full traceback
            import traceback
            traceback.print_exc()
        else:
            print_error(f"Error: {str(e)}")
        return 1


def cli():
    """Console script entry point."""
    sys.exit(main())


if __name__ == "__main__":
    cli()