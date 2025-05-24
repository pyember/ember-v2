"""Model command implementation."""

import argparse
import sys
from typing import Optional, List

from ember.core.utils.output import print_header, print_models, print_info
from ember.core.utils.progress import ProgressReporter
from ember.core.utils.verbosity import get_verbosity
from ember.api.models import models


def register(subparsers) -> argparse.ArgumentParser:
    """Register model command and return parser."""
    parser = subparsers.add_parser(
        "model",
        help="Manage and inspect models",
        description="Commands for listing, searching, and getting information about available models"
    )
    
    subcommands = parser.add_subparsers(dest="subcommand", help="Model subcommands")
    
    # List command
    list_parser = subcommands.add_parser("list", help="List available models")
    list_parser.add_argument(
        "--provider",
        help="Filter by provider (e.g., openai, anthropic)"
    )
    list_parser.add_argument(
        "--format",
        choices=["table", "json", "simple"],
        default="table",
        help="Output format"
    )
    list_parser.add_argument(
        "--no-group",
        action="store_true",
        help="Don't group models by provider"
    )
    
    # Info command
    info_parser = subcommands.add_parser("info", help="Get detailed information about a model")
    info_parser.add_argument("model", help="Model name (e.g., gpt-4, claude-3)")
    
    # Search command
    search_parser = subcommands.add_parser("search", help="Search for models")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--in",
        dest="search_in",
        choices=["name", "description", "all"],
        default="all",
        help="Where to search"
    )
    
    # Set the function to execute
    parser.set_defaults(func=execute)
    return parser


def execute(args: argparse.Namespace) -> int:
    """Execute model command."""
    if not args.subcommand:
        print("Error: No subcommand specified. Use 'ember model --help' for usage.", file=sys.stderr)
        return 1
    
    if args.subcommand == "list":
        return list_models(args)
    elif args.subcommand == "info":
        return model_info(args)
    elif args.subcommand == "search":
        return search_models(args)
    else:
        print(f"Error: Unknown subcommand '{args.subcommand}'", file=sys.stderr)
        return 1


def list_models(args: argparse.Namespace) -> int:
    """List available models."""
    progress = ProgressReporter()
    
    # Discovery phase
    if get_verbosity() >= 1:
        progress.discovery_start()
    
    try:
        # Get available models
        available_models = models.available()
        
        if get_verbosity() >= 1:
            progress.discovery_complete(len(available_models))
        
        # Filter by provider if specified
        if args.provider:
            filtered = []
            for model in available_models:
                provider = model.split("-")[0] if "-" in model else model.split("/")[0]
                if provider.lower() == args.provider.lower():
                    filtered.append(model)
            available_models = filtered
        
        if not available_models:
            if args.provider:
                print(f"No models found for provider '{args.provider}'")
            else:
                print("No models found")
            return 0
        
        # Output based on format
        if args.format == "json":
            import json
            print(json.dumps(available_models, indent=2))
        elif args.format == "simple":
            for model in sorted(available_models):
                print(model)
        else:  # table format
            print_header("Available Models")
            print_models(available_models, group_by_provider=not args.no_group)
            print_info(f"Total: {len(available_models)} models")
        
        return 0
        
    except Exception as e:
        print(f"Error listing models: {e}", file=sys.stderr)
        return 1


def model_info(args: argparse.Namespace) -> int:
    """Get detailed information about a model."""
    try:
        # Check if model exists
        available = models.available()
        if args.model not in available:
            print(f"Error: Model '{args.model}' not found", file=sys.stderr)
            print("\nDid you mean one of these?")
            # Find similar models
            similar = [m for m in available if args.model.lower() in m.lower()][:5]
            for m in similar:
                print(f"  - {m}")
            return 1
        
        # Get model info
        print_header(f"Model: {args.model}")
        
        # Extract provider from model name
        provider = args.model.split("-")[0] if "-" in args.model else args.model.split("/")[0]
        print(f"Provider: {provider}")
        
        # TODO: When model registry provides more metadata, display it here
        # For now, just confirm it exists
        print(f"Status: Available")
        
        return 0
        
    except Exception as e:
        print(f"Error getting model info: {e}", file=sys.stderr)
        return 1


def search_models(args: argparse.Namespace) -> int:
    """Search for models."""
    try:
        available = models.available()
        query = args.query.lower()
        
        matches = []
        for model in available:
            if args.search_in == "name" or args.search_in == "all":
                if query in model.lower():
                    matches.append(model)
                    continue
            
            # TODO: When model registry provides descriptions, search those too
            # if args.search_in == "description" or args.search_in == "all":
            #     if query in model_description.lower():
            #         matches.append(model)
        
        if not matches:
            print(f"No models found matching '{args.query}'")
            return 0
        
        print_header(f"Search Results for '{args.query}'")
        print_models(matches, group_by_provider=True)
        print_info(f"Found {len(matches)} models")
        
        return 0
        
    except Exception as e:
        print(f"Error searching models: {e}", file=sys.stderr)
        return 1