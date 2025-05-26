"""Version command implementation."""

import argparse
import sys
import platform
from typing import Optional

from ember.core.utils.output import print_header, print_table


def register(subparsers) -> argparse.ArgumentParser:
    """Add version command to subparsers."""
    parser = subparsers.add_parser(
        "version",
        help="Show version information",
        description="Display Ember and system version information"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed version information"
    )
    
    parser.set_defaults(func=execute)
    return parser


def execute(args: argparse.Namespace) -> int:
    """Execute version command."""
    try:
        # Get Ember version
        try:
            from ember import __version__
            ember_version = __version__
        except ImportError:
            ember_version = "Unknown (dev)"
        
        if args.detailed:
            print_header("Ember Version Information")
            
            # Collect version info
            info = [
                {"Component": "Ember", "Version": ember_version},
                {"Component": "Python", "Version": platform.python_version()},
                {"Component": "Platform", "Version": platform.platform()},
                {"Component": "Architecture", "Version": f"{platform.machine()} ({platform.processor()})"}]
            
            # Try to get dependency versions
            try:
                import torch
                info.append({"Component": "PyTorch", "Version": torch.__version__})
            except ImportError:
                pass
            
            try:
                import transformers
                info.append({"Component": "Transformers", "Version": transformers.__version__})
            except ImportError:
                pass
            
            try:
                import openai
                info.append({"Component": "OpenAI", "Version": openai.__version__})
            except ImportError:
                pass
            
            try:
                import anthropic
                info.append({"Component": "Anthropic", "Version": anthropic.__version__})
            except ImportError:
                pass
            
            print_table(info)
            
            # Show paths
            print("\nPaths:")
            import ember
            print(f"  Ember installation: {ember.__file__}")
            
            import os
            if "EMBER_CONFIG_PATH" in os.environ:
                print(f"  Config path (env): {os.environ['EMBER_CONFIG_PATH']}")
            
            from pathlib import Path
            user_config = Path.home() / ".ember" / "config.yaml"
            if user_config.exists():
                print(f"  Config path (user): {user_config}")
            
        else:
            # Simple version output
            print(f"ember {ember_version}")
        
        return 0
        
    except Exception as e:
        print(f"Error getting version information: {e}", file=sys.stderr)
        return 1