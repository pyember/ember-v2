"""Invoke command implementation."""

import argparse
import sys
import json
from typing import Optional, Dict, Any

from ember.core.utils.output import print_header, print_info, print_error
from ember.core.utils.progress import ProgressReporter
from ember.core.utils.verbosity import get_verbosity
from ember.api.models import models


def register(subparsers) -> argparse.ArgumentParser:
    """Add invoke command to subparsers."""
    parser = subparsers.add_parser(
        "invoke",
        help="Invoke a model with a prompt",
        description="Send a prompt to a model and get a response"
    )
    
    parser.add_argument("model", help="Model to invoke (e.g., gpt-4, claude-3)")
    parser.add_argument("prompt", help="Prompt to send to the model")
    
    # Optional arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (0.0-2.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--system",
        help="System prompt"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response"
    )
    parser.add_argument(
        "--save",
        help="Save response to file"
    )
    
    parser.set_defaults(func=execute)
    return parser


def execute(args: argparse.Namespace) -> int:
    """Execute invoke command."""
    progress = ProgressReporter()
    
    try:
        # Check if model exists
        available = models.available()
        if args.model not in available:
            print_error(f"Model '{args.model}' not found")
            print("\nAvailable models:")
            for m in sorted(available)[:10]:
                print(f"  - {m}")
            if len(available) > 10:
                print(f"  ... and {len(available) - 10} more")
            return 1
        
        # Prepare the request
        if get_verbosity() >= 1:
            print_header(f"Invoking {args.model}")
            print(f"Prompt: {args.prompt[:50]}..." if len(args.prompt) > 50 else f"Prompt: {args.prompt}")
        
        # Build messages
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        
        # Prepare generation kwargs
        kwargs = {
            "temperature": args.temperature,
        }
        if args.max_tokens:
            kwargs["max_tokens"] = args.max_tokens
        
        # Invoke the model
        if get_verbosity() >= 1:
            progress.execution_start("Generating response")
        
        if args.stream:
            # Stream response
            response_text = ""
            for chunk in models(args.model)(messages, stream=True, **kwargs):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    response_text += chunk
                elif hasattr(chunk, "content"):
                    print(chunk.content, end="", flush=True)
                    response_text += chunk.content
            print()  # New line after streaming
        else:
            # Get full response
            response = models(args.model)(messages, **kwargs)
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Format output
            if args.format == "json":
                output = {
                    "model": args.model,
                    "prompt": args.prompt,
                    "response": response_text,
                    "temperature": args.temperature
                }
                if args.system:
                    output["system"] = args.system
                print(json.dumps(output, indent=2))
            elif args.format == "markdown":
                print(f"## Model: {args.model}\n")
                if args.system:
                    print(f"**System:** {args.system}\n")
                print(f"**Prompt:** {args.prompt}\n")
                print(f"**Response:**\n\n{response_text}")
            else:
                print(response_text)
        
        if get_verbosity() >= 1:
            progress.execution_complete()
        
        # Save if requested
        if args.save:
            with open(args.save, "w") as f:
                f.write(response_text)
            print_info(f"Response saved to {args.save}")
        
        return 0
        
    except Exception as e:
        print_error(f"Error invoking model: {e}")
        return 1