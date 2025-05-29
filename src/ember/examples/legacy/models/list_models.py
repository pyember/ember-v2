"""List Available Models Example

This script demonstrates how to list and inspect available models using
the simplified Ember models API.

To run:
    uv run python src/ember/examples/models/list_models.py
    uv run python src/ember/examples/models/list_models.py --verbose
    uv run python src/ember/examples/models/list_models.py --quiet

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key
    ANTHROPIC_API_KEY (optional): Your Anthropic API key
"""

import os

from ember.api.models import models
from ember.core.utils.output import (
    print_header, print_models, print_summary, print_table,
    print_warning, print_error, print_success, print_info
)
from ember.core.utils.progress import ProgressReporter
from ember.core.utils.verbosity import create_argument_parser, setup_verbosity_from_args, vprint
from ember.core.utils.logging import suppress_logs


def main():
    """Example demonstrating the simplified XCS architecture."""
    """Run the list models example."""
    # Set up argument parser
    parser = create_argument_parser("List and inspect available Ember models")
    args = parser.parse_args()
    setup_verbosity_from_args(args)
    
    print_header("List Available Models Example")
    
    # Check if API keys are set
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print_warning("No API keys set. Model discovery may be limited.")
        print_info("Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY for full functionality.")
    
    # Initialize progress reporter
    reporter = ProgressReporter(quiet=args.quiet)
    
    # List all available models
    reporter.discovery_start()
    try:
        # Suppress model discovery logs
        with suppress_logs(["ember.core.registry.model", "ember.core.registry.model.initialization"]):
            available_models = models.list()
        
        reporter.discovery_complete(len(available_models))
        
        if not available_models:
            print_warning("No models found. Make sure API keys are set.")
            return
        
        # Display models using the clean formatter
        print_models(available_models, group_by_provider=True)
        
    except Exception as e:
        reporter.discovery_error(str(e))
        return
    
    # List models by specific provider (verbose mode)
    if args.verbose:
        print("\nModel counts by provider:")
        provider_counts = {}
        for provider in ["openai", "anthropic", "google"]:
            try:
                with suppress_logs(["ember.core.registry.model"]):
                    provider_models = models.list(provider=provider)
                provider_counts[provider] = len(provider_models)
            except Exception:
                provider_counts[provider] = 0
        
        print_summary(provider_counts, title="Provider Summary")
    
    # Get detailed info for specific models
    if not args.quiet:
        print("\nDetailed Model Information:")
        example_models = ["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"]
        
        model_details = []
        for model_id in example_models:
            try:
                with suppress_logs(["ember.core.registry.model"]):
                    info = models.info(model_id)
                
                details = {
                    "Model": model_id,
                    "Full ID": info.get('id', 'N/A'),
                    "Provider": info.get('provider', 'N/A'),
                }
                
                if 'context_window' in info:
                    details["Context"] = f"{info['context_window']:,}"
                
                if 'pricing' in info:
                    pricing = info['pricing']
                    details["Input $/1K"] = f"${pricing.get('input', 0):.6f}"
                    details["Output $/1K"] = f"${pricing.get('output', 0):.6f}"
                
                model_details.append(details)
                
            except Exception as e:
                vprint(f"Could not get info for {model_id}: {e}")
                model_details.append({
                    "Model": model_id,
                    "Full ID": "Not available",
                    "Provider": "Unknown",
                })
        
        if model_details:
            print_table(model_details, title="Model Details")
    
    print_success("Example completed!")


if __name__ == "__main__":
    main()