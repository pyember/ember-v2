"""Registry introspection commands for Ember CLI.

This module provides commands for inspecting the model and data registries,
allowing users to discover available resources.
"""

from ember.context import context


def cmd_registry_list_models(args) -> int:
    """List all available models in the registry.

    Provides a more explicit and organized view of available models compared
    to the legacy 'ember models' command.

    Args:
        args: Parsed command line arguments containing:
            - provider (str, optional): Filter by specific provider
            - verbose (bool): Show detailed information

    Returns:
        int: Exit code (0 for success)
    """
    ctx = context.get()

    try:
        # Get available models
        all_models = ctx.list_models()

        # Group by provider if verbose
        if getattr(args, "verbose", False):
            # Import here to avoid circular dependency
            from ember.models.catalog import MODEL_CATALOG

            # Group models by provider
            by_provider = {}
            for model_id in all_models:
                if model_id in MODEL_CATALOG:
                    info = MODEL_CATALOG[model_id]
                    provider = info["provider"]
                    if provider not in by_provider:
                        by_provider[provider] = []
                    by_provider[provider].append((model_id, info))

            # Display grouped
            for provider, models in sorted(by_provider.items()):
                print(f"\n{provider.upper()} Models:")
                for model_id, info in sorted(models):
                    print(f"  {model_id:<25} {info['description']}")
                    if "context_window" in info:
                        print(f"    Context: {info['context_window']:,} tokens")
        else:
            # Simple list with optional filtering
            provider_filter = getattr(args, "provider", None)

            if provider_filter:
                # Import catalog for filtering
                from ember.models.catalog import MODEL_CATALOG

                models = [
                    m
                    for m in all_models
                    if m in MODEL_CATALOG and MODEL_CATALOG[m]["provider"] == provider_filter
                ]
                print(f"Available {provider_filter} models:")
            else:
                models = all_models
                print("Available models:")

            for model_id in sorted(models):
                print(f"  {model_id}")

        return 0

    except Exception as e:
        print(f"Error listing models: {e}")
        return 1


def cmd_registry_list_providers(args) -> int:
    """List all configured providers.

    Shows which providers are available and configured with API keys.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code (0 for success)
    """
    ctx = context.get()

    # Known providers
    providers = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    print("Provider Status:")
    for provider, env_var in providers.items():
        # Check if configured
        has_key = bool(ctx.get_credential(provider, env_var))
        status = "✅ Configured" if has_key else "❌ Not configured"
        print(f"  {provider:<12} {status}")

    return 0


def cmd_registry_info(args) -> int:
    """Show detailed information about a specific model.

    Args:
        args: Parsed command line arguments containing:
            - model_id (str): The model to get info about

    Returns:
        int: Exit code (0 for success, 1 for not found)
    """
    from ember.models.catalog import MODEL_CATALOG

    model_id = args.model_id

    if model_id not in MODEL_CATALOG:
        print(f"Model '{model_id}' not found in catalog")
        return 1

    info = MODEL_CATALOG[model_id]

    print(f"Model: {model_id}")
    print(f"Provider: {info['provider']}")
    print(f"Description: {info['description']}")

    if "context_window" in info:
        print(f"Context Window: {info['context_window']:,} tokens")

    if "input_cost" in info and "output_cost" in info:
        print(
            f"Cost: ${info['input_cost']:.4f} input / "
            f"${info['output_cost']:.4f} output per 1K tokens"
        )

    if "capabilities" in info:
        print(f"Capabilities: {', '.join(info['capabilities'])}")

    return 0
