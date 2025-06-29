"""Minimal launcher for the npm setup wizard.

Following the principle of "do one thing well", this module simply
launches the npm-based setup wizard when needed. The wizard itself
handles all the UI complexity.
"""

import os
import shutil
import subprocess
import sys
from typing import Optional


def launch_setup_if_needed(provider: str, env_var: str, model_id: str) -> Optional[str]:
    """Launch setup wizard if in interactive mode.

    Args:
        provider: Provider name (e.g., "openai")
        env_var: Environment variable name
        model_id: Model being accessed

    Returns:
        API key if setup succeeds, None otherwise
    """
    # Check if we're in an interactive terminal
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    # Check if npx is available
    if not shutil.which("npx"):
        # Fall back to simple text prompt
        return _simple_prompt(provider, env_var, model_id)

    # Show the Codex-style prompt
    print(f"\n{model_id} requires an API key from {provider.title()}.")
    print("Sign in to get an API key or paste one you already have.")
    print("\033[90m[use arrows to move, enter to select]\033[0m\n")

    # Launch the npm wizard with specific context
    try:
        env = os.environ.copy()
        env["EMBER_SETUP_PROVIDER"] = provider
        env["EMBER_SETUP_MODEL"] = model_id
        env["EMBER_SETUP_CONTEXT"] = "missing-key"

        result = subprocess.run(["npx", "-y", "@ember-ai/setup"], env=env, capture_output=False)

        if result.returncode == 0:
            # Check if key is now available through context
            from ember._internal.context import EmberContext

            ctx = EmberContext.current()
            api_key = ctx.get_credential(provider, env_var)
            if api_key:
                return api_key

    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
    except Exception:
        # Fall back to simple prompt
        return _simple_prompt(provider, env_var, model_id)

    return None


def _simple_prompt(provider: str, env_var: str, model_id: str) -> Optional[str]:
    """Simple fallback prompt when npm not available."""
    print(f"\n{model_id} requires an API key.")
    print(f"Enter your {provider.title()} API key: ", end="", flush=True)

    api_key = input().strip()

    if api_key:
        # Set for current session
        os.environ[env_var] = api_key
        print("\n✓ API key set for this session.")
        return api_key

    return None


def format_non_interactive_error(provider: str, env_var: str, model_id: str) -> str:
    """Format error for non-interactive environments."""
    urls = {
        "openai": "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com/api-keys",
        "google": "https://makersuite.google.com/app/apikey",
    }

    url = urls.get(provider, f"https://{provider}.com")

    return (
        f"No API key found for {model_id}.\n\n"
        f"To fix this, choose one:\n\n"
        f"Option 1: Run interactive setup (recommended)\n"
        f"   npx @ember-ai/setup\n\n"
        f"Option 2: Set environment variable\n"
        f'   export {env_var}="your-api-key"\n\n'
        f"Option 3: Save to config (like AWS CLI)\n"
        f"   ember configure set {provider}.api_key YOUR_KEY\n\n"
        f"Get your API key from: {url}"
    )
