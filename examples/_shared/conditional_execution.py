"""Conditional execution decorator for Ember examples.

This module provides a decorator that allows examples to run in two modes:
1. Real mode - Makes actual API calls when keys are available
2. Simulated mode - Shows realistic output without API calls

This ensures all examples are runnable and educational regardless of configuration.
"""

import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class SimulatedResponse:
    """Mock response object for simulated LLM calls."""

    text: str
    model_id: str = "simulated-model"
    usage: Dict[str, Any] = None

    def __post_init__(self):
        if self.usage is None:
            # Realistic token counts based on text length
            prompt_tokens = len(self.text.split()) * 2
            completion_tokens = len(self.text.split())
            self.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": 0.0,  # Free in simulation!
            }


class ConditionalLLMDecorator:
    """Decorator for conditional execution of LLM examples."""

    # Track if we've shown setup instructions
    _setup_shown = False

    def __init__(
        self,
        providers: Optional[List[str]] = None,
        show_setup: bool = True,
        record_metrics: bool = True,
    ):
        """Initialize the decorator.

        Args:
            providers: List of provider names to check (e.g., ["openai", "anthropic"])
            show_setup: Whether to show setup instructions on first run
            record_metrics: Whether to record execution metrics
        """
        self.providers = providers or ["openai", "anthropic", "google"]
        self.show_setup = show_setup
        self.record_metrics = record_metrics
        self.metrics = []

    def _check_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are available."""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "cohere": "COHERE_API_KEY",
        }

        available = {}
        for provider in self.providers:
            env_var = key_mapping.get(provider, f"{provider.upper()}_API_KEY")
            available[provider] = bool(os.environ.get(env_var))

        return available

    def _show_setup_instructions(self, missing_providers: List[str]):
        """Show setup instructions once per session."""
        if not self.show_setup or ConditionalLLMDecorator._setup_shown:
            return

        ConditionalLLMDecorator._setup_shown = True

        print("\n" + "=" * 60)
        print("🔧 Running in simulated mode (no API keys detected)")
        print("=" * 60)
        print("\nTo run this example with real API calls, set one of:")

        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }

        for provider in missing_providers:
            env_var = key_mapping.get(provider, f"{provider.upper()}_API_KEY")
            print(f"  export {env_var}='your-key-here'")

        print("\nSimulated output will demonstrate the expected behavior.")
        print("=" * 60 + "\n")

    def __call__(
        self,
        real_execution: Optional[Callable] = None,
        simulated_execution: Optional[Callable] = None,
    ):
        """Decorate a function for conditional execution.

        Args:
            real_execution: Function to run with API keys (defaults to decorated function)
            simulated_execution: Function to run without API keys
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Check API key availability
                available_keys = self._check_api_keys()
                has_any_key = any(available_keys.values())

                # Record start time
                start_time = time.time()

                try:
                    if has_any_key:
                        # Run real execution
                        execution_func = real_execution or func
                        result = execution_func(*args, **kwargs)
                        mode = "real"
                    else:
                        # Show setup instructions if needed
                        missing = [
                            p for p, avail in available_keys.items() if not avail
                        ]
                        self._show_setup_instructions(missing)

                        # Run simulated execution
                        if simulated_execution:
                            result = simulated_execution(*args, **kwargs)
                        else:
                            # Use the original function in simulation mode
                            # Inject a flag so the function knows it's simulated
                            kwargs["_simulated_mode"] = True
                            result = func(*args, **kwargs)
                        mode = "simulated"

                    # Record metrics
                    if self.record_metrics:
                        duration = time.time() - start_time
                        self.metrics.append(
                            {
                                "function": func.__name__,
                                "mode": mode,
                                "duration": duration,
                                "providers_checked": list(available_keys.keys()),
                                "providers_available": [
                                    k for k, v in available_keys.items() if v
                                ],
                            }
                        )

                    return result

                except Exception as e:
                    # Record failure metrics
                    if self.record_metrics:
                        duration = time.time() - start_time
                        self.metrics.append(
                            {
                                "function": func.__name__,
                                "mode": "failed",
                                "duration": duration,
                                "error": str(e),
                            }
                        )
                    raise

            # Attach metrics access to wrapper
            wrapper.get_metrics = lambda: self.metrics
            return wrapper

        # Handle being called with or without arguments
        if real_execution is None and callable(simulated_execution):
            # Called as @conditional_llm(simulated_execution=func)
            return decorator
        elif callable(real_execution) and simulated_execution is None:
            # Called as @conditional_llm on a function directly
            return decorator(real_execution)
        else:
            # Called as @conditional_llm() or with both arguments
            return decorator


# Convenience instance
conditional_llm = ConditionalLLMDecorator
