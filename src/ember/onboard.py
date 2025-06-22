"""Ember interactive onboarding experience.

This module provides a delightful, conversational onboarding flow that helps
new users get from zero to their first successful AI interaction in under 60 seconds.

Example:
    >>> from ember import onboard
    >>> onboard.start()
    
Or from command line:
    $ python -m ember.onboard
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ember._internal.exceptions import EmberError


class OnboardingState(Enum):
    """Tracks user's progress through onboarding."""
    NEW_USER = "new"
    PROVIDER_SELECTED = "provider_selected"
    API_KEY_CONFIGURED = "api_configured"
    FIRST_CALL_MADE = "first_call"
    EXAMPLE_CREATED = "example_created"
    COMPLETED = "completed"


@dataclass
class UserProfile:
    """Stores user's onboarding choices and progress."""
    state: OnboardingState = OnboardingState.NEW_USER
    provider: Optional[str] = None
    examples_created: List[str] = None
    first_call_timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.examples_created is None:
            self.examples_created = []
            
    def save(self, path: Path):
        """Persist user profile."""
        data = {
            'state': self.state.value,
            'provider': self.provider,
            'examples_created': self.examples_created,
            'first_call_timestamp': self.first_call_timestamp
        }
        path.write_text(json.dumps(data, indent=2))
        
    @classmethod
    def load(cls, path: Path) -> 'UserProfile':
        """Load user profile from disk."""
        if not path.exists():
            return cls()
            
        data = json.loads(path.read_text())
        return cls(
            state=OnboardingState(data.get('state', 'new')),
            provider=data.get('provider'),
            examples_created=data.get('examples_created', []),
            first_call_timestamp=data.get('first_call_timestamp')
        )


class InteractiveOnboarding:
    """Main onboarding experience coordinator."""
    
    def __init__(self):
        self.config_dir = Path.home() / '.ember'
        self.config_dir.mkdir(exist_ok=True)
        self.profile_path = self.config_dir / 'onboarding.json'
        self.profile = UserProfile.load(self.profile_path)
        
    def start(self):
        """Start appropriate onboarding flow based on user state."""
        if self.profile.state == OnboardingState.COMPLETED:
            return self._quick_start_menu()
        else:
            return self._full_onboarding()
            
    def _full_onboarding(self):
        """Run the complete onboarding experience."""
        # Delegate to the POC implementation for now
        # In production, this would be the full rich terminal experience
        from ember_onboard_poc import main
        main()
        
    def _quick_start_menu(self):
        """Show menu for returning users."""
        print("🎭 Welcome back to Ember!\n")
        print("What would you like to do?")
        print("1. Create a new example")
        print("2. Explore models")  
        print("3. View documentation")
        print("4. Join community")
        print("5. Exit")
        
        choice = input("\n→ Choose [1-5]: ").strip()
        
        actions = {
            '1': self._create_example,
            '2': self._explore_models,
            '3': self._open_docs,
            '4': self._show_community,
            '5': lambda: print("Happy building! 🚀")
        }
        
        if choice in actions:
            actions[choice]()
            
    def _create_example(self):
        """Create a new example based on user interest."""
        print("\nWhat would you like to build?")
        print("1. Chatbot")
        print("2. Content generator")
        print("3. Data analyzer")
        print("4. Custom pipeline")
        
        # Implementation would generate appropriate examples
        
    def _explore_models(self):
        """Interactive model exploration."""
        from ember.models import ModelRegistry
        registry = ModelRegistry()
        
        print("\nAvailable models:")
        for i, model in enumerate(registry.list_models(), 1):
            print(f"{i}. {model}")
            
    def _open_docs(self):
        """Open documentation in browser."""
        import webbrowser
        webbrowser.open("https://ember.ai/docs")
        
    def _show_community(self):
        """Show community links."""
        print("\n🌟 Join the Ember community:")
        print("GitHub: https://github.com/anthropics/ember")
        print("Discord: https://discord.gg/ember-ai")
        print("Twitter: https://twitter.com/EmberAI")


def start():
    """Main entry point for onboarding experience."""
    onboarding = InteractiveOnboarding()
    onboarding.start()


def is_first_run() -> bool:
    """Check if this is the user's first time using Ember."""
    config_dir = Path.home() / '.ember'
    return not (config_dir / 'onboarding.json').exists()


def suggest_onboarding():
    """Suggest running onboarding if it's the first run."""
    if is_first_run():
        print("👋 New to Ember? Run 'python -m ember.onboard' for a guided setup!")
        return True
    return False


# CLI entry point
if __name__ == "__main__":
    start()