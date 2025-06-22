#!/usr/bin/env python3
"""
Ember Onboarding Experience - Proof of Concept

This demonstrates the delightful, interactive onboarding flow.
"""

import os
import sys
import time
import textwrap
from typing import Optional, Dict, List
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class OnboardingWizard:
    """Interactive onboarding experience for Ember."""
    
    PROVIDERS = {
        '1': {
            'name': 'OpenAI',
            'models': 'GPT-4, GPT-3.5',
            'description': 'Most popular',
            'url': 'https://platform.openai.com/api-keys',
            'env_var': 'OPENAI_API_KEY',
            'test_model': 'gpt-3.5-turbo'
        },
        '2': {
            'name': 'Anthropic',
            'models': 'Claude 3',
            'description': 'Best for complex reasoning',
            'url': 'https://console.anthropic.com/account/keys',
            'env_var': 'ANTHROPIC_API_KEY',
            'test_model': 'claude-3-haiku'
        },
        '3': {
            'name': 'Google',
            'models': 'Gemini',
            'description': 'Great free tier',
            'url': 'https://makersuite.google.com/app/apikey',
            'env_var': 'GOOGLE_API_KEY',
            'test_model': 'gemini-pro'
        },
        '4': {
            'name': 'Local Models',
            'models': 'Llama, Mistral',
            'description': 'No API key needed',
            'url': None,
            'env_var': None,
            'test_model': 'local'
        }
    }
    
    def __init__(self):
        self.api_keys = {}
        self.selected_provider = None
        
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')
        
    def print_header(self):
        """Print welcome header with ASCII art."""
        self.clear_screen()
        print(f"{Colors.BOLD}{Colors.MAGENTA}")
        print("╔═══════════════════════════╗")
        print("║   🎭 Welcome to Ember!    ║")
        print("╚═══════════════════════════╝")
        print(f"{Colors.END}\n")
        
    def type_text(self, text: str, delay: float = 0.02):
        """Simulate typing effect."""
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()
        
    def prompt(self, text: str, choices: Optional[List[str]] = None) -> str:
        """Interactive prompt with optional choices."""
        print(f"\n{Colors.CYAN}→ {text}{Colors.END}", end='')
        if choices:
            print(f" [{'/'.join(choices)}]", end='')
        print(": ", end='')
        return input().strip()
        
    def success(self, text: str):
        """Print success message."""
        print(f"{Colors.GREEN}✓ {text}{Colors.END}")
        
    def error(self, text: str):
        """Print error message."""
        print(f"{Colors.RED}✗ {text}{Colors.END}")
        
    def info(self, text: str):
        """Print info message."""
        print(f"{Colors.YELLOW}💡 {text}{Colors.END}")
        
    def step1_welcome(self):
        """Initial welcome and environment check."""
        self.print_header()
        self.type_text("I'll help you get started in just a few steps.")
        time.sleep(0.5)
        print("\nFirst, let me check your environment...\n")
        time.sleep(0.5)
        
        # Simulate environment checks
        checks = [
            ("Python 3.10+ detected", True),
            ("Required packages installed", True),
            ("No API keys found", False)
        ]
        
        for check, status in checks:
            time.sleep(0.3)
            if status:
                self.success(check)
            else:
                self.error(check)
                
    def step2_choose_provider(self):
        """Provider selection."""
        print("\nLet's fix that! Which AI provider would you like to start with?\n")
        
        for key, provider in self.PROVIDERS.items():
            print(f"{key}. {Colors.BOLD}{provider['name']}{Colors.END} ({provider['models']}) - {provider['description']}")
        
        choice = self.prompt("Choose", list(self.PROVIDERS.keys()))
        
        if choice in self.PROVIDERS:
            self.selected_provider = self.PROVIDERS[choice]
            return True
        return False
        
    def step3_setup_api_key(self):
        """API key configuration."""
        provider = self.selected_provider
        
        if provider['url'] is None:
            print(f"\n{Colors.GREEN}Great choice! Local models don't need API keys.{Colors.END}")
            self.info("We'll help you set up Ollama or similar in the next step.")
            return True
            
        print(f"\nGreat choice! Let's set up {provider['name']}.\n")
        
        print(f"📎 Need an API key? Here's how:")
        print(f"   1. Visit: {Colors.UNDERLINE}{provider['url']}{Colors.END}")
        print(f"   2. Click 'Create new secret key'")
        print(f"   3. Copy the key (starts with...)\n")
        
        # Simulate API key input
        api_key = self.prompt("Paste your API key (hidden)")
        
        if api_key:
            self.success("Valid API key detected!")
            
            print("\nWould you like me to:")
            print("1. Save to .env file (recommended)")
            print("2. Set environment variable")
            print("3. Use for this session only")
            
            save_choice = self.prompt("Choose", ['1', '2', '3'])
            
            if save_choice == '1':
                self.save_to_env_file(provider['env_var'], api_key)
                self.success("Saved to .env file")
                self.info("Tip: Ember will automatically load this file")
            
            return True
        return False
        
    def save_to_env_file(self, var_name: str, value: str):
        """Save API key to .env file."""
        env_path = Path('.env')
        
        # Read existing content
        existing = ""
        if env_path.exists():
            existing = env_path.read_text()
            
        # Update or append
        if var_name in existing:
            lines = existing.split('\n')
            for i, line in enumerate(lines):
                if line.startswith(f"{var_name}="):
                    lines[i] = f"{var_name}={value}"
                    break
            existing = '\n'.join(lines)
        else:
            existing += f"\n{var_name}={value}\n"
            
        env_path.write_text(existing.strip() + '\n')
        
    def step4_first_magic(self):
        """First successful API call."""
        print("\nAwesome! Let's try something fun:\n")
        print(f"{Colors.MAGENTA}🎭 Running your first Ember command...{Colors.END}\n")
        
        time.sleep(1)
        
        model = self.selected_provider['test_model']
        print(f'>>> ember.models("{model}", "Write a haiku about AI")\n')
        
        time.sleep(1.5)
        
        # Simulated response
        haiku = """Silent circuits think,
Digital dreams take new form—
Future blooms in code."""
        
        for line in haiku.split('\n'):
            print(f"{Colors.CYAN}{line}{Colors.END}")
            time.sleep(0.5)
            
        print(f"\n{Colors.GREEN}✨ Nice! You just made your first AI call with Ember.{Colors.END}\n")
        
        time.sleep(1)
        
        print("Want to try something more advanced?")
        print("1. Build a simple chatbot")
        print("2. Create an AI pipeline")
        print("3. Explore more models")
        print("4. Join our community")
        
        return self.prompt("Choose", ['1', '2', '3', '4'])
        
    def step5_create_example(self, choice: str):
        """Create example based on user choice."""
        if choice == '1':
            self.create_chatbot_example()
        elif choice == '2':
            self.create_pipeline_example()
        elif choice == '3':
            self.explore_models()
        elif choice == '4':
            self.show_community_links()
            
    def create_chatbot_example(self):
        """Create a simple chatbot example."""
        print("\nLet's build a simple chatbot!\n")
        print(f"{Colors.YELLOW}📝 Creating chatbot.py...{Colors.END}\n")
        
        time.sleep(1)
        
        code = '''from ember import models

def chat():
    print("🤖 Chatbot ready! Type 'quit' to exit.\\n")
    
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        # Add context from history
        prompt = f"Previous: {history[-2:]}\\nUser: {user_input}\\nAssistant:"
        
        response = models("gpt-3.5-turbo", prompt)
        print(f"Bot: {response}\\n")
        
        history.append(f"User: {user_input}")
        history.append(f"Bot: {response}")

if __name__ == "__main__":
    chat()'''
        
        # Display code with syntax highlighting (simulated)
        for line in code.split('\n'):
            if line.strip().startswith(('from', 'import', 'def', 'if', 'while')):
                print(f"{Colors.BLUE}{line}{Colors.END}")
            elif line.strip().startswith('#'):
                print(f"{Colors.GREEN}{line}{Colors.END}")
            else:
                print(line)
                
        # Save the file
        with open('chatbot.py', 'w') as f:
            f.write(code)
            
        print(f"\n{Colors.GREEN}✓ Created chatbot.py{Colors.END}")
        print(f"\nRun it with: {Colors.BOLD}python chatbot.py{Colors.END}")
        
    def show_community_links(self):
        """Show community connection options."""
        print(f"\n{Colors.BOLD}🎉 You're all set up with Ember!{Colors.END}\n")
        print("Before you go, want to:\n")
        
        links = [
            ("⭐ Star us on GitHub?", "https://github.com/anthropics/ember"),
            ("💬 Join our Discord community? (500+ members)", "https://discord.gg/ember-ai"),
            ("📬 Get weekly AI tips and updates?", "ember.ai/newsletter"),
            ("🐦 Follow @EmberAI for updates", "https://twitter.com/EmberAI")
        ]
        
        for text, url in links:
            print(f"{text}")
            print(f"   → {Colors.UNDERLINE}{url}{Colors.END}\n")
            
    def run(self):
        """Run the complete onboarding flow."""
        try:
            # Step 1: Welcome
            self.step1_welcome()
            
            # Step 2: Choose provider
            if not self.step2_choose_provider():
                return
                
            # Step 3: Setup API key
            if not self.step3_setup_api_key():
                return
                
            # Step 4: First magic moment
            choice = self.step4_first_magic()
            
            # Step 5: Create example or show community
            self.step5_create_example(choice)
            
            # Final message
            print(f"\n{Colors.BOLD}{Colors.GREEN}Happy building with Ember! 🚀{Colors.END}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Onboarding paused. Run 'ember' anytime to continue!{Colors.END}\n")
            

def main():
    """Entry point for the onboarding experience."""
    wizard = OnboardingWizard()
    wizard.run()


if __name__ == "__main__":
    main()