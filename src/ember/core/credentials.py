"""Credential management for Ember.

Follows the same pattern as AWS CLI, gcloud, etc:
- Credentials stored in ~/.ember/credentials
- Environment variables take precedence
- Secure file permissions (0600)
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


class CredentialManager:
    """Manages API key storage and retrieval."""
    
    def __init__(self):
        self.config_dir = Path.home() / '.ember'
        self.credentials_file = self.config_dir / 'credentials'
        self.config_file = self.config_dir / 'config.json'
        
    def get(self, provider: str) -> Optional[str]:
        """Get API key for provider from credentials file only.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            API key if found in credentials file, None otherwise
        """
        try:
            if self.credentials_file.exists():
                with open(self.credentials_file, 'r') as f:
                    credentials = json.load(f)
                    provider_creds = credentials.get(provider, {})
                    return provider_creds.get('api_key')
        except Exception:
            # Invalid JSON or permissions issue
            pass
            
        return None
    
    def get_api_key(self, provider: str, env_var: str) -> Optional[str]:
        """Get API key with precedence: env var > credentials file.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            env_var: Environment variable name
            
        Returns:
            API key if found, None otherwise
        """
        # 1. Check environment variable (highest precedence)
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key
            
        # 2. Check credentials file
        try:
            if self.credentials_file.exists():
                with open(self.credentials_file, 'r') as f:
                    credentials = json.load(f)
                    provider_creds = credentials.get(provider, {})
                    return provider_creds.get('api_key')
        except Exception:
            # Invalid JSON or permissions issue
            pass
            
        return None
        
    def store(self, provider: str, api_key: str) -> None:
        """Store API key in credentials file.
        
        Args:
            provider: Provider name
            api_key: API key to save
        """
        self.save_api_key(provider, api_key)
    
    def delete(self, provider: str) -> bool:
        """Delete API key for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            True if deleted, False if not found
        """
        if not self.credentials_file.exists():
            return False
            
        try:
            with open(self.credentials_file, 'r') as f:
                credentials = json.load(f)
            
            if provider in credentials:
                del credentials[provider]
                
                # Write back
                with open(self.credentials_file, 'w') as f:
                    json.dump(credentials, f, indent=2)
                
                # Maintain secure permissions
                os.chmod(self.credentials_file, 0o600)
                return True
                
        except Exception:
            pass
            
        return False
    
    def list_providers(self) -> list[str]:
        """List all providers with stored credentials.
        
        Returns:
            List of provider names
        """
        if not self.credentials_file.exists():
            return []
            
        try:
            with open(self.credentials_file, 'r') as f:
                credentials = json.load(f)
            return list(credentials.keys())
        except Exception:
            return []
    
    def save_api_key(self, provider: str, api_key: str) -> None:
        """Save API key to credentials file atomically with secure permissions.
        
        Args:
            provider: Provider name
            api_key: API key to save
        """
        # Create directory if needed
        self.config_dir.mkdir(exist_ok=True)
        
        # Load existing credentials
        credentials = {}
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, 'r') as f:
                    credentials = json.load(f)
            except Exception:
                pass
                
        # Update credentials
        from datetime import datetime
        credentials[provider] = {
            'api_key': api_key,
            'created_at': datetime.now().isoformat()
        }
        
        # Write atomically with secure permissions
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.config_dir,
            prefix='.credentials_',
            suffix='.tmp'
        )
        
        try:
            # Write with secure permissions from the start
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            # Set permissions before rename (belt and suspenders)
            os.chmod(temp_path, 0o600)
            
            # Atomic rename
            Path(temp_path).replace(self.credentials_file)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
        
    def get_config(self) -> Dict[str, Any]:
        """Load general configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


# Global instance
_credential_manager = CredentialManager()


def get_api_key(provider: str, env_var: str) -> Optional[str]:
    """Get API key for a provider.
    
    Checks in order:
    1. Environment variable
    2. ~/.ember/credentials file
    
    Args:
        provider: Provider name (e.g., 'openai')
        env_var: Environment variable name (e.g., 'OPENAI_API_KEY')
        
    Returns:
        API key if found, None otherwise
    """
    return _credential_manager.get_api_key(provider, env_var)