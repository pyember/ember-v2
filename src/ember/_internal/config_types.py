"""Type definitions for Ember configuration.

Provides TypedDict definitions for configuration structure to enable
better type checking and IDE support.
"""

from typing import TypedDict, Optional, Dict, Any


class ProviderConfig(TypedDict, total=False):
    """Configuration for a single provider."""
    api_key: Optional[str]
    base_url: Optional[str]
    organization_id: Optional[str]
    default_model: Optional[str]
    timeout: Optional[int]
    max_retries: Optional[int]


class ModelsConfig(TypedDict, total=False):
    """Configuration for model defaults."""
    default: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float


class LoggingConfig(TypedDict, total=False):
    """Configuration for logging behavior."""
    level: str
    format: str
    file: Optional[str]
    components: Dict[str, str]


class EmberConfig(TypedDict, total=False):
    """Complete Ember configuration structure."""
    version: str
    models: ModelsConfig
    providers: Dict[str, ProviderConfig]
    logging: LoggingConfig
    data: Dict[str, Any]  # Dataset-specific config
    xcs: Dict[str, Any]   # XCS-specific config
    
    
class CredentialEntry(TypedDict):
    """Structure for a single credential entry."""
    api_key: str
    created_at: str
    last_used: Optional[str]


# Type alias for credentials file structure
CredentialsDict = Dict[str, CredentialEntry]