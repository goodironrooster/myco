# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Model provider support for myco.

Supports multiple local and remote model providers:
- Ollama (http://localhost:11434)
- LM Studio (http://localhost:1234)
- OpenAI-compatible APIs
- Custom endpoints

All providers maintain myco's local-only, privacy-focused philosophy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests
import yaml


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    name: str
    base_url: str
    api_key: Optional[str] = None
    model_param: str = "model"
    supports_streaming: bool = True
    health_endpoint: str = "/health"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model_param": self.model_param,
            "supports_streaming": self.supports_streaming,
            "health_endpoint": self.health_endpoint,
        }


# Pre-configured providers
PROVIDERS = {
    "lmstudio": ProviderConfig(
        name="LM Studio",
        base_url="http://localhost:1234",
        api_key=None,  # LM Studio doesn't require API key
        supports_streaming=True,
        health_endpoint="/health"
    ),
    "ollama": ProviderConfig(
        name="Ollama",
        base_url="http://localhost:11434",
        api_key=None,  # Ollama doesn't require API key
        supports_streaming=True,
        health_endpoint="/api/tags"  # Ollama uses different endpoint
    ),
    "openai": ProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key=None,  # Will be set from environment
        supports_streaming=True,
        health_endpoint=None  # OpenAI doesn't have health endpoint
    ),
    "localai": ProviderConfig(
        name="LocalAI",
        base_url="http://localhost:8080",
        api_key=None,
        supports_streaming=True,
        health_endpoint="/healthz"
    ),
}


def detect_provider(base_url: str) -> Optional[str]:
    """Detect which provider is running at a URL.
    
    Args:
        base_url: Base URL to check
        
    Returns:
        Provider name or None if unknown
    """
    # Check against known provider URLs
    for name, config in PROVIDERS.items():
        if config.base_url in base_url:
            return name
    
    # Try to detect by probing endpoints
    try:
        # Try Ollama endpoint
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            return "ollama"
    except requests.RequestException:
        pass
    
    try:
        # Try LM Studio endpoint
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            return "lmstudio"
    except requests.RequestException:
        pass
    
    return None


def get_provider_config(provider_name: str) -> Optional[ProviderConfig]:
    """Get configuration for a provider.
    
    Args:
        provider_name: Name of provider
        
    Returns:
        ProviderConfig or None if not found
    """
    if provider_name in PROVIDERS:
        return PROVIDERS[provider_name]
    
    # Try case-insensitive match
    for name, config in PROVIDERS.items():
        if name.lower() == provider_name.lower():
            return config
    
    return None


def check_provider_health(base_url: str, timeout: int = 5) -> tuple[bool, str]:
    """Check if a model provider is healthy.
    
    Args:
        base_url: Base URL of provider
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (is_healthy, status_message)
    """
    # Try to detect provider
    provider_name = detect_provider(base_url)
    config = get_provider_config(provider_name) if provider_name else None
    
    # Determine health endpoint
    if config and config.health_endpoint:
        health_url = f"{base_url}{config.health_endpoint}"
    else:
        # Try common health endpoints
        health_endpoints = ["/health", "/api/tags", "/healthz", "/v1/models"]
        health_url = None
        
        for endpoint in health_endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=timeout)
                if response.status_code == 200:
                    health_url = f"{base_url}{endpoint}"
                    break
            except requests.RequestException:
                continue
        
        if not health_url:
            # Fall back to base URL
            health_url = base_url
    
    # Check health
    try:
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            return True, f"Provider healthy at {base_url}"
        else:
            return False, f"Provider returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {base_url}"
    except requests.exceptions.Timeout:
        return False, f"Connection timed out to {base_url}"
    except requests.RequestException as e:
        return False, f"Provider error: {e}"


def list_available_models(base_url: str, timeout: int = 5) -> list[str]:
    """List available models from a provider.
    
    Args:
        base_url: Base URL of provider
        timeout: Request timeout in seconds
        
    Returns:
        List of model names
    """
    try:
        # Try OpenAI-compatible endpoint
        response = requests.get(f"{base_url}/v1/models", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            return [m.get("id", "") for m in models if m.get("id")]
        
        # Try Ollama endpoint
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
            
    except requests.RequestException:
        pass
    
    return []


def create_provider_session(provider_name: str, base_url: str, api_key: Optional[str] = None) -> ProviderConfig:
    """Create a custom provider configuration.

    Args:
        provider_name: Name for this provider
        base_url: Base URL
        api_key: Optional API key

    Returns:
        ProviderConfig for the custom provider
    """
    return ProviderConfig(
        name=provider_name,
        base_url=base_url,
        api_key=api_key,
        supports_streaming=True,
        health_endpoint="/health"
    )


def get_config_path() -> Path:
    """Get path to myco config directory.
    
    Returns:
        Path to ~/.myco directory
    """
    return Path.home() / ".myco"


def get_providers_config_path() -> Path:
    """Get path to providers config file.
    
    Returns:
        Path to ~/.myco/providers.yaml
    """
    return get_config_path() / "providers.yaml"


def load_custom_providers() -> dict[str, ProviderConfig]:
    """Load custom providers from config file.
    
    Returns:
        Dict of provider name -> ProviderConfig
    """
    config_path = get_providers_config_path()
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        if not data or "providers" not in data:
            return {}
        
        providers = {}
        for name, config in data["providers"].items():
            providers[name] = ProviderConfig(
                name=name,
                base_url=config.get("base_url", ""),
                api_key=config.get("api_key"),
                supports_streaming=config.get("supports_streaming", True),
                health_endpoint=config.get("health_endpoint", "/health")
            )
        
        return providers
    except (yaml.YAMLError, IOError):
        return {}


def save_custom_providers(providers: dict[str, ProviderConfig]) -> None:
    """Save custom providers to config file.
    
    Args:
        providers: Dict of provider name -> ProviderConfig
    """
    config_path = get_providers_config_path()
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "providers": {
            name: {
                "base_url": config.base_url,
                "api_key": config.api_key,
                "supports_streaming": config.supports_streaming,
                "health_endpoint": config.health_endpoint
            }
            for name, config in providers.items()
        }
    }
    
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def get_all_providers() -> dict[str, ProviderConfig]:
    """Get all providers (built-in + custom).
    
    Returns:
        Dict of all available providers
    """
    # Start with built-in providers
    all_providers = PROVIDERS.copy()
    
    # Add custom providers (can override built-ins)
    custom = load_custom_providers()
    all_providers.update(custom)
    
    return all_providers


def add_custom_provider(name: str, base_url: str, api_key: Optional[str] = None) -> ProviderConfig:
    """Add a custom provider to config.
    
    Args:
        name: Provider name
        base_url: Base URL
        api_key: Optional API key
        
    Returns:
        Created ProviderConfig
    """
    config = create_provider_session(name, base_url, api_key)
    
    # Load existing, add new, save
    providers = load_custom_providers()
    providers[name] = config
    save_custom_providers(providers)
    
    return config


def remove_custom_provider(name: str) -> bool:
    """Remove a custom provider from config.
    
    Args:
        name: Provider name to remove
        
    Returns:
        True if removed, False if not found
    """
    providers = load_custom_providers()
    
    if name not in providers:
        return False
    
    del providers[name]
    save_custom_providers(providers)
    return True
