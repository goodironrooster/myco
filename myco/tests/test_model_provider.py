# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Tests for myco model_provider module."""

import pytest
from myco.model_provider import (
    ProviderConfig,
    PROVIDERS,
    detect_provider,
    get_provider_config,
    check_provider_health,
    list_available_models,
    create_provider_session,
)


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_create_provider_config(self):
        """Test creating provider config."""
        config = ProviderConfig(
            name="Test Provider",
            base_url="http://localhost:8080",
            api_key="test-key"
        )

        assert config.name == "Test Provider"
        assert config.base_url == "http://localhost:8080"
        assert config.api_key == "test-key"
        assert config.supports_streaming is True

    def test_provider_config_to_dict(self):
        """Test converting config to dict."""
        config = ProviderConfig(
            name="Test",
            base_url="http://test"
        )

        d = config.to_dict()

        assert d["name"] == "Test"
        assert d["base_url"] == "http://test"
        assert d["api_key"] is None


class TestPreconfiguredProviders:
    """Tests for pre-configured providers."""

    def test_lmstudio_provider(self):
        """Test LM Studio provider config."""
        config = PROVIDERS["lmstudio"]

        assert config.name == "LM Studio"
        assert "localhost:1234" in config.base_url
        assert config.api_key is None

    def test_ollama_provider(self):
        """Test Ollama provider config."""
        config = PROVIDERS["ollama"]

        assert config.name == "Ollama"
        assert "localhost:11434" in config.base_url
        assert config.api_key is None

    def test_openai_provider(self):
        """Test OpenAI provider config."""
        config = PROVIDERS["openai"]

        assert config.name == "OpenAI"
        assert "api.openai.com" in config.base_url

    def test_localai_provider(self):
        """Test LocalAI provider config."""
        config = PROVIDERS["localai"]

        assert config.name == "LocalAI"
        assert "localhost:8080" in config.base_url


class TestDetectProvider:
    """Tests for provider detection."""

    def test_detect_lmstudio(self):
        """Test detecting LM Studio by URL."""
        provider = detect_provider("http://localhost:1234")
        assert provider == "lmstudio"

    def test_detect_ollama(self):
        """Test detecting Ollama by URL."""
        provider = detect_provider("http://localhost:11434")
        assert provider == "ollama"

    def test_detect_openai(self):
        """Test detecting OpenAI by URL."""
        provider = detect_provider("https://api.openai.com/v1")
        assert provider == "openai"

    def test_detect_unknown(self):
        """Test detecting unknown provider."""
        provider = detect_provider("http://unknown:9999")
        assert provider is None


class TestGetProviderConfig:
    """Tests for getting provider config."""

    def test_get_existing_provider(self):
        """Test getting config for existing provider."""
        config = get_provider_config("lmstudio")

        assert config is not None
        assert config.name == "LM Studio"

    def test_get_case_insensitive(self):
        """Test case-insensitive provider lookup."""
        config = get_provider_config("LMSTUDIO")

        assert config is not None
        assert config.name == "LM Studio"

    def test_get_unknown_provider(self):
        """Test getting config for unknown provider."""
        config = get_provider_config("unknown")

        assert config is None


class TestCheckProviderHealth:
    """Tests for provider health checking."""

    def test_health_check_connection_error(self):
        """Test health check with connection error."""
        is_healthy, msg = check_provider_health("http://nonexistent:9999", timeout=1)

        assert is_healthy is False
        assert "Cannot connect" in msg or "timed out" in msg.lower()


class TestListAvailableModels:
    """Tests for listing available models."""

    def test_list_models_connection_error(self):
        """Test listing models with connection error."""
        models = list_available_models("http://nonexistent:9999", timeout=1)

        assert models == []


class TestCreateProviderSession:
    """Tests for creating custom provider sessions."""

    def test_create_custom_provider(self):
        """Test creating custom provider config."""
        config = create_provider_session(
            "Custom",
            "http://custom:8080",
            "api-key-123"
        )

        assert config.name == "Custom"
        assert config.base_url == "http://custom:8080"
        assert config.api_key == "api-key-123"

    def test_create_provider_defaults(self):
        """Test creating provider with defaults."""
        config = create_provider_session("Test", "http://test")

        assert config.supports_streaming is True
        assert config.health_endpoint == "/health"
