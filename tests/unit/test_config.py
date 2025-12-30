import os
import pytest
from src import config

def test_default_config_values():
    # Assuming config is loaded, check for default keys
    assert hasattr(config, "WINDOWS_HOST_IP")
    assert hasattr(config, "OLLAMA_BASE_URL")
    assert hasattr(config, "EMBEDDING_MODEL")
    assert hasattr(config, "LANGUAGE_MODEL")
    assert hasattr(config, "QDRANT_HOST")
    assert hasattr(config, "QDRANT_PORT")
    assert hasattr(config, "COLLECTION_NAME")

def test_environment_variable_override(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "test-host")
    monkeypatch.setenv("QDRANT_PORT", "9999")
    
    # Reload config to pick up env vars
    import importlib
    importlib.reload(config)
    
    assert config.QDRANT_HOST == "test-host"
    assert config.QDRANT_PORT == 9999
