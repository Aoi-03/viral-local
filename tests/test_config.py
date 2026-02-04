"""
Tests for configuration management.
"""

import pytest
import tempfile
import os
from pathlib import Path
from viral_local.config import SystemConfig, ConfigManager


class TestSystemConfig:
    """Test SystemConfig model."""
    
    def test_valid_config(self):
        """Test creating a valid SystemConfig."""
        config = SystemConfig(
            gemini_api_key="test_key_123",
            whisper_model_size="base",
            tts_engine="edge-tts",
            supported_languages=["hi", "bn", "ta"]
        )
        
        assert config.gemini_api_key == "test_key_123"
        assert config.whisper_model_size == "base"
        assert config.tts_engine == "edge-tts"
        assert config.supported_languages == ["hi", "bn", "ta"]
    
    def test_missing_api_key(self):
        """Test SystemConfig with missing API key."""
        with pytest.raises(ValueError, match="Gemini API key is required"):
            SystemConfig(gemini_api_key="")
    
    def test_invalid_whisper_model(self):
        """Test SystemConfig with invalid Whisper model."""
        with pytest.raises(ValueError, match="Invalid Whisper model size"):
            SystemConfig(
                gemini_api_key="test_key",
                whisper_model_size="invalid"
            )
    
    def test_invalid_tts_engine(self):
        """Test SystemConfig with invalid TTS engine."""
        with pytest.raises(ValueError, match="Invalid TTS engine"):
            SystemConfig(
                gemini_api_key="test_key",
                tts_engine="invalid"
            )
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = SystemConfig(
            gemini_api_key="test_key",
            whisper_model_size="base"
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["gemini_api_key"] == "test_key"
        assert config_dict["whisper_model_size"] == "base"


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["VIRAL_LOCAL_GEMINI_API_KEY"] = "env_test_key"
        os.environ["VIRAL_LOCAL_WHISPER_MODEL"] = "small"
        os.environ["VIRAL_LOCAL_MAX_DURATION"] = "900"
        
        try:
            manager = ConfigManager()
            config = manager.load_config()
            
            assert config.gemini_api_key == "env_test_key"
            assert config.whisper_model_size == "small"
            assert config.max_video_duration == 900
            
        finally:
            # Clean up environment variables
            for key in ["VIRAL_LOCAL_GEMINI_API_KEY", "VIRAL_LOCAL_WHISPER_MODEL", "VIRAL_LOCAL_MAX_DURATION"]:
                os.environ.pop(key, None)
    
    def test_validate_api_keys(self):
        """Test API key validation."""
        config = SystemConfig(
            gemini_api_key="valid_key_123456789",
            groq_api_key="another_valid_key_123"
        )
        
        manager = ConfigManager()
        manager.config = config
        
        validation = manager.validate_api_keys()
        assert validation["gemini"] is True
        assert validation["groq"] is True
    
    def test_validate_invalid_api_keys(self):
        """Test validation with invalid API keys."""
        config = SystemConfig(
            gemini_api_key="short",  # Too short
            groq_api_key=""  # Empty
        )
        
        manager = ConfigManager()
        manager.config = config
        
        validation = manager.validate_api_keys()
        assert validation["gemini"] is False
        assert validation["groq"] is False


if __name__ == "__main__":
    pytest.main([__file__])