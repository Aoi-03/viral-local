"""
Configuration management system for the Viral-Local application.

This module handles loading, validation, and management of system configuration
including API keys, model settings, processing limits, and quality parameters.
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging


@dataclass
class SystemConfig:
    """Main configuration class for the Viral-Local system."""
    
    # API Configuration
    gemini_api_key: str = ""
    groq_api_key: Optional[str] = None
    
    # Model Settings
    whisper_model_size: str = "base"
    tts_engine: str = "edge-tts"  # or "kokoro-82m"
    
    # Processing Limits
    max_video_duration: int = 1800  # 30 minutes in seconds
    max_concurrent_requests: int = 3
    max_file_size_mb: int = 500
    
    # Quality Settings
    target_audio_quality: str = "high"
    video_output_format: str = "mp4"
    audio_sample_rate: int = 22050
    
    # Language Support
    supported_languages: List[str] = field(default_factory=lambda: ["hi", "bn", "ta"])
    default_target_language: str = "hi"
    
    # Directory Settings
    temp_dir: str = "temp"
    output_dir: str = "output"
    cache_dir: str = "cache"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "viral_local.log"
    enable_file_logging: bool = True
    
    # Performance Settings
    enable_gpu: bool = True
    batch_size: int = 16
    cache_enabled: bool = True
    cache_max_size_mb: int = 1000
    cache_default_ttl: Optional[int] = None  # seconds, None for no expiration
    max_memory_mb: int = 2000
    
    # Retry Settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Rate Limiting Settings
    api_requests_per_minute: int = 60
    api_requests_per_hour: int = 1000
    
    # Circuit Breaker Settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    
    # Error Recovery Settings
    enable_error_recovery: bool = True
    error_log_max_files: int = 10
    error_log_max_size_mb: int = 50
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._ensure_directories()
    
    def _validate_config(self):
        """Validate configuration values."""
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required")
        
        if self.whisper_model_size not in ["tiny", "base", "small", "medium", "large"]:
            raise ValueError("Invalid Whisper model size")
        
        if self.tts_engine not in ["edge-tts", "kokoro-82m"]:
            raise ValueError("Invalid TTS engine")
        
        if self.max_video_duration <= 0:
            raise ValueError("Max video duration must be positive")
        
        if self.max_concurrent_requests <= 0:
            raise ValueError("Max concurrent requests must be positive")
        
        if self.target_audio_quality not in ["low", "medium", "high"]:
            raise ValueError("Invalid audio quality setting")
        
        if self.video_output_format not in ["mp4", "avi", "mov"]:
            raise ValueError("Invalid video output format")
        
        if not self.supported_languages:
            raise ValueError("At least one supported language must be specified")
        
        if self.default_target_language not in self.supported_languages:
            raise ValueError("Default target language must be in supported languages")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level")
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.temp_dir, self.output_dir, self.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "gemini_api_key": self.gemini_api_key,
            "groq_api_key": self.groq_api_key,
            "whisper_model_size": self.whisper_model_size,
            "tts_engine": self.tts_engine,
            "max_video_duration": self.max_video_duration,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_file_size_mb": self.max_file_size_mb,
            "target_audio_quality": self.target_audio_quality,
            "video_output_format": self.video_output_format,
            "audio_sample_rate": self.audio_sample_rate,
            "supported_languages": self.supported_languages,
            "default_target_language": self.default_target_language,
            "temp_dir": self.temp_dir,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_file_logging": self.enable_file_logging,
            "enable_gpu": self.enable_gpu,
            "batch_size": self.batch_size,
            "cache_enabled": self.cache_enabled,
            "cache_max_size_mb": self.cache_max_size_mb,
            "cache_default_ttl": self.cache_default_ttl,
            "max_memory_mb": self.max_memory_mb,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "exponential_backoff": self.exponential_backoff,
            "api_requests_per_minute": self.api_requests_per_minute,
            "api_requests_per_hour": self.api_requests_per_hour,
            "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
            "circuit_breaker_recovery_timeout": self.circuit_breaker_recovery_timeout,
            "enable_error_recovery": self.enable_error_recovery,
            "error_log_max_files": self.error_log_max_files,
            "error_log_max_size_mb": self.error_log_max_size_mb
        }


class ConfigManager:
    """Manages loading and saving of system configuration."""
    
    DEFAULT_CONFIG_PATHS = [
        "config.yaml",
        "config.yml", 
        "config.json",
        os.path.expanduser("~/.viral_local/config.yaml"),
        "/etc/viral_local/config.yaml"
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> SystemConfig:
        """Load configuration from file or environment variables.
        
        Returns:
            SystemConfig: Loaded configuration
            
        Raises:
            FileNotFoundError: If no configuration file is found
            ValueError: If configuration is invalid
        """
        config_data = {}
        
        # Try to load from file
        if self.config_path:
            config_data = self._load_from_file(self.config_path)
        else:
            # Try default paths
            for path in self.DEFAULT_CONFIG_PATHS:
                if os.path.exists(path):
                    config_data = self._load_from_file(path)
                    self.config_path = path
                    break
        
        # Override with environment variables
        config_data.update(self._load_from_env())
        
        # Create configuration object
        self.config = SystemConfig(**config_data)
        
        self.logger.info(f"Configuration loaded from {self.config_path or 'environment'}")
        return self.config
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from a file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Dict containing configuration data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                else:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load config from {file_path}: {e}")
            return {}
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Returns:
            Dict containing configuration data from environment
        """
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            "VIRAL_LOCAL_GEMINI_API_KEY": "gemini_api_key",
            "VIRAL_LOCAL_GROQ_API_KEY": "groq_api_key",
            "VIRAL_LOCAL_WHISPER_MODEL": "whisper_model_size",
            "VIRAL_LOCAL_TTS_ENGINE": "tts_engine",
            "VIRAL_LOCAL_MAX_DURATION": "max_video_duration",
            "VIRAL_LOCAL_MAX_REQUESTS": "max_concurrent_requests",
            "VIRAL_LOCAL_AUDIO_QUALITY": "target_audio_quality",
            "VIRAL_LOCAL_OUTPUT_FORMAT": "video_output_format",
            "VIRAL_LOCAL_TARGET_LANG": "default_target_language",
            "VIRAL_LOCAL_LOG_LEVEL": "log_level",
            "VIRAL_LOCAL_ENABLE_GPU": "enable_gpu",
            "VIRAL_LOCAL_CACHE_ENABLED": "cache_enabled"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ["max_video_duration", "max_concurrent_requests", "audio_sample_rate", "batch_size", "max_retries", "cache_max_size_mb", "max_memory_mb", "api_requests_per_minute", "api_requests_per_hour", "circuit_breaker_failure_threshold", "error_log_max_files", "error_log_max_size_mb", "cache_default_ttl"]:
                    env_config[config_key] = int(value) if value != "None" else None
                elif config_key in ["retry_delay", "circuit_breaker_recovery_timeout"]:
                    env_config[config_key] = float(value)
                elif config_key in ["enable_gpu", "cache_enabled", "enable_file_logging", "exponential_backoff", "enable_error_recovery"]:
                    env_config[config_key] = value.lower() in ["true", "1", "yes", "on"]
                elif config_key == "supported_languages":
                    env_config[config_key] = [lang.strip() for lang in value.split(",")]
                else:
                    env_config[config_key] = value
        
        return env_config
    
    def save_config(self, config: SystemConfig, file_path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            file_path: Optional path to save to (defaults to loaded path)
        """
        save_path = file_path or self.config_path or "config.yaml"
        
        try:
            config_data = config.to_dict()
            
            # Remove sensitive data from saved config
            config_data.pop("gemini_api_key", None)
            config_data.pop("groq_api_key", None)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.endswith('.json'):
                    json.dump(config_data, f, indent=2)
                else:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {save_path}: {e}")
            raise
    
    def get_config(self) -> SystemConfig:
        """Get current configuration, loading if necessary.
        
        Returns:
            SystemConfig: Current configuration
        """
        if self.config is None:
            self.load_config()
        return self.config
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present and valid.
        
        Returns:
            Dict mapping API names to validation status
        """
        config = self.get_config()
        validation_results = {}
        
        # Check Gemini API key
        validation_results["gemini"] = bool(config.gemini_api_key and len(config.gemini_api_key) > 10)
        
        # Check Groq API key (optional)
        validation_results["groq"] = bool(config.groq_api_key and len(config.groq_api_key) > 10)
        
        return validation_results