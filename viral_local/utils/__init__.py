"""
Utility modules for the Viral-Local system.

This package contains logging, error handling, caching, retry mechanisms,
and other utility functions used throughout the video localization pipeline.
"""

from .logging import setup_logging, get_logger
from .errors import (
    ViralLocalError, ConfigurationError, ProcessingError, APIError,
    ValidationError, DownloadError, TranscriptionError, TranslationError,
    AudioGenerationError, VideoAssemblyError, handle_error, create_user_friendly_message
)
from .cache import (
    CacheManager, ResourceManager, cached, get_cache_manager, get_resource_manager
)
from .retry import (
    RetryManager, RetryConfig, RetryStrategy, RateLimiter, CircuitBreaker,
    retry, rate_limited, circuit_breaker, get_rate_limiter
)
from .error_recovery import (
    ErrorReport, ErrorLogger, RecoveryManager, get_error_logger, get_recovery_manager
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger", 
    
    # Error handling
    "ViralLocalError",
    "ConfigurationError",
    "ProcessingError",
    "APIError",
    "ValidationError",
    "DownloadError",
    "TranscriptionError", 
    "TranslationError",
    "AudioGenerationError",
    "VideoAssemblyError",
    "handle_error",
    "create_user_friendly_message",
    
    # Caching and resource management
    "CacheManager",
    "ResourceManager",
    "cached",
    "get_cache_manager",
    "get_resource_manager",
    
    # Retry and resilience
    "RetryManager",
    "RetryConfig", 
    "RetryStrategy",
    "RateLimiter",
    "CircuitBreaker",
    "retry",
    "rate_limited",
    "circuit_breaker",
    "get_rate_limiter",
    
    # Error recovery and logging
    "ErrorReport",
    "ErrorLogger",
    "RecoveryManager",
    "get_error_logger",
    "get_recovery_manager"
]