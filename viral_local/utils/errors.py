"""
Custom exception classes for the Viral-Local system.

This module defines a hierarchy of custom exceptions that provide specific
error handling for different types of failures in the video localization pipeline.
"""

from typing import Optional, Dict, Any


class ViralLocalError(Exception):
    """Base exception class for all Viral-Local errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize base error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            return f"{self.message} (Code: {self.error_code}, Details: {self.details})"
        return f"{self.message} (Code: {self.error_code})"


class ConfigurationError(ViralLocalError):
    """Raised when there are configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: The configuration key that caused the error
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class ProcessingError(ViralLocalError):
    """Raised when there are errors during video processing operations."""
    
    def __init__(
        self, 
        message: str, 
        stage: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize processing error.
        
        Args:
            message: Error message
            stage: The processing stage where the error occurred
            input_data: Input data that caused the error
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if stage:
            details['stage'] = stage
        if input_data:
            details['input_data'] = input_data
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class APIError(ViralLocalError):
    """Raised when there are errors with external API calls."""
    
    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize API error.
        
        Args:
            message: Error message
            api_name: Name of the API that failed
            status_code: HTTP status code (if applicable)
            response_data: Response data from the API
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if api_name:
            details['api_name'] = api_name
        if status_code:
            details['status_code'] = status_code
        if response_data:
            details['response_data'] = response_data
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class ValidationError(ViralLocalError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = str(field_value)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DownloadError(ProcessingError):
    """Raised when video download operations fail."""
    
    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        """Initialize download error.
        
        Args:
            message: Error message
            url: URL that failed to download
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if url:
            details['url'] = url
        kwargs['details'] = details
        super().__init__(message, stage="download", **kwargs)


class TranscriptionError(ProcessingError):
    """Raised when transcription operations fail."""
    
    def __init__(
        self,
        message: str,
        audio_file: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize transcription error.
        
        Args:
            message: Error message
            audio_file: Path to audio file that failed transcription
            model_name: Name of the transcription model used
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if audio_file:
            details['audio_file'] = audio_file
        if model_name:
            details['model_name'] = model_name
        kwargs['details'] = details
        super().__init__(message, stage="transcription", **kwargs)


class TranslationError(ProcessingError):
    """Raised when translation operations fail."""
    
    def __init__(
        self,
        message: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        **kwargs
    ):
        """Initialize translation error.
        
        Args:
            message: Error message
            source_language: Source language code
            target_language: Target language code
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if source_language:
            details['source_language'] = source_language
        if target_language:
            details['target_language'] = target_language
        kwargs['details'] = details
        super().__init__(message, stage="translation", **kwargs)


class AudioGenerationError(ProcessingError):
    """Raised when text-to-speech audio generation fails."""
    
    def __init__(
        self,
        message: str,
        tts_engine: Optional[str] = None,
        voice_config: Optional[str] = None,
        **kwargs
    ):
        """Initialize audio generation error.
        
        Args:
            message: Error message
            tts_engine: Name of the TTS engine used
            voice_config: Voice configuration that failed
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if tts_engine:
            details['tts_engine'] = tts_engine
        if voice_config:
            details['voice_config'] = voice_config
        kwargs['details'] = details
        super().__init__(message, stage="audio_generation", **kwargs)


class VideoAssemblyError(ProcessingError):
    """Raised when video assembly operations fail."""
    
    def __init__(
        self,
        message: str,
        video_file: Optional[str] = None,
        audio_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize video assembly error.
        
        Args:
            message: Error message
            video_file: Path to video file
            audio_file: Path to audio file
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.get('details', {})
        if video_file:
            details['video_file'] = video_file
        if audio_file:
            details['audio_file'] = audio_file
        kwargs['details'] = details
        super().__init__(message, stage="video_assembly", **kwargs)


def handle_error(error: Exception, logger=None) -> Dict[str, Any]:
    """Handle and format errors for consistent error reporting.
    
    Args:
        error: Exception to handle
        logger: Optional logger instance
        
    Returns:
        Dictionary containing formatted error information
    """
    if isinstance(error, ViralLocalError):
        error_info = error.to_dict()
    else:
        error_info = {
            "error_type": error.__class__.__name__,
            "message": str(error),
            "error_code": "UNKNOWN_ERROR",
            "details": {}
        }
    
    if logger:
        logger.error(f"Error occurred: {error_info}")
    
    return error_info


def create_user_friendly_message(error: ViralLocalError) -> str:
    """Create a user-friendly error message with actionable suggestions.
    
    Args:
        error: ViralLocalError instance
        
    Returns:
        User-friendly error message
    """
    base_message = error.message
    
    # Add specific suggestions based on error type
    if isinstance(error, ConfigurationError):
        return f"{base_message}\n\nSuggestion: Check your configuration file and ensure all required settings are provided."
    
    elif isinstance(error, DownloadError):
        return f"{base_message}\n\nSuggestion: Verify the YouTube URL is valid and accessible. Check your internet connection."
    
    elif isinstance(error, TranscriptionError):
        return f"{base_message}\n\nSuggestion: Ensure the audio quality is good and the file is not corrupted. Try a different Whisper model size."
    
    elif isinstance(error, TranslationError):
        return f"{base_message}\n\nSuggestion: Check your API keys and ensure the target language is supported."
    
    elif isinstance(error, APIError):
        return f"{base_message}\n\nSuggestion: Verify your API keys are valid and you haven't exceeded rate limits. Check your internet connection."
    
    else:
        return f"{base_message}\n\nSuggestion: Check the logs for more detailed error information."