"""
System validation tests for the Viral-Local application.

This module contains comprehensive tests to validate the entire system
meets performance requirements, supports all languages, and handles
error scenarios correctly.
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from viral_local.main import ViralLocalPipeline, ProgressTracker
from viral_local.config import SystemConfig, ConfigManager
from viral_local.models import VideoFile, AudioFile, ProcessingResult
from viral_local.utils import ViralLocalError


class TestSystemValidation:
    """Comprehensive system validation tests."""
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid system configuration for testing."""
        return SystemConfig(
            gemini_api_key="test_api_key_for_validation_12345",
            groq_api_key="test_groq_key_12345",
            whisper_model_size="base",
            tts_engine="edge-tts",
            supported_languages=["hi", "bn", "ta"],
            default_target_language="hi",
            temp_dir="temp",
            output_dir="output",
            cache_dir="cache",
            enable_gpu=False,
            cache_enabled=True
        )
    
    @pytest.fixture
    def pipeline(self, valid_config):
        """Create a pipeline instance for testing."""
        with patch('viral_local.main.ConfigManager') as mock_config_manager:
            mock_config_manager.return_value.load_config.return_value = valid_config
            
            # Mock all service dependencies
            with patch('viral_local.main.DownloaderService'), \
                 patch('viral_local.main.TranscriberService'), \
                 patch('viral_local.main.LocalizationEngine'), \
                 patch('viral_local.main.DubbingStudio'):
                
                pipeline = ViralLocalPipeline()
                return pipeline
    
    def test_system_setup_validation(self, valid_config):
        """Test that system setup validation works correctly."""
        with patch('viral_local.main.ConfigManager') as mock_config_manager:
            mock_config_manager.return_value.load_config.return_value = valid_config
            mock_config_manager.return_value.validate_api_keys.return_value = {
                "gemini": True,
                "groq": True
            }
            
            with patch('viral_local.main.DownloaderService'), \
                 patch('viral_local.main.TranscriberService'), \
                 patch('viral_local.main.LocalizationEngine'), \
                 patch('viral_local.main.DubbingStudio'):
                
                pipeline = ViralLocalPipeline()
                
                # Mock directory existence
                with patch('pathlib.Path.exists', return_value=True):
                    results = pipeline.validate_setup()
                
                # Verify all validations pass
                assert results["gemini"] is True
                assert results["groq"] is True
                assert results["directories"] is True
                assert results["languages"] is True
    
    def test_supported_languages_validation(self, pipeline):
        """Test that all supported languages are properly configured."""
        config = pipeline.config
        
        # Verify all required languages are supported
        required_languages = ["hi", "bn", "ta"]
        for lang in required_languages:
            assert lang in config.supported_languages, f"Language {lang} not supported"
        
        # Verify default language is in supported languages
        assert config.default_target_language in config.supported_languages
    
    def test_error_handling_scenarios(self, pipeline):
        """Test various error handling scenarios."""
        
        # Test invalid YouTube URL - the pipeline should return a failed ProcessingResult
        # Mock the downloader to raise an error
        pipeline.downloader.download_video = Mock(side_effect=ViralLocalError("Invalid URL", "INVALID_URL"))
        result = pipeline.process_video("invalid_url", "hi")
        
        # Verify the result indicates failure
        assert isinstance(result, ProcessingResult)
        assert result.success is False
        assert result.error_code == "INVALID_URL"
    
    def test_configuration_validation(self):
        """Test configuration validation with various scenarios."""
        
        # Test valid configuration
        valid_config = SystemConfig(
            gemini_api_key="valid_key_12345",
            whisper_model_size="base",
            tts_engine="edge-tts",
            supported_languages=["hi", "bn", "ta"]
        )
        assert valid_config.gemini_api_key == "valid_key_12345"
        
        # Test invalid configurations
        with pytest.raises(ValueError, match="Gemini API key is required"):
            SystemConfig(gemini_api_key="")
        
        with pytest.raises(ValueError, match="Invalid Whisper model size"):
            SystemConfig(
                gemini_api_key="valid_key",
                whisper_model_size="invalid_size"
            )
        
        with pytest.raises(ValueError, match="Invalid TTS engine"):
            SystemConfig(
                gemini_api_key="valid_key",
                tts_engine="invalid_engine"
            )
    
    def test_progress_tracking_functionality(self):
        """Test progress tracking works correctly."""
        tracker = ProgressTracker()
        
        # Test initialization
        assert tracker.start_time is None
        assert tracker.current_stage is None
        
        # Test stage management
        tracker.start_processing("https://youtube.com/watch?v=test", "hi")
        assert tracker.start_time is not None
        
        tracker.start_stage("download", "Downloading video")
        assert tracker.current_stage == "download"
        assert "download" in tracker.stage_times
        
        tracker.update_progress("Processing...")
        tracker.complete_stage("Download completed")
        
        # Should not raise any exceptions
        tracker.show_completion("output/test.mp4")
    
    def test_api_key_validation(self):
        """Test API key validation functionality."""
        config_manager = ConfigManager()
        
        # Test with valid keys
        config = SystemConfig(
            gemini_api_key="valid_gemini_key_123456789",
            groq_api_key="valid_groq_key_123456789"
        )
        config_manager.config = config
        
        validation = config_manager.validate_api_keys()
        assert validation["gemini"] is True
        assert validation["groq"] is True
        
        # Test with invalid keys
        config = SystemConfig(
            gemini_api_key="short",  # Too short
            groq_api_key=""  # Empty
        )
        config_manager.config = config
        
        validation = config_manager.validate_api_keys()
        assert validation["gemini"] is False
        assert validation["groq"] is False
    
    def test_directory_creation(self, valid_config):
        """Test that required directories are created."""
        # The config should create directories during initialization
        assert Path(valid_config.temp_dir).exists()
        assert Path(valid_config.output_dir).exists()
        assert Path(valid_config.cache_dir).exists()
    
    def test_processing_result_model(self):
        """Test ProcessingResult model functionality."""
        # Test successful result
        success_result = ProcessingResult(
            success=True,
            data={"output_file": "test.mp4"},
            processing_time=120.5
        )
        
        assert success_result.success is True
        assert success_result.data["output_file"] == "test.mp4"
        assert success_result.processing_time == 120.5
        
        # Test error result
        error_result = ProcessingResult(
            success=False,
            error_message="Processing failed",
            error_code="PROCESSING_ERROR"
        )
        
        assert error_result.success is False
        assert error_result.error_message == "Processing failed"
        assert error_result.error_code == "PROCESSING_ERROR"
        
        # Test serialization
        result_dict = success_result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        
        # Test deserialization
        restored_result = ProcessingResult.from_dict(result_dict)
        assert restored_result.success == success_result.success
        assert restored_result.data == success_result.data
    
    @patch('viral_local.main.DownloaderService')
    @patch('viral_local.main.TranscriberService')
    @patch('viral_local.main.LocalizationEngine')
    @patch('viral_local.main.DubbingStudio')
    def test_end_to_end_pipeline_mock(self, mock_dubbing, mock_localization, 
                                     mock_transcriber, mock_downloader, valid_config):
        """Test end-to-end pipeline with mocked services."""
        
        # Setup mocks
        mock_video = VideoFile(
            file_path="test.mp4",
            duration=300.0,  # 5 minutes
            resolution=(1920, 1080),
            format="mp4"
        )
        
        mock_audio = AudioFile(
            file_path="test.wav",
            duration=300.0,
            sample_rate=22050,
            channels=1
        )
        
        # Configure service mocks
        mock_downloader.return_value.download_video.return_value = mock_video
        mock_downloader.return_value.extract_audio.return_value = mock_audio
        
        mock_transcription = Mock()
        mock_transcription.segments = [Mock(start_time=0.0, end_time=5.0)]
        mock_transcriber.return_value.transcribe_audio.return_value = mock_transcription
        mock_transcriber.return_value.get_speaker_statistics.return_value = {
            "total_speakers": 1,
            "speaker_details": {
                "speaker_1": {"words_per_minute": 150}
            }
        }
        
        mock_localization.return_value.analyze_viral_segments.return_value = []
        mock_translated_segment = Mock()
        mock_translated_segment.original_segment = Mock(start_time=0.0, end_time=5.0)
        mock_localization.return_value.translate_content.return_value = [mock_translated_segment]
        
        mock_dubbing.return_value.generate_speech.return_value = mock_audio
        mock_dubbing.return_value.synchronize_audio.return_value = mock_audio
        mock_dubbing.return_value.merge_audio_video.return_value = mock_video
        
        # Create pipeline
        with patch('viral_local.main.ConfigManager') as mock_config_manager:
            mock_config_manager.return_value.load_config.return_value = valid_config
            
            pipeline = ViralLocalPipeline()
            
            # Test processing
            result = pipeline.process_video("https://youtube.com/watch?v=test", "hi")
            
            # Verify result
            assert isinstance(result, ProcessingResult)
            assert result.success is True
            assert result.data == mock_video
    
    def test_performance_requirements_validation(self):
        """Test that performance requirements are properly configured."""
        config = SystemConfig(
            gemini_api_key="test_key_12345",
            max_video_duration=1800,  # 30 minutes
            max_concurrent_requests=3,
            cache_enabled=True,
            enable_gpu=True
        )
        
        # Verify performance settings
        assert config.max_video_duration == 1800  # 30 minutes
        assert config.max_concurrent_requests >= 1
        assert config.cache_enabled is True
        
        # Verify processing limits are reasonable
        assert config.max_video_duration > 0
        assert config.max_concurrent_requests > 0
        assert config.max_file_size_mb > 0
    
    def test_language_specific_configurations(self, valid_config):
        """Test language-specific configuration handling."""
        
        # Test each supported language
        for language in valid_config.supported_languages:
            assert language in ["hi", "bn", "ta"], f"Unsupported language: {language}"
        
        # Test default language
        assert valid_config.default_target_language in valid_config.supported_languages
        
        # Test language validation
        assert len(valid_config.supported_languages) > 0


class TestSystemIntegration:
    """Integration tests for system components."""
    
    def test_config_manager_integration(self):
        """Test ConfigManager integration with environment variables."""
        
        # Set test environment variables
        test_env = {
            "VIRAL_LOCAL_GEMINI_API_KEY": "env_test_key_12345",
            "VIRAL_LOCAL_WHISPER_MODEL": "small",
            "VIRAL_LOCAL_MAX_DURATION": "900",
            "VIRAL_LOCAL_ENABLE_GPU": "true"
        }
        
        with patch.dict(os.environ, test_env):
            manager = ConfigManager()
            config = manager.load_config()
            
            assert config.gemini_api_key == "env_test_key_12345"
            assert config.whisper_model_size == "small"
            assert config.max_video_duration == 900
            assert config.enable_gpu is True
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery and retry mechanisms."""
        config = SystemConfig(
            gemini_api_key="test_key_12345",
            max_retries=3,
            retry_delay=1.0,
            exponential_backoff=True,
            enable_error_recovery=True
        )
        
        # Verify error recovery settings
        assert config.max_retries > 0
        assert config.retry_delay > 0
        assert config.exponential_backoff is True
        assert config.enable_error_recovery is True
    
    def test_cache_and_performance_settings(self):
        """Test cache and performance configuration."""
        config = SystemConfig(
            gemini_api_key="test_key_12345",
            cache_enabled=True,
            cache_max_size_mb=1000,
            max_memory_mb=2000,
            batch_size=16
        )
        
        # Verify performance settings
        assert config.cache_enabled is True
        assert config.cache_max_size_mb > 0
        assert config.max_memory_mb > 0
        assert config.batch_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])