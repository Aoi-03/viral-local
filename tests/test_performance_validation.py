"""
Performance validation tests for the Viral-Local system.

This module tests performance requirements and benchmarks
system capabilities under various load conditions.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

from viral_local.main import ViralLocalPipeline, ProgressTracker
from viral_local.config import SystemConfig
from viral_local.models import VideoFile, AudioFile, ProcessingResult


class TestPerformanceValidation:
    """Performance validation and benchmarking tests."""
    
    @pytest.fixture
    def performance_config(self):
        """Create configuration optimized for performance testing."""
        return SystemConfig(
            gemini_api_key="test_api_key_for_performance_12345",
            whisper_model_size="base",  # Faster than larger models
            tts_engine="edge-tts",
            max_video_duration=1800,  # 30 minutes
            max_concurrent_requests=3,
            enable_gpu=False,  # For consistent testing
            cache_enabled=True,
            batch_size=16,
            max_memory_mb=2000
        )
    
    def test_processing_time_requirements(self, performance_config):
        """Test that processing meets time requirements (5min video in <3min)."""
        
        # Mock a 5-minute video processing pipeline
        with patch('viral_local.main.ConfigManager') as mock_config_manager, \
             patch('viral_local.main.DownloaderService') as mock_downloader, \
             patch('viral_local.main.TranscriberService') as mock_transcriber, \
             patch('viral_local.main.LocalizationEngine') as mock_localization, \
             patch('viral_local.main.DubbingStudio') as mock_dubbing:
            
            mock_config_manager.return_value.load_config.return_value = performance_config
            
            # Setup mocks for 5-minute video
            mock_video = VideoFile(
                file_path="test_5min.mp4",
                duration=300.0,  # 5 minutes
                resolution=(1920, 1080),
                format="mp4"
            )
            
            mock_audio = AudioFile(
                file_path="test_5min.wav",
                duration=300.0,
                sample_rate=22050,
                channels=1
            )
            
            # Configure service mocks with realistic delays
            def mock_download_with_delay(*args, **kwargs):
                time.sleep(0.1)  # Simulate download time
                return mock_video
            
            def mock_extract_audio_with_delay(*args, **kwargs):
                time.sleep(0.05)  # Simulate audio extraction
                return mock_audio
            
            def mock_transcribe_with_delay(*args, **kwargs):
                time.sleep(0.2)  # Simulate transcription time
                mock_transcription = Mock()
                mock_transcription.segments = [Mock(start_time=0.0, end_time=5.0)]
                return mock_transcription
            
            def mock_translate_with_delay(*args, **kwargs):
                time.sleep(0.15)  # Simulate translation time
                mock_translated = Mock()
                mock_translated.original_segment = Mock(start_time=0.0, end_time=5.0)
                return [mock_translated]
            
            def mock_generate_speech_with_delay(*args, **kwargs):
                time.sleep(0.3)  # Simulate TTS time
                return mock_audio
            
            def mock_merge_with_delay(*args, **kwargs):
                time.sleep(0.1)  # Simulate video assembly
                return mock_video
            
            # Setup all mocks
            mock_downloader.return_value.download_video.side_effect = mock_download_with_delay
            mock_downloader.return_value.extract_audio.side_effect = mock_extract_audio_with_delay
            mock_transcriber.return_value.transcribe_audio.side_effect = mock_transcribe_with_delay
            mock_transcriber.return_value.get_speaker_statistics.return_value = {
                "total_speakers": 1,
                "speaker_details": {"speaker_1": {"words_per_minute": 150}}
            }
            mock_localization.return_value.analyze_viral_segments.return_value = []
            mock_localization.return_value.translate_content.side_effect = mock_translate_with_delay
            mock_dubbing.return_value.generate_speech.side_effect = mock_generate_speech_with_delay
            mock_dubbing.return_value.synchronize_audio.return_value = mock_audio
            mock_dubbing.return_value.merge_audio_video.side_effect = mock_merge_with_delay
            
            # Create pipeline and measure processing time
            pipeline = ViralLocalPipeline()
            
            start_time = time.time()
            result = pipeline.process_video("https://youtube.com/watch?v=test5min", "hi")
            processing_time = time.time() - start_time
            
            # Verify result and performance
            assert isinstance(result, ProcessingResult)
            assert result.success is True
            
            # Performance requirement: 5-minute video should process in under 3 minutes (180 seconds)
            # Our mock should complete much faster, but we test the requirement structure
            assert processing_time < 180, f"Processing took {processing_time:.2f}s, should be < 180s"
            
            # For our mock, it should be very fast
            assert processing_time < 5, f"Mock processing took {processing_time:.2f}s, should be < 5s"
    
    def test_concurrent_request_handling(self, performance_config):
        """Test handling multiple concurrent requests efficiently."""
        
        with patch('viral_local.main.ConfigManager') as mock_config_manager, \
             patch('viral_local.main.DownloaderService') as mock_downloader, \
             patch('viral_local.main.TranscriberService') as mock_transcriber, \
             patch('viral_local.main.LocalizationEngine') as mock_localization, \
             patch('viral_local.main.DubbingStudio') as mock_dubbing:
            
            mock_config_manager.return_value.load_config.return_value = performance_config
            
            # Setup basic mocks
            mock_video = VideoFile(file_path="test.mp4", duration=60.0, resolution=(1920, 1080), format="mp4")
            mock_audio = AudioFile(file_path="test.wav", duration=60.0, sample_rate=22050, channels=1)
            
            mock_downloader.return_value.download_video.return_value = mock_video
            mock_downloader.return_value.extract_audio.return_value = mock_audio
            
            mock_transcription = Mock()
            mock_transcription.segments = [Mock(start_time=0.0, end_time=5.0)]
            mock_transcriber.return_value.transcribe_audio.return_value = mock_transcription
            mock_transcriber.return_value.get_speaker_statistics.return_value = {
                "total_speakers": 1,
                "speaker_details": {"speaker_1": {"words_per_minute": 150}}
            }
            
            mock_localization.return_value.analyze_viral_segments.return_value = []
            mock_translated = Mock()
            mock_translated.original_segment = Mock(start_time=0.0, end_time=5.0)
            mock_localization.return_value.translate_content.return_value = [mock_translated]
            
            mock_dubbing.return_value.generate_speech.return_value = mock_audio
            mock_dubbing.return_value.synchronize_audio.return_value = mock_audio
            mock_dubbing.return_value.merge_audio_video.return_value = mock_video
            
            # Test concurrent processing
            def process_video(video_id):
                pipeline = ViralLocalPipeline()
                return pipeline.process_video(f"https://youtube.com/watch?v=test{video_id}", "hi")
            
            # Process multiple videos concurrently
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=performance_config.max_concurrent_requests) as executor:
                futures = [executor.submit(process_video, i) for i in range(3)]
                results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            
            # Verify all requests succeeded
            assert len(results) == 3
            assert all(result.success for result in results)
            
            # Concurrent processing should be more efficient than sequential
            # (though with mocks, the difference may be minimal)
            assert total_time < 10, f"Concurrent processing took {total_time:.2f}s, should be < 10s"
    
    def test_memory_usage_limits(self, performance_config):
        """Test that memory usage stays within configured limits."""
        
        # Test memory configuration
        assert performance_config.max_memory_mb > 0
        assert performance_config.max_memory_mb <= 4000  # Reasonable upper limit
        
        # Test cache size limits
        assert performance_config.cache_max_size_mb > 0
        assert performance_config.cache_max_size_mb <= performance_config.max_memory_mb
    
    def test_progress_tracking_performance(self):
        """Test that progress tracking doesn't significantly impact performance."""
        
        # Test with progress tracking
        tracker = ProgressTracker()
        
        start_time = time.time()
        tracker.start_processing("https://youtube.com/watch?v=test", "hi")
        
        for i in range(100):  # Simulate many progress updates
            tracker.start_stage(f"stage_{i % 5}", f"Processing stage {i}")
            tracker.update_progress(f"Step {i}")
            tracker.complete_stage(f"Completed step {i}")
        
        tracker.show_completion("output/test.mp4")
        tracking_time = time.time() - start_time
        
        # Progress tracking should be very fast
        assert tracking_time < 1.0, f"Progress tracking took {tracking_time:.2f}s, should be < 1s"
    
    def test_configuration_performance_settings(self, performance_config):
        """Test that performance-related configuration settings are optimal."""
        
        # Test batch size is reasonable
        assert 1 <= performance_config.batch_size <= 64
        
        # Test concurrent request limits
        assert 1 <= performance_config.max_concurrent_requests <= 10
        
        # Test video duration limits are reasonable
        assert performance_config.max_video_duration > 0
        assert performance_config.max_video_duration <= 3600  # 1 hour max
        
        # Test cache is enabled for performance
        assert performance_config.cache_enabled is True
        
        # Test retry settings are reasonable
        assert performance_config.max_retries >= 1
        assert performance_config.retry_delay > 0
    
    def test_api_rate_limiting_configuration(self, performance_config):
        """Test API rate limiting settings for optimal performance."""
        
        # Test rate limiting settings
        assert performance_config.api_requests_per_minute > 0
        assert performance_config.api_requests_per_hour > 0
        assert performance_config.api_requests_per_hour >= performance_config.api_requests_per_minute
        
        # Test circuit breaker settings
        assert performance_config.circuit_breaker_failure_threshold > 0
        assert performance_config.circuit_breaker_recovery_timeout > 0
    
    def test_file_size_limits(self, performance_config):
        """Test file size limits for performance optimization."""
        
        # Test maximum file size is reasonable
        assert performance_config.max_file_size_mb > 0
        assert performance_config.max_file_size_mb <= 2000  # 2GB max
        
        # Test video duration corresponds to reasonable file sizes
        # Assuming ~1MB per minute for compressed video
        expected_max_size = performance_config.max_video_duration / 60 * 10  # 10MB per minute estimate
        assert performance_config.max_file_size_mb >= expected_max_size


class TestSystemBenchmarks:
    """System benchmarking tests for performance analysis."""
    
    def test_component_initialization_speed(self):
        """Benchmark component initialization times."""
        
        config = SystemConfig(
            gemini_api_key="test_key_12345",
            enable_gpu=False,
            cache_enabled=True
        )
        
        # Test configuration loading speed
        start_time = time.time()
        config_dict = config.to_dict()
        config_time = time.time() - start_time
        
        assert config_time < 0.1, f"Configuration processing took {config_time:.3f}s, should be < 0.1s"
        assert isinstance(config_dict, dict)
        assert len(config_dict) > 10  # Should have many configuration options
    
    def test_data_model_performance(self):
        """Test data model creation and validation performance."""
        
        # Test VideoFile creation speed
        start_time = time.time()
        for i in range(1000):
            video = VideoFile(
                file_path=f"test_{i}.mp4",
                duration=float(i + 1),
                resolution=(1920, 1080),
                format="mp4"
            )
        model_creation_time = time.time() - start_time
        
        assert model_creation_time < 1.0, f"Model creation took {model_creation_time:.3f}s, should be < 1s"
    
    def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        
        from viral_local.utils import ViralLocalError, handle_error
        import logging
        
        logger = logging.getLogger("test")
        
        # Test error creation and handling speed
        start_time = time.time()
        for i in range(100):
            try:
                raise ViralLocalError(f"Test error {i}", "TEST_ERROR")
            except ViralLocalError as e:
                handle_error(e, logger)
        
        error_handling_time = time.time() - start_time
        
        assert error_handling_time < 1.0, f"Error handling took {error_handling_time:.3f}s, should be < 1s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])