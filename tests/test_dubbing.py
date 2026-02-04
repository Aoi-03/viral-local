"""
Tests for DubbingStudio service.
"""

import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from viral_local.services.dubbing import DubbingStudio
from viral_local.models import (
    TranslatedSegment, TranscriptSegment, VoiceConfig, 
    AudioFile, VideoFile, TimingData
)
from viral_local.config import SystemConfig
from viral_local.utils import AudioGenerationError, VideoAssemblyError


class TestDubbingStudio:
    """Test DubbingStudio functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SystemConfig(
            gemini_api_key="test_key",
            tts_engine="edge-tts",
            temp_dir="test_temp",
            output_dir="test_output",
            audio_sample_rate=22050
        )
    
    @pytest.fixture
    def dubbing_studio(self, config):
        """Create DubbingStudio instance."""
        return DubbingStudio(config)
    
    @pytest.fixture
    def voice_config(self):
        """Create test voice configuration."""
        return VoiceConfig(
            language="hi",
            gender="female",
            age_range="adult",
            speaking_rate=1.0,
            pitch_adjustment=0.0
        )
    
    @pytest.fixture
    def translated_segments(self):
        """Create test translated segments."""
        original_segment = TranscriptSegment(
            text="Hello world",
            start_time=0.0,
            end_time=2.0,
            confidence=0.95
        )
        
        return [
            TranslatedSegment(
                original_segment=original_segment,
                translated_text="नमस्ते दुनिया",
                target_language="hi",
                quality_score=0.9
            )
        ]
    
    def test_voice_selection_hindi_female_adult(self, dubbing_studio, voice_config):
        """Test voice selection for Hindi female adult."""
        voice_name = dubbing_studio._select_voice(voice_config)
        assert voice_name == "hi-IN-SwaraNeural"
    
    def test_voice_selection_with_explicit_voice(self, dubbing_studio):
        """Test voice selection with explicit voice name."""
        voice_config = VoiceConfig(
            language="hi",
            gender="female", 
            age_range="adult",
            voice_name="custom-voice"
        )
        
        voice_name = dubbing_studio._select_voice(voice_config)
        assert voice_name == "custom-voice"
    
    def test_voice_selection_unsupported_language(self, dubbing_studio):
        """Test voice selection with unsupported language."""
        voice_config = VoiceConfig(
            language="unsupported",
            gender="female",
            age_range="adult"
        )
        
        with pytest.raises(AudioGenerationError, match="Language 'unsupported' not supported"):
            dubbing_studio._select_voice(voice_config)
    
    def test_ssml_creation(self, dubbing_studio, voice_config):
        """Test SSML creation with voice configuration."""
        text = "Test text"
        voice_name = "hi-IN-SwaraNeural"
        
        ssml = dubbing_studio._create_ssml(text, voice_name, voice_config)
        
        assert "hi-IN-SwaraNeural" in ssml
        assert "Test text" in ssml
        assert "prosody" in ssml
        assert "rate" in ssml
        assert "pitch" in ssml
    
    def test_audio_normalization(self, dubbing_studio):
        """Test audio quality normalization."""
        import numpy as np
        
        # Create test audio data with varying amplitudes
        audio_data = np.array([0.1, 0.8, -0.5, 0.3, -0.9])
        sample_rate = 22050
        
        normalized = dubbing_studio._normalize_audio_quality(audio_data, sample_rate)
        
        # Check that audio is normalized
        assert np.max(np.abs(normalized)) <= 1.0
        assert len(normalized) == len(audio_data)
    
    def test_timing_adjustments_calculation(self, dubbing_studio):
        """Test calculation of timing adjustments."""
        original_segments = [
            TranscriptSegment(
                text="Short text",
                start_time=0.0,
                end_time=2.0,
                confidence=0.95
            )
        ]
        
        translated_segments = [
            TranslatedSegment(
                original_segment=original_segments[0],
                translated_text="Much longer translated text here",
                target_language="hi",
                quality_score=0.9
            )
        ]
        
        timing_data = TimingData(
            original_segments=original_segments,
            target_segments=translated_segments,
            sync_points=[(0.0, 0.0), (2.0, 2.0)]
        )
        
        adjustments = dubbing_studio._calculate_timing_adjustments(timing_data)
        
        assert len(adjustments) == 1
        start_time, end_time, stretch_factor = adjustments[0]
        assert start_time == 0.0
        assert end_time == 2.0
        assert 0.5 <= stretch_factor <= 2.0  # Within reasonable bounds
    
    def test_generate_speech_empty_segments(self, dubbing_studio, voice_config):
        """Test speech generation with empty segments."""
        with pytest.raises(AudioGenerationError, match="No translated segments provided"):
            dubbing_studio.generate_speech([], voice_config)
    
    @patch('viral_local.services.dubbing.VideoFileClip')
    @patch('viral_local.services.dubbing.AudioFileClip')
    @patch('viral_local.services.dubbing.Path.exists')
    def test_merge_audio_video_success(self, mock_exists, mock_audio_clip, mock_video_clip, dubbing_studio):
        """Test successful audio-video merging."""
        # Create test files
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file:
            video_path = video_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
            audio_path = audio_file.name
        
        try:
            # Mock Path.exists to return True for output file
            mock_exists.return_value = True
            
            # Mock video and audio clips
            mock_video_instance = Mock()
            mock_video_instance.duration = 10.0
            mock_video_instance.w = 1920
            mock_video_instance.h = 1080
            mock_video_instance.set_audio.return_value = mock_video_instance
            mock_video_instance.write_videofile = Mock()
            mock_video_instance.close = Mock()
            mock_video_clip.return_value = mock_video_instance
            
            mock_audio_instance = Mock()
            mock_audio_instance.duration = 10.0
            mock_audio_instance.close = Mock()
            mock_audio_clip.return_value = mock_audio_instance
            
            # Mock the final VideoFileClip call for verification
            mock_final_clip = Mock()
            mock_final_clip.duration = 10.0
            mock_final_clip.w = 1920
            mock_final_clip.h = 1080
            mock_final_clip.__enter__ = Mock(return_value=mock_final_clip)
            mock_final_clip.__exit__ = Mock(return_value=None)
            
            # Make VideoFileClip return different instances for different calls
            mock_video_clip.side_effect = [mock_video_instance, mock_final_clip]
            
            # Create test objects
            video = VideoFile(
                file_path=video_path,
                duration=10.0,
                resolution=(1920, 1080),
                format="mp4"
            )
            
            audio = AudioFile(
                file_path=audio_path,
                duration=10.0,
                sample_rate=22050,
                channels=1
            )
            
            # Test merging
            result = dubbing_studio.merge_audio_video(video, audio)
            
            assert isinstance(result, VideoFile)
            assert result.duration == 10.0
            assert result.resolution == (1920, 1080)
            assert result.format == dubbing_studio.config.video_output_format
            
        finally:
            # Clean up test files
            os.unlink(video_path)
            os.unlink(audio_path)
    
    def test_merge_audio_video_missing_files(self, dubbing_studio):
        """Test audio-video merging with missing files."""
        video = VideoFile(
            file_path="nonexistent_video.mp4",
            duration=10.0,
            resolution=(1920, 1080),
            format="mp4"
        )
        
        audio = AudioFile(
            file_path="nonexistent_audio.wav",
            duration=10.0,
            sample_rate=22050,
            channels=1
        )
        
        with pytest.raises(VideoAssemblyError, match="Video file not found"):
            dubbing_studio.merge_audio_video(video, audio)


if __name__ == "__main__":
    pytest.main([__file__])