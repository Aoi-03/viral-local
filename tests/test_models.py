"""
Tests for core data models.
"""

import pytest
from viral_local.models import (
    VideoFile, AudioFile, TranscriptSegment, ViralSegment, 
    TranslatedSegment, VoiceConfig
)


class TestVideoFile:
    """Test VideoFile model."""
    
    def test_valid_video_file(self):
        """Test creating a valid VideoFile."""
        video = VideoFile(
            file_path="test.mp4",
            duration=120.5,
            resolution=(1920, 1080),
            format="mp4",
            metadata={"title": "Test Video"}
        )
        
        assert video.file_path == "test.mp4"
        assert video.duration == 120.5
        assert video.resolution == (1920, 1080)
        assert video.format == "mp4"
        assert video.metadata["title"] == "Test Video"
    
    def test_invalid_duration(self):
        """Test VideoFile with invalid duration."""
        with pytest.raises(ValueError, match="Video duration must be positive"):
            VideoFile(
                file_path="test.mp4",
                duration=-10,
                resolution=(1920, 1080),
                format="mp4"
            )
    
    def test_invalid_resolution(self):
        """Test VideoFile with invalid resolution."""
        with pytest.raises(ValueError, match="Resolution must be a tuple of two positive integers"):
            VideoFile(
                file_path="test.mp4",
                duration=120,
                resolution=(1920, -1080),
                format="mp4"
            )
    
    def test_empty_file_path(self):
        """Test VideoFile with empty file path."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            VideoFile(
                file_path="",
                duration=120,
                resolution=(1920, 1080),
                format="mp4"
            )


class TestTranscriptSegment:
    """Test TranscriptSegment model."""
    
    def test_valid_segment(self):
        """Test creating a valid TranscriptSegment."""
        segment = TranscriptSegment(
            text="Hello world",
            start_time=0.0,
            end_time=2.5,
            confidence=0.95,
            speaker_id="speaker_1"
        )
        
        assert segment.text == "Hello world"
        assert segment.start_time == 0.0
        assert segment.end_time == 2.5
        assert segment.confidence == 0.95
        assert segment.speaker_id == "speaker_1"
        assert segment.duration == 2.5
    
    def test_empty_text(self):
        """Test TranscriptSegment with empty text."""
        with pytest.raises(ValueError, match="Transcript text cannot be empty"):
            TranscriptSegment(
                text="   ",
                start_time=0.0,
                end_time=2.5,
                confidence=0.95
            )
    
    def test_invalid_timing(self):
        """Test TranscriptSegment with invalid timing."""
        with pytest.raises(ValueError, match="End time must be greater than start time"):
            TranscriptSegment(
                text="Hello",
                start_time=2.5,
                end_time=2.0,
                confidence=0.95
            )
    
    def test_invalid_confidence(self):
        """Test TranscriptSegment with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            TranscriptSegment(
                text="Hello",
                start_time=0.0,
                end_time=2.5,
                confidence=1.5
            )


class TestVoiceConfig:
    """Test VoiceConfig model."""
    
    def test_valid_voice_config(self):
        """Test creating a valid VoiceConfig."""
        config = VoiceConfig(
            language="hi",
            gender="female",
            age_range="adult",
            speaking_rate=1.2,
            pitch_adjustment=0.1
        )
        
        assert config.language == "hi"
        assert config.gender == "female"
        assert config.age_range == "adult"
        assert config.speaking_rate == 1.2
        assert config.pitch_adjustment == 0.1
    
    def test_invalid_gender(self):
        """Test VoiceConfig with invalid gender."""
        with pytest.raises(ValueError, match="Gender must be 'male', 'female', or 'neutral'"):
            VoiceConfig(
                language="hi",
                gender="unknown",
                age_range="adult"
            )
    
    def test_invalid_age_range(self):
        """Test VoiceConfig with invalid age range."""
        with pytest.raises(ValueError, match="Age range must be 'child', 'young', 'adult', or 'elderly'"):
            VoiceConfig(
                language="hi",
                gender="female",
                age_range="middle-aged"
            )
    
    def test_invalid_speaking_rate(self):
        """Test VoiceConfig with invalid speaking rate."""
        with pytest.raises(ValueError, match="Speaking rate must be between 0.5 and 2.0"):
            VoiceConfig(
                language="hi",
                gender="female",
                age_range="adult",
                speaking_rate=3.0
            )


if __name__ == "__main__":
    pytest.main([__file__])