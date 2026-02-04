"""
Tests for TranscriberService.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from viral_local.services.transcriber import TranscriberService
from viral_local.models import AudioFile, Transcription, TranscriptSegment
from viral_local.config import SystemConfig
from viral_local.utils import TranscriptionError


class TestTranscriberService:
    """Test TranscriberService functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SystemConfig(
            gemini_api_key="test_api_key_for_testing_12345",
            whisper_model_size="base",
            enable_gpu=False,
            cache_enabled=True,
            temp_dir="temp",
            cache_dir="cache"
        )
    
    @pytest.fixture
    def audio_file(self):
        """Create test audio file."""
        return AudioFile(
            file_path="test_audio.wav",
            duration=10.0,
            sample_rate=16000,
            channels=1,
            format="wav"
        )
    
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    @patch('viral_local.services.transcriber.whisper')
    def test_device_initialization_cpu(self, mock_whisper, config):
        """Test device initialization defaults to CPU."""
        mock_whisper.load_model.return_value = Mock()
        
        with patch('viral_local.services.transcriber.TORCH_AVAILABLE', False):
            transcriber = TranscriberService(config)
            assert transcriber.device == "cpu"
    
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', False)
    def test_whisper_not_available(self, config):
        """Test error when Whisper is not available."""
        with pytest.raises(TranscriptionError, match="Whisper not available"):
            TranscriberService(config)
    
    @patch('viral_local.services.transcriber.whisper')
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    def test_model_loading_success(self, mock_whisper, config):
        """Test successful model loading."""
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        transcriber = TranscriberService(config)
        
        assert transcriber.model == mock_model
        mock_whisper.load_model.assert_called_once()
    
    @patch('viral_local.services.transcriber.whisper')
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    def test_model_loading_fallback(self, mock_whisper, config):
        """Test model loading fallback to base model."""
        config.whisper_model_size = "large"
        
        # First call fails, second succeeds
        mock_whisper.load_model.side_effect = [Exception("Model not found"), Mock()]
        
        transcriber = TranscriberService(config)
        
        assert transcriber.config.whisper_model_size == "base"
        assert mock_whisper.load_model.call_count == 2
    
    @patch('viral_local.services.transcriber.whisper')
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    @patch('os.path.exists')
    def test_transcribe_audio_success(self, mock_exists, mock_whisper, config, audio_file):
        """Test successful audio transcription."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        # Mock transcription result
        mock_result = {
            "language": "en",
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 2.0,
                    "avg_logprob": -0.5,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.9},
                        {"word": "world", "start": 1.0, "end": 2.0, "probability": 0.8}
                    ]
                }
            ]
        }
        mock_model.transcribe.return_value = mock_result
        
        transcriber = TranscriberService(config)
        result = transcriber.transcribe_audio(audio_file)
        
        assert isinstance(result, Transcription)
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 2.0
        assert result.segments[0].confidence > 0.0
    
    @patch('viral_local.services.transcriber.whisper')
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    def test_transcribe_audio_file_not_found(self, mock_whisper, config, audio_file):
        """Test transcription with missing audio file."""
        mock_whisper.load_model.return_value = Mock()
        
        transcriber = TranscriberService(config)
        
        with pytest.raises(TranscriptionError, match="Audio file not found"):
            transcriber.transcribe_audio(audio_file)
    
    @patch('viral_local.services.transcriber.whisper')
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    @patch('os.path.exists')
    def test_detect_language_success(self, mock_exists, mock_whisper, config, audio_file):
        """Test successful language detection."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        mock_whisper.load_audio.return_value = Mock()
        mock_whisper.pad_or_trim.return_value = Mock()
        mock_whisper.log_mel_spectrogram.return_value = Mock(to=Mock(return_value=Mock()))
        
        # Mock language detection
        mock_model.detect_language.return_value = (None, {"en": 0.9, "hi": 0.1})
        
        transcriber = TranscriberService(config)
        language = transcriber.detect_language(audio_file)
        
        assert language == "en"
    
    @patch('viral_local.services.transcriber.whisper')
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    def test_segment_transcription(self, mock_whisper, config):
        """Test transcription segmentation."""
        mock_whisper.load_model.return_value = Mock()
        
        # Create test transcription with multiple segments
        segments = [
            TranscriptSegment(
                text="First segment",
                start_time=0.0,
                end_time=5.0,
                confidence=0.9,
                language="en"
            ),
            TranscriptSegment(
                text="Second segment",
                start_time=5.0,
                end_time=10.0,
                confidence=0.8,
                language="en"
            ),
            TranscriptSegment(
                text="Third segment",
                start_time=10.0,
                end_time=15.0,
                confidence=0.85,
                language="en"
            )
        ]
        
        transcription = Transcription(
            segments=segments,
            language="en",
            total_duration=15.0,
            confidence_avg=0.85
        )
        
        transcriber = TranscriberService(config)
        result = transcriber.segment_transcription(transcription)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(seg, TranscriptSegment) for seg in result)
    
    @patch('viral_local.services.transcriber.whisper')
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    def test_get_model_info(self, mock_whisper, config):
        """Test getting model information."""
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(numel=Mock(return_value=1000))]
        mock_whisper.load_model.return_value = mock_model
        
        transcriber = TranscriberService(config)
        info = transcriber.get_model_info()
        
        assert info["status"] == "loaded"
        assert info["model_size"] == "base"
        assert info["device"] == "cpu"
        assert "total_parameters" in info
    
    @patch('viral_local.services.transcriber.whisper')
    @patch('viral_local.services.transcriber.WHISPER_AVAILABLE', True)
    def test_simple_speaker_detection(self, mock_whisper, config, audio_file):
        """Test simple speaker detection fallback."""
        mock_whisper.load_model.return_value = Mock()
        
        transcriber = TranscriberService(config)
        result = transcriber._simple_speaker_detection(audio_file)
        
        assert len(result) == 1
        assert result[0] == (0.0, audio_file.duration, "speaker_1")


if __name__ == "__main__":
    pytest.main([__file__])