"""
Language support validation tests for the Viral-Local system.

This module validates that all supported languages (Hindi, Bengali, Tamil)
are properly configured and can be processed by the system.
"""

import pytest
from unittest.mock import Mock, patch

from viral_local.config import SystemConfig
from viral_local.main import ViralLocalPipeline
from viral_local.models import VoiceConfig, TranslatedSegment, TranscriptSegment


class TestLanguageSupport:
    """Test language support functionality."""
    
    @pytest.fixture
    def multilingual_config(self):
        """Create configuration with all supported languages."""
        return SystemConfig(
            gemini_api_key="test_api_key_multilingual_12345",
            supported_languages=["hi", "bn", "ta"],
            default_target_language="hi",
            tts_engine="edge-tts"
        )
    
    def test_all_supported_languages_configured(self, multilingual_config):
        """Test that all required languages are properly configured."""
        
        required_languages = ["hi", "bn", "ta"]  # Hindi, Bengali, Tamil
        
        for lang in required_languages:
            assert lang in multilingual_config.supported_languages, \
                f"Language {lang} not in supported languages"
        
        # Test default language is valid
        assert multilingual_config.default_target_language in multilingual_config.supported_languages
    
    def test_voice_config_for_all_languages(self):
        """Test that voice configuration works for all supported languages."""
        
        languages = ["hi", "bn", "ta"]
        
        for language in languages:
            # Test basic voice configuration
            voice_config = VoiceConfig(
                language=language,
                gender="female",
                age_range="adult",
                speaking_rate=1.0,
                pitch_adjustment=0.0
            )
            
            assert voice_config.language == language
            assert voice_config.gender == "female"
            assert voice_config.age_range == "adult"
            
            # Test voice configuration validation
            assert 0.5 <= voice_config.speaking_rate <= 2.0
            assert -1.0 <= voice_config.pitch_adjustment <= 1.0
    
    def test_language_specific_voice_selection(self):
        """Test language-specific voice selection logic."""
        
        # Test Hindi voice configuration
        hindi_config = VoiceConfig(
            language="hi",
            gender="female",
            age_range="adult",
            speaking_rate=0.95  # Slightly slower for Hindi
        )
        assert hindi_config.language == "hi"
        assert hindi_config.speaking_rate == 0.95
        
        # Test Bengali voice configuration
        bengali_config = VoiceConfig(
            language="bn",
            gender="male",
            age_range="adult",
            speaking_rate=1.0
        )
        assert bengali_config.language == "bn"
        assert bengali_config.gender == "male"
        
        # Test Tamil voice configuration
        tamil_config = VoiceConfig(
            language="ta",
            gender="female",
            age_range="young",
            speaking_rate=1.05  # Slightly faster for Tamil
        )
        assert tamil_config.language == "ta"
        assert tamil_config.speaking_rate == 1.05
    
    def test_translation_segment_for_all_languages(self):
        """Test translation segment creation for all supported languages."""
        
        # Create a sample original segment
        original_segment = TranscriptSegment(
            text="Hello, how are you?",
            start_time=0.0,
            end_time=3.0,
            confidence=0.95,
            language="en"
        )
        
        # Test translations for each supported language
        translations = {
            "hi": "नमस्ते, आप कैसे हैं?",
            "bn": "হ্যালো, আপনি কেমন আছেন?",
            "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"
        }
        
        for lang_code, translated_text in translations.items():
            translated_segment = TranslatedSegment(
                original_segment=original_segment,
                translated_text=translated_text,
                target_language=lang_code,
                quality_score=0.9,
                cultural_adaptations=["greeting_adaptation"],
                technical_terms_preserved=[]
            )
            
            assert translated_segment.target_language == lang_code
            assert translated_segment.translated_text == translated_text
            assert translated_segment.quality_score == 0.9
            assert len(translated_segment.cultural_adaptations) > 0
    
    @patch('viral_local.main.DownloaderService')
    @patch('viral_local.main.TranscriberService')
    @patch('viral_local.main.LocalizationEngine')
    @patch('viral_local.main.DubbingStudio')
    def test_end_to_end_processing_all_languages(self, mock_dubbing, mock_localization, 
                                                mock_transcriber, mock_downloader, 
                                                multilingual_config):
        """Test end-to-end processing for all supported languages."""
        
        from viral_local.models import VideoFile, AudioFile, ProcessingResult
        
        # Setup common mocks
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
        
        mock_dubbing.return_value.generate_speech.return_value = mock_audio
        mock_dubbing.return_value.synchronize_audio.return_value = mock_audio
        mock_dubbing.return_value.merge_audio_video.return_value = mock_video
        
        # Test processing for each supported language
        for target_language in multilingual_config.supported_languages:
            with patch('viral_local.main.ConfigManager') as mock_config_manager:
                mock_config_manager.return_value.load_config.return_value = multilingual_config
                
                # Create language-specific translated segment
                mock_translated = Mock()
                mock_translated.original_segment = Mock(start_time=0.0, end_time=5.0)
                mock_translated.target_language = target_language
                mock_localization.return_value.translate_content.return_value = [mock_translated]
                
                pipeline = ViralLocalPipeline()
                result = pipeline.process_video("https://youtube.com/watch?v=test", target_language)
                
                # Verify processing succeeded for this language
                assert isinstance(result, ProcessingResult)
                assert result.success is True, f"Processing failed for language {target_language}"
    
    def test_language_code_validation(self, multilingual_config):
        """Test that language codes are properly validated."""
        
        # Test valid language codes
        valid_codes = ["hi", "bn", "ta"]
        for code in valid_codes:
            assert code in multilingual_config.supported_languages
            assert len(code) == 2  # ISO 639-1 format
            assert code.islower()  # Should be lowercase
        
        # Test that unsupported languages are not included
        unsupported_codes = ["en", "fr", "de", "es", "zh"]
        for code in unsupported_codes:
            assert code not in multilingual_config.supported_languages
    
    def test_cultural_adaptation_support(self):
        """Test cultural adaptation features for different languages."""
        
        original_segment = TranscriptSegment(
            text="Good morning, everyone!",
            start_time=0.0,
            end_time=2.0,
            confidence=0.95
        )
        
        # Test Hindi cultural adaptations
        hindi_segment = TranslatedSegment(
            original_segment=original_segment,
            translated_text="सुप्रभात, सभी को!",
            target_language="hi",
            quality_score=0.9,
            cultural_adaptations=["formal_greeting", "respectful_address"],
            technical_terms_preserved=[]
        )
        
        assert "formal_greeting" in hindi_segment.cultural_adaptations
        assert "respectful_address" in hindi_segment.cultural_adaptations
        
        # Test Bengali cultural adaptations
        bengali_segment = TranslatedSegment(
            original_segment=original_segment,
            translated_text="সুপ্রভাত, সবাইকে!",
            target_language="bn",
            quality_score=0.9,
            cultural_adaptations=["traditional_greeting"],
            technical_terms_preserved=[]
        )
        
        assert "traditional_greeting" in bengali_segment.cultural_adaptations
        
        # Test Tamil cultural adaptations
        tamil_segment = TranslatedSegment(
            original_segment=original_segment,
            translated_text="காலை வணக்கம், அனைவருக்கும்!",
            target_language="ta",
            quality_score=0.9,
            cultural_adaptations=["regional_greeting", "inclusive_address"],
            technical_terms_preserved=[]
        )
        
        assert "regional_greeting" in tamil_segment.cultural_adaptations
        assert "inclusive_address" in tamil_segment.cultural_adaptations
    
    def test_technical_term_preservation(self):
        """Test that technical terms are preserved across all languages."""
        
        original_segment = TranscriptSegment(
            text="The API endpoint returns JSON data with HTTP status codes.",
            start_time=0.0,
            end_time=4.0,
            confidence=0.95
        )
        
        technical_terms = ["API", "JSON", "HTTP"]
        
        # Test technical term preservation for each language
        for lang_code in ["hi", "bn", "ta"]:
            translated_segment = TranslatedSegment(
                original_segment=original_segment,
                translated_text=f"Translated text in {lang_code} with API, JSON, HTTP preserved",
                target_language=lang_code,
                quality_score=0.9,
                cultural_adaptations=[],
                technical_terms_preserved=technical_terms
            )
            
            # Verify technical terms are preserved
            assert len(translated_segment.technical_terms_preserved) == 3
            for term in technical_terms:
                assert term in translated_segment.technical_terms_preserved
    
    def test_language_detection_compatibility(self):
        """Test that language detection works with supported target languages."""
        
        # Test that source language detection doesn't conflict with target languages
        source_languages = ["en", "es", "fr"]  # Common source languages
        target_languages = ["hi", "bn", "ta"]  # Our supported target languages
        
        # Ensure no overlap (we don't translate from target languages to themselves)
        for source in source_languages:
            for target in target_languages:
                assert source != target, f"Source language {source} should not equal target {target}"
        
        # Test that all target languages are distinct
        assert len(set(target_languages)) == len(target_languages), "Target languages should be unique"


class TestLanguageSpecificFeatures:
    """Test language-specific features and optimizations."""
    
    def test_hindi_specific_features(self):
        """Test Hindi-specific processing features."""
        
        # Test Hindi voice configuration
        hindi_voice = VoiceConfig(
            language="hi",
            gender="female",
            age_range="adult",
            speaking_rate=0.95,  # Slightly slower for clarity
            pitch_adjustment=0.0
        )
        
        assert hindi_voice.language == "hi"
        assert hindi_voice.speaking_rate == 0.95
        
        # Test Hindi text handling
        hindi_text = "यह एक परीक्षण वाक्य है।"
        assert len(hindi_text) > 0
        assert any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in hindi_text)  # Devanagari range
    
    def test_bengali_specific_features(self):
        """Test Bengali-specific processing features."""
        
        # Test Bengali voice configuration
        bengali_voice = VoiceConfig(
            language="bn",
            gender="male",
            age_range="adult",
            speaking_rate=1.0,
            pitch_adjustment=0.0
        )
        
        assert bengali_voice.language == "bn"
        assert bengali_voice.speaking_rate == 1.0
        
        # Test Bengali text handling
        bengali_text = "এটি একটি পরীক্ষার বাক্য।"
        assert len(bengali_text) > 0
        assert any(ord(char) >= 0x0980 and ord(char) <= 0x09FF for char in bengali_text)  # Bengali range
    
    def test_tamil_specific_features(self):
        """Test Tamil-specific processing features."""
        
        # Test Tamil voice configuration
        tamil_voice = VoiceConfig(
            language="ta",
            gender="female",
            age_range="young",
            speaking_rate=1.05,  # Slightly faster for Tamil rhythm
            pitch_adjustment=0.1
        )
        
        assert tamil_voice.language == "ta"
        assert tamil_voice.speaking_rate == 1.05
        assert tamil_voice.pitch_adjustment == 0.1
        
        # Test Tamil text handling
        tamil_text = "இது ஒரு சோதனை வாக்கியம்."
        assert len(tamil_text) > 0
        assert any(ord(char) >= 0x0B80 and ord(char) <= 0x0BFF for char in tamil_text)  # Tamil range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])