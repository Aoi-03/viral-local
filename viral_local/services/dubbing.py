"""
DubbingStudio for audio synthesis and video assembly.

This module handles text-to-speech generation, audio-video synchronization,
and final video assembly using TTS engines and MoviePy.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import edge_tts
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import time

from ..models import TranslatedSegment, VoiceConfig, AudioFile, VideoFile, TimingData, ProcessingResult
from ..config import SystemConfig
from ..utils import get_logger, AudioGenerationError, VideoAssemblyError
from ..utils.logging import LoggerMixin, log_performance


class DubbingStudio(LoggerMixin):
    """Studio for synthesizing localized audio and assembling final video."""
    
    # Voice mappings for different languages and characteristics
    EDGE_TTS_VOICES = {
        "hi": {  # Hindi
            "male": {
                "young": "hi-IN-MadhurNeural",
                "adult": "hi-IN-SwarasNeural", 
                "elderly": "hi-IN-MadhurNeural"
            },
            "female": {
                "young": "hi-IN-SwaraNeural",
                "adult": "hi-IN-SwaraNeural",
                "elderly": "hi-IN-SwaraNeural"
            }
        },
        "bn": {  # Bengali
            "male": {
                "young": "bn-IN-BashkarNeural",
                "adult": "bn-IN-BashkarNeural",
                "elderly": "bn-IN-BashkarNeural"
            },
            "female": {
                "young": "bn-IN-TanishaaNeural", 
                "adult": "bn-IN-TanishaaNeural",
                "elderly": "bn-IN-TanishaaNeural"
            }
        },
        "ta": {  # Tamil
            "male": {
                "young": "ta-IN-ValluvarNeural",
                "adult": "ta-IN-ValluvarNeural", 
                "elderly": "ta-IN-ValluvarNeural"
            },
            "female": {
                "young": "ta-IN-PallaviNeural",
                "adult": "ta-IN-PallaviNeural",
                "elderly": "ta-IN-PallaviNeural"
            }
        }
    }
    
    def __init__(self, config: SystemConfig):
        """Initialize the dubbing studio.
        
        Args:
            config: System configuration
        """
        self.config = config
        self._temp_dir = Path(config.temp_dir) / "dubbing"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
    
    
    def _select_voice(self, voice_config: VoiceConfig) -> str:
        """Select appropriate voice based on configuration.
        
        Args:
            voice_config: Voice configuration parameters
            
        Returns:
            Voice name for the TTS engine
            
        Raises:
            AudioGenerationError: If voice selection fails
        """
        try:
            # Use explicit voice name if provided
            if voice_config.voice_name:
                return voice_config.voice_name
            
            # Get language voices
            lang_voices = self.EDGE_TTS_VOICES.get(voice_config.language)
            if not lang_voices:
                raise AudioGenerationError(
                    f"Language '{voice_config.language}' not supported",
                    tts_engine="edge-tts",
                    voice_config=str(voice_config)
                )
            
            # Get gender voices
            gender_voices = lang_voices.get(voice_config.gender)
            if not gender_voices:
                # Fallback to first available gender
                gender_voices = list(lang_voices.values())[0]
                self.logger.warning(f"Gender '{voice_config.gender}' not available, using fallback")
            
            # Get age-specific voice
            voice_name = gender_voices.get(voice_config.age_range)
            if not voice_name:
                # Fallback to adult voice
                voice_name = gender_voices.get("adult") or list(gender_voices.values())[0]
                self.logger.warning(f"Age range '{voice_config.age_range}' not available, using fallback")
            
            self.logger.info(f"Selected voice: {voice_name} for {voice_config.language}/{voice_config.gender}/{voice_config.age_range}")
            return voice_name
            
        except Exception as e:
            raise AudioGenerationError(
                f"Failed to select voice: {str(e)}",
                tts_engine="edge-tts",
                voice_config=str(voice_config)
            )
    
    async def _generate_segment_audio(self, text: str, voice_name: str, voice_config: VoiceConfig) -> bytes:
        """Generate audio for a single text segment.
        
        Args:
            text: Text to synthesize
            voice_name: Voice to use for synthesis
            voice_config: Voice configuration parameters
            
        Returns:
            Audio data as bytes
            
        Raises:
            AudioGenerationError: If audio generation fails
        """
        try:
            # Create SSML with voice configuration
            ssml = self._create_ssml(text, voice_name, voice_config)
            
            # Generate audio using Edge-TTS
            communicate = edge_tts.Communicate(ssml, voice_name)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if not audio_data:
                raise AudioGenerationError(
                    f"No audio data generated for text: {text[:50]}...",
                    tts_engine="edge-tts",
                    voice_config=str(voice_config)
                )
            
            return audio_data
            
        except Exception as e:
            raise AudioGenerationError(
                f"Failed to generate audio for segment: {str(e)}",
                tts_engine="edge-tts",
                voice_config=str(voice_config)
            )
    
    def _generate_segment_audio_fallback(self, text: str, voice_config: VoiceConfig) -> bytes:
        """Generate audio using fallback TTS method (pyttsx3).
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration parameters
            
        Returns:
            Audio data as bytes
            
        Raises:
            AudioGenerationError: If fallback audio generation fails
        """
        try:
            import pyttsx3
            import tempfile
            import os
            
            # Initialize pyttsx3 engine
            engine = pyttsx3.init()
            
            # Configure voice settings
            voices = engine.getProperty('voices')
            
            # Try to find a suitable voice for the language/gender
            selected_voice = None
            for voice in voices:
                voice_id = voice.id.lower()
                voice_name = voice.name.lower() if hasattr(voice, 'name') else voice_id
                
                # Look for language-specific voices first
                if voice_config.language == 'hi' and any(keyword in voice_name for keyword in ['hindi', 'india']):
                    selected_voice = voice.id
                    break
                elif voice_config.language == 'bn' and any(keyword in voice_name for keyword in ['bengali', 'bangla']):
                    selected_voice = voice.id
                    break
                elif voice_config.language == 'ta' and any(keyword in voice_name for keyword in ['tamil']):
                    selected_voice = voice.id
                    break
                
                # Gender preference as secondary criteria
                if voice_config.gender == 'female' and any(keyword in voice_name for keyword in ['female', 'woman', 'zira', 'hazel']):
                    if not selected_voice:  # Only if no language match found
                        selected_voice = voice.id
                elif voice_config.gender == 'male' and any(keyword in voice_name for keyword in ['male', 'man', 'david', 'mark']):
                    if not selected_voice:
                        selected_voice = voice.id
            
            # If no specific voice found, use the first available
            if not selected_voice and voices:
                selected_voice = voices[0].id
                self.logger.info(f"Using default voice: {voices[0].name if hasattr(voices[0], 'name') else selected_voice}")
            
            if selected_voice:
                engine.setProperty('voice', selected_voice)
                self.logger.info(f"Selected TTS voice: {selected_voice}")
            
            # Set speech rate and volume
            rate = int(200 * voice_config.speaking_rate)  # Base rate of 200 WPM
            engine.setProperty('rate', max(100, min(300, rate)))  # Clamp between 100-300
            engine.setProperty('volume', 0.9)
            
            # Create a temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_file = temp_file.name
            
            try:
                # Generate speech to file
                engine.save_to_file(text, temp_audio_file)
                engine.runAndWait()
                
                # Wait a moment for file to be written
                import time
                time.sleep(1.0)
                
                # Check if file was created and has content
                if os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 1000:  # At least 1KB
                    with open(temp_audio_file, 'rb') as f:
                        audio_data = f.read()
                    
                    self.logger.info(f"Successfully generated fallback TTS audio: {len(audio_data)} bytes for text: {text[:50]}...")
                    return audio_data
                else:
                    self.logger.warning("pyttsx3 failed to generate audio file, creating silence")
                    return self._generate_silence_audio(max(1.0, len(text) * 0.1))
                    
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_audio_file):
                        os.unlink(temp_audio_file)
                except:
                    pass
                
        except ImportError:
            self.logger.warning("pyttsx3 not available, creating silence")
            return self._generate_silence_audio(max(1.0, len(text) * 0.1))
        except Exception as e:
            self.logger.error(f"Fallback TTS generation failed: {e}")
            return self._generate_silence_audio(max(1.0, len(text) * 0.1))
    
    def _generate_segment_audio_gtts(self, text: str, voice_config: VoiceConfig) -> bytes:
        """Generate audio using Google Text-to-Speech (gTTS).
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration parameters
            
        Returns:
            Audio data as bytes
        """
        try:
            from gtts import gTTS
            import tempfile
            import os
            
            # Map language codes for gTTS
            gtts_lang_map = {
                'hi': 'hi',  # Hindi
                'bn': 'bn',  # Bengali  
                'ta': 'ta',  # Tamil
                'en': 'en'   # English fallback
            }
            
            lang = gtts_lang_map.get(voice_config.language, 'en')
            
            # Create gTTS object
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_audio_file = temp_file.name
            
            try:
                # Save to temporary file
                tts.save(temp_audio_file)
                
                # Convert MP3 to WAV using pydub
                from pydub import AudioSegment
                
                # Load MP3 and convert to WAV
                audio = AudioSegment.from_mp3(temp_audio_file)
                
                # Convert to WAV format in memory
                wav_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                audio.export(wav_temp.name, format="wav")
                
                # Read WAV data
                with open(wav_temp.name, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up temp files
                os.unlink(wav_temp.name)
                
                if audio_data and len(audio_data) > 1000:  # At least 1KB
                    self.logger.info(f"Successfully generated gTTS audio: {len(audio_data)} bytes for text: {text[:50]}...")
                    return audio_data
                else:
                    raise Exception("gTTS generated empty audio")
                    
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_audio_file):
                        os.unlink(temp_audio_file)
                except:
                    pass
                    
        except ImportError:
            raise Exception("gTTS not available. Install with: pip install gtts")
        except Exception as e:
            raise Exception(f"gTTS generation failed: {str(e)}")
    
    def _generate_silence_audio(self, duration_seconds: float) -> bytes:
        """Generate silence audio as fallback.
        
        Args:
            duration_seconds: Duration of silence in seconds
            
        Returns:
            WAV audio data as bytes
        """
        try:
            import wave
            import struct
            
            sample_rate = self.config.audio_sample_rate
            num_samples = int(duration_seconds * sample_rate)
            
            # Ensure minimum duration
            if num_samples < sample_rate:  # At least 1 second
                num_samples = sample_rate
                duration_seconds = 1.0
            
            # Create WAV file in memory using BytesIO
            import io
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Create silence data (zeros)
                silence_data = b'\x00\x00' * num_samples  # 16-bit silence
                wav_file.writeframes(silence_data)
            
            # Get the WAV data
            audio_data = wav_buffer.getvalue()
            wav_buffer.close()
            
            self.logger.info(f"Generated {duration_seconds:.1f}s of silence audio ({len(audio_data)} bytes)")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate silence audio: {e}")
            # Return minimal WAV with 1 second of silence as last resort
            sample_rate = 22050
            num_samples = sample_rate  # 1 second
            
            # Create minimal WAV header + silence data
            header = struct.pack('<4sI4s4sIHHIIHH4sI', 
                b'RIFF', 36 + num_samples * 2, b'WAVE', b'fmt ', 16, 1, 1, 
                sample_rate, sample_rate * 2, 2, 16, b'data', num_samples * 2)
            silence = b'\x00\x00' * num_samples
            
            return header + silence
    
    def _create_ssml(self, text: str, voice_name: str, voice_config: VoiceConfig) -> str:
        """Create SSML markup for enhanced speech synthesis.
        
        Args:
            text: Text to synthesize
            voice_name: Voice to use
            voice_config: Voice configuration parameters
            
        Returns:
            SSML markup string
        """
        # Calculate rate and pitch adjustments
        rate_percent = int((voice_config.speaking_rate - 1.0) * 100)
        pitch_percent = int(voice_config.pitch_adjustment * 50)  # Scale to reasonable range
        
        # Create SSML with prosody controls
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{voice_config.language}">
            <voice name="{voice_name}">
                <prosody rate="{rate_percent:+d}%" pitch="{pitch_percent:+d}%">
                    {text}
                </prosody>
            </voice>
        </speak>"""
        
        return ssml
    
    def _normalize_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio quality and apply processing.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Processed audio data
        """
        try:
            # Normalize volume to prevent clipping
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Apply gentle compression to even out volume
            audio_data = np.tanh(audio_data * 1.2) * 0.8
            
            # Apply high-pass filter to remove low-frequency noise
            if sample_rate > 8000:  # Only if sample rate is high enough
                from scipy.signal import butter, filtfilt
                nyquist = sample_rate / 2
                low_cutoff = 80 / nyquist  # 80 Hz high-pass
                b, a = butter(2, low_cutoff, btype='high')
                audio_data = filtfilt(b, a, audio_data)
            
            return audio_data
            
        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {e}, using original audio")
            return audio_data
    
    @log_performance("TTS Generation")
    def generate_speech(self, translated_segments: List[TranslatedSegment], voice_config: VoiceConfig) -> AudioFile:
        """Generate speech from translated text segments.
        
        Args:
            translated_segments: List of TranslatedSegment objects
            voice_config: Voice configuration for TTS
            
        Returns:
            AudioFile object with generated speech
            
        Raises:
            AudioGenerationError: If speech generation fails
        """
        try:
            self.logger.info(f"Generating speech for {len(translated_segments)} segments using {self.config.tts_engine}")
            
            if not translated_segments:
                raise AudioGenerationError("No translated segments provided for speech generation")
            
            # Select appropriate voice
            voice_name = self._select_voice(voice_config)
            
            # Generate audio for each segment
            segment_audio_files = []
            total_duration = 0
            
            for i, segment in enumerate(translated_segments):
                try:
                    self.logger.debug(f"Processing segment {i+1}/{len(translated_segments)}: {segment.translated_text[:50]}...")
                    
                    # Generate audio for this segment
                    try:
                        audio_data = asyncio.run(self._generate_segment_audio(
                            segment.translated_text, voice_name, voice_config
                        ))
                    except Exception as edge_error:
                        self.logger.warning(f"Edge-TTS failed for segment {i}, trying fallback TTS: {edge_error}")
                        # Try multiple fallback options
                        audio_data = None
                        
                        # First try gTTS (Google Text-to-Speech)
                        try:
                            audio_data = self._generate_segment_audio_gtts(
                                segment.translated_text, voice_config
                            )
                            self.logger.info(f"Successfully used gTTS for segment {i}")
                        except Exception as gtts_error:
                            self.logger.warning(f"gTTS also failed for segment {i}: {gtts_error}")
                            
                            # Then try pyttsx3
                            try:
                                audio_data = self._generate_segment_audio_fallback(
                                    segment.translated_text, voice_config
                                )
                                self.logger.info(f"Successfully used pyttsx3 for segment {i}")
                            except Exception as pyttsx3_error:
                                self.logger.warning(f"pyttsx3 also failed for segment {i}: {pyttsx3_error}")
                                # Last resort: silence
                                duration = max(1.0, len(segment.translated_text) * 0.1)
                                audio_data = self._generate_silence_audio(duration)
                                self.logger.warning(f"Using silence for segment {i}")
                        
                        if not audio_data:
                            duration = max(1.0, len(segment.translated_text) * 0.1)
                            audio_data = self._generate_silence_audio(duration)
                    
                    # Save segment audio to temporary file
                    segment_file = self._temp_dir / f"segment_{i:04d}.wav"
                    with open(segment_file, 'wb') as f:
                        f.write(audio_data)
                    
                    # Load and process audio
                    audio_array, sr = librosa.load(str(segment_file), sr=self.config.audio_sample_rate)
                    
                    # Check if audio is valid
                    if len(audio_array) == 0:
                        self.logger.warning(f"Empty audio generated for segment {i}, skipping")
                        segment_file.unlink(missing_ok=True)
                        continue
                    
                    audio_array = self._normalize_audio_quality(audio_array, sr)
                    
                    # Save processed audio
                    processed_file = self._temp_dir / f"processed_segment_{i:04d}.wav"
                    sf.write(str(processed_file), audio_array, sr)
                    
                    segment_audio_files.append({
                        'file': str(processed_file),
                        'duration': len(audio_array) / sr,
                        'start_time': segment.original_segment.start_time,
                        'end_time': segment.original_segment.end_time
                    })
                    
                    total_duration += len(audio_array) / sr
                    
                    # Clean up temporary segment file
                    segment_file.unlink(missing_ok=True)
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate audio for segment {i}: {e}")
                    raise AudioGenerationError(
                        f"Failed to generate audio for segment {i}: {str(e)}",
                        tts_engine=self.config.tts_engine,
                        voice_config=str(voice_config)
                    )
            
            # Combine all segments into final audio file
            if not segment_audio_files:
                raise AudioGenerationError("No valid audio segments were generated")
            
            final_audio_file = self._temp_dir / f"generated_speech_{int(time.time())}.wav"
            self._combine_audio_segments(segment_audio_files, str(final_audio_file))
            
            # Clean up segment files
            for segment_info in segment_audio_files:
                Path(segment_info['file']).unlink(missing_ok=True)
            
            # Create AudioFile object
            if total_duration <= 0:
                raise AudioGenerationError("Generated audio has invalid duration")
            
            audio_file = AudioFile(
                file_path=str(final_audio_file),
                duration=total_duration,
                sample_rate=self.config.audio_sample_rate,
                channels=1,  # Mono audio for speech
                format="wav"
            )
            
            self.logger.info(f"Successfully generated speech audio: {total_duration:.2f}s duration")
            return audio_file
            
        except AudioGenerationError:
            raise
        except Exception as e:
            raise AudioGenerationError(
                f"Speech generation failed: {str(e)}",
                tts_engine=self.config.tts_engine,
                voice_config=str(voice_config)
            )
    
    def _combine_audio_segments(self, segment_files: List[Dict], output_file: str) -> None:
        """Combine individual audio segments into a single file with proper timing.
        
        Args:
            segment_files: List of segment file information
            output_file: Path to output combined audio file
        """
        try:
            # Sort segments by start time
            segment_files.sort(key=lambda x: x['start_time'])
            
            # Create silence for gaps and combine segments
            combined_audio = AudioSegment.empty()
            current_time = 0
            
            for segment_info in segment_files:
                start_time = segment_info['start_time']
                
                # Add silence if there's a gap
                if start_time > current_time:
                    silence_duration = (start_time - current_time) * 1000  # Convert to milliseconds
                    silence = AudioSegment.silent(duration=int(silence_duration))
                    combined_audio += silence
                
                # Add segment audio
                segment_audio = AudioSegment.from_wav(segment_info['file'])
                combined_audio += segment_audio
                
                current_time = start_time + segment_info['duration']
            
            # Export combined audio
            combined_audio.export(output_file, format="wav")
            
        except Exception as e:
            raise AudioGenerationError(f"Failed to combine audio segments: {str(e)}")
    
    
    def _calculate_timing_adjustments(self, original_timing: TimingData) -> List[Tuple[float, float, float]]:
        """Calculate timing adjustments needed for synchronization.
        
        Args:
            original_timing: Original timing data from video
            
        Returns:
            List of (start_time, end_time, stretch_factor) tuples
        """
        adjustments = []
        
        for orig_seg, trans_seg in zip(original_timing.original_segments, original_timing.target_segments):
            original_duration = orig_seg.end_time - orig_seg.start_time
            
            # Estimate target duration based on text length ratio
            # This is a heuristic - actual TTS duration may vary
            text_ratio = len(trans_seg.translated_text) / max(len(orig_seg.text), 1)
            estimated_duration = original_duration * text_ratio
            
            # Calculate stretch factor to fit original timing
            stretch_factor = original_duration / max(estimated_duration, 0.1)
            
            # Limit stretch factor to reasonable bounds
            stretch_factor = max(0.5, min(2.0, stretch_factor))
            
            adjustments.append((orig_seg.start_time, orig_seg.end_time, stretch_factor))
        
        return adjustments
    
    def _apply_time_stretching(self, audio_data: np.ndarray, sample_rate: int, stretch_factor: float) -> np.ndarray:
        """Apply time stretching to audio while preserving pitch.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            stretch_factor: Factor to stretch/compress time (>1 = slower, <1 = faster)
            
        Returns:
            Time-stretched audio data
        """
        try:
            if abs(stretch_factor - 1.0) < 0.05:  # No significant change needed
                return audio_data
            
            # Use librosa's time stretching with pitch preservation
            stretched_audio = librosa.effects.time_stretch(audio_data, rate=1.0/stretch_factor)
            
            return stretched_audio
            
        except Exception as e:
            self.logger.warning(f"Time stretching failed: {e}, using original audio")
            return audio_data
    
    def _adjust_audio_timing(self, audio_file: AudioFile, timing_adjustments: List[Tuple[float, float, float]]) -> AudioFile:
        """Adjust audio timing to match original video segments.
        
        Args:
            audio_file: Generated audio file
            timing_adjustments: List of timing adjustments to apply
            
        Returns:
            AudioFile with adjusted timing
        """
        try:
            # Load the generated audio
            audio_data, sr = librosa.load(audio_file.file_path, sr=None)
            
            # Create output audio array
            max_end_time = max(adj[1] for adj in timing_adjustments)
            output_length = int(max_end_time * sr)
            output_audio = np.zeros(output_length)
            
            # Process each segment with its timing adjustment
            current_pos = 0
            
            for i, (start_time, end_time, stretch_factor) in enumerate(timing_adjustments):
                # Calculate segment boundaries in the generated audio
                segment_start = current_pos
                
                # Estimate segment length (this is approximate)
                estimated_segment_length = int((end_time - start_time) * sr / stretch_factor)
                segment_end = min(segment_start + estimated_segment_length, len(audio_data))
                
                if segment_start >= len(audio_data):
                    break
                
                # Extract segment
                segment_audio = audio_data[segment_start:segment_end]
                
                if len(segment_audio) > 0:
                    # Apply time stretching
                    adjusted_segment = self._apply_time_stretching(segment_audio, sr, stretch_factor)
                    
                    # Place in output at correct timing
                    output_start = int(start_time * sr)
                    output_end = min(output_start + len(adjusted_segment), len(output_audio))
                    
                    if output_start < len(output_audio):
                        copy_length = min(len(adjusted_segment), output_end - output_start)
                        output_audio[output_start:output_start + copy_length] = adjusted_segment[:copy_length]
                
                current_pos = segment_end
            
            # Save adjusted audio
            adjusted_file = self._temp_dir / f"synchronized_{int(time.time())}.wav"
            sf.write(str(adjusted_file), output_audio, sr)
            
            # Create new AudioFile object
            synchronized_audio = AudioFile(
                file_path=str(adjusted_file),
                duration=len(output_audio) / sr,
                sample_rate=sr,
                channels=1,
                format="wav"
            )
            
            return synchronized_audio
            
        except Exception as e:
            self.logger.warning(f"Audio timing adjustment failed: {e}, using original audio")
            return audio_file
    
    def _apply_audio_processing(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply audio processing for consistent volume and quality.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Processed audio data
        """
        try:
            # Dynamic range compression
            audio_data = np.tanh(audio_data * 1.5) * 0.85
            
            # Noise gate to remove low-level noise
            noise_threshold = 0.01
            audio_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)
            
            # Gentle low-pass filter to remove harsh frequencies
            if sample_rate > 16000:
                from scipy.signal import butter, filtfilt
                nyquist = sample_rate / 2
                high_cutoff = 8000 / nyquist  # 8kHz low-pass
                b, a = butter(2, high_cutoff, btype='low')
                audio_data = filtfilt(b, a, audio_data)
            
            # Final normalization
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
            
            return audio_data
            
        except Exception as e:
            self.logger.warning(f"Audio processing failed: {e}, using original audio")
            return audio_data
    
    @log_performance("Audio Synchronization")
    def synchronize_audio(self, new_audio: AudioFile, original_timing: TimingData) -> AudioFile:
        """Synchronize generated audio with original video timing.
        
        Args:
            new_audio: Generated AudioFile object
            original_timing: TimingData from original video
            
        Returns:
            Synchronized AudioFile object
            
        Raises:
            AudioGenerationError: If synchronization fails
        """
        try:
            self.logger.info(f"Synchronizing audio with {len(original_timing.original_segments)} segments")
            
            if not original_timing.original_segments:
                raise AudioGenerationError("No timing data provided for synchronization")
            
            # Calculate timing adjustments needed
            timing_adjustments = self._calculate_timing_adjustments(original_timing)
            
            # Apply timing adjustments to the audio
            synchronized_audio = self._adjust_audio_timing(new_audio, timing_adjustments)
            
            # Load and process the synchronized audio for quality
            audio_data, sr = librosa.load(synchronized_audio.file_path, sr=None)
            processed_audio = self._apply_audio_processing(audio_data, sr)
            
            # Save final processed audio
            final_file = self._temp_dir / f"final_synchronized_{int(time.time())}.wav"
            sf.write(str(final_file), processed_audio, sr)
            
            # Update AudioFile object
            final_audio = AudioFile(
                file_path=str(final_file),
                duration=len(processed_audio) / sr,
                sample_rate=sr,
                channels=1,
                format="wav"
            )
            
            # Clean up intermediate file
            Path(synchronized_audio.file_path).unlink(missing_ok=True)
            
            self.logger.info(f"Audio synchronization completed: {final_audio.duration:.2f}s duration")
            return final_audio
            
        except AudioGenerationError:
            raise
        except Exception as e:
            raise AudioGenerationError(
                f"Audio synchronization failed: {str(e)}",
                tts_engine=self.config.tts_engine
            )
    
    
    def _validate_video_audio_compatibility(self, video: VideoFile, audio: AudioFile) -> None:
        """Validate that video and audio files are compatible for merging.
        
        Args:
            video: VideoFile object
            audio: AudioFile object
            
        Raises:
            VideoAssemblyError: If files are not compatible
        """
        try:
            # Check if files exist
            if not Path(video.file_path).exists():
                raise VideoAssemblyError(f"Video file not found: {video.file_path}")
            
            if not Path(audio.file_path).exists():
                raise VideoAssemblyError(f"Audio file not found: {audio.file_path}")
            
            # Check duration compatibility (allow some tolerance)
            duration_diff = abs(video.duration - audio.duration)
            max_tolerance = max(1.0, video.duration * 0.05)  # 5% tolerance or 1 second
            
            if duration_diff > max_tolerance:
                self.logger.warning(
                    f"Duration mismatch: video={video.duration:.2f}s, audio={audio.duration:.2f}s, "
                    f"diff={duration_diff:.2f}s"
                )
            
        except VideoAssemblyError:
            raise
        except Exception as e:
            raise VideoAssemblyError(f"Compatibility validation failed: {str(e)}")
    
    def _preserve_video_metadata(self, original_video: VideoFile, output_path: str) -> Dict[str, any]:
        """Extract and prepare metadata for preservation in output video.
        
        Args:
            original_video: Original VideoFile object
            output_path: Path where output video will be saved
            
        Returns:
            Dictionary of metadata to preserve
        """
        try:
            preserved_metadata = {}
            
            # Copy relevant metadata
            if original_video.metadata:
                # Preserve safe metadata fields
                safe_fields = ['title', 'description', 'upload_date', 'uploader', 'tags']
                for field in safe_fields:
                    if field in original_video.metadata:
                        preserved_metadata[field] = original_video.metadata[field]
            
            # Add processing metadata
            preserved_metadata.update({
                'processed_by': 'Viral-Local',
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'original_duration': original_video.duration,
                'original_resolution': f"{original_video.resolution[0]}x{original_video.resolution[1]}",
                'localization_engine': 'viral-local-v1.0'
            })
            
            return preserved_metadata
            
        except Exception as e:
            self.logger.warning(f"Metadata preservation failed: {e}")
            return {'processed_by': 'Viral-Local'}
    
    def _get_optimal_codec_settings(self, video: VideoFile) -> Dict[str, any]:
        """Get optimal codec settings based on video characteristics.
        
        Args:
            video: VideoFile object
            
        Returns:
            Dictionary of codec settings
        """
        try:
            settings = {
                'codec': 'libx264',
                'audio_codec': 'aac',
                'preset': 'medium',
                'crf': 23,  # Constant Rate Factor for quality
            }
            
            # Adjust settings based on resolution
            width, height = video.resolution
            total_pixels = width * height
            
            if total_pixels >= 1920 * 1080:  # 1080p or higher
                settings.update({
                    'preset': 'slow',  # Better compression for high-res
                    'crf': 21,  # Higher quality
                    'bitrate': '5000k'
                })
            elif total_pixels >= 1280 * 720:  # 720p
                settings.update({
                    'crf': 22,
                    'bitrate': '3000k'
                })
            else:  # Lower resolution
                settings.update({
                    'preset': 'fast',
                    'crf': 24,
                    'bitrate': '1500k'
                })
            
            # Audio settings
            settings.update({
                'audio_bitrate': '128k',
                'audio_fps': 44100
            })
            
            return settings
            
        except Exception as e:
            self.logger.warning(f"Failed to determine optimal codec settings: {e}")
            return {
                'codec': 'libx264',
                'audio_codec': 'aac',
                'preset': 'medium',
                'crf': 23,
                'audio_bitrate': '128k'
            }
    
    def _merge_audio_with_video(self, video_path: str, audio_path: str, output_path: str, codec_settings: Dict[str, any]) -> None:
        """Merge audio with video using MoviePy.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path for output file
            codec_settings: Codec settings for encoding
        """
        video_clip = None
        audio_clip = None
        
        try:
            # Load video clip
            video_clip = VideoFileClip(video_path)
            
            # Load new audio clip
            audio_clip = AudioFileClip(audio_path)
            
            # Adjust audio duration to match video if needed
            if abs(video_clip.duration - audio_clip.duration) > 0.1:
                if audio_clip.duration < video_clip.duration:
                    # For simplicity, just use the audio as-is and let MoviePy handle it
                    self.logger.warning(f"Audio shorter than video by {video_clip.duration - audio_clip.duration:.1f}s")
                else:
                    # Trim audio if it's longer - use set_duration instead of subclip
                    audio_clip = audio_clip.set_duration(video_clip.duration)
            
            # Set the new audio to the video - use with_audio instead of set_audio
            final_video = video_clip.with_audio(audio_clip)
            
            # Write the final video with optimal settings
            write_params = {
                'filename': output_path,
                'codec': codec_settings['codec'],
                'audio_codec': codec_settings['audio_codec'],
                'preset': codec_settings.get('preset', 'medium'),
                'ffmpeg_params': ['-crf', str(codec_settings.get('crf', 23))]
            }
            
            # Add bitrate if specified
            if 'bitrate' in codec_settings:
                write_params['bitrate'] = codec_settings['bitrate']
            
            if 'audio_bitrate' in codec_settings:
                write_params['audio_bitrate'] = codec_settings['audio_bitrate']
            
            # Remove verbose and logger parameters for newer MoviePy versions
            final_video.write_videofile(**write_params)
            
        finally:
            # Clean up clips to free memory
            if video_clip:
                video_clip.close()
            if audio_clip:
                audio_clip.close()
    
    @log_performance("Video Assembly")
    def merge_audio_video(self, video: VideoFile, audio: AudioFile) -> VideoFile:
        """Merge localized audio with original video.
        
        Args:
            video: Original VideoFile object
            audio: Localized AudioFile object
            
        Returns:
            Final VideoFile object with merged audio
            
        Raises:
            VideoAssemblyError: If video assembly fails
        """
        try:
            self.logger.info(f"Merging audio with video: {video.file_path}")
            
            # Validate compatibility
            self._validate_video_audio_compatibility(video, audio)
            
            # Prepare output path
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            output_filename = f"localized_video_{timestamp}.{self.config.video_output_format}"
            output_path = output_dir / output_filename
            
            # Get optimal codec settings
            codec_settings = self._get_optimal_codec_settings(video)
            
            # Preserve metadata
            preserved_metadata = self._preserve_video_metadata(video, str(output_path))
            
            # Merge audio and video
            self._merge_audio_with_video(
                video.file_path,
                audio.file_path,
                str(output_path),
                codec_settings
            )
            
            # Verify output file was created
            if not output_path.exists():
                raise VideoAssemblyError("Output video file was not created")
            
            # Get output file info
            with VideoFileClip(str(output_path)) as clip:
                output_duration = clip.duration
                output_resolution = (clip.w, clip.h)
            
            # Create final VideoFile object
            final_video = VideoFile(
                file_path=str(output_path),
                duration=output_duration,
                resolution=output_resolution,
                format=self.config.video_output_format,
                metadata=preserved_metadata
            )
            
            self.logger.info(
                f"Video assembly completed: {output_path.name}, "
                f"duration={output_duration:.2f}s, "
                f"resolution={output_resolution[0]}x{output_resolution[1]}"
            )
            
            return final_video
            
        except VideoAssemblyError:
            raise
        except Exception as e:
            raise VideoAssemblyError(
                f"Video assembly failed: {str(e)}",
                video_file=video.file_path,
                audio_file=audio.file_path
            )
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files created during processing."""
        try:
            temp_files = list(self._temp_dir.glob("*"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            
            self.logger.info(f"Cleaned up {len(temp_files)} temporary files")
            
        except Exception as e:
            self.logger.warning(f"Temp file cleanup failed: {e}")