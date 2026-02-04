"""
TranscriberService for speech recognition and transcription.

This module handles audio transcription using Whisper, language detection,
and transcription segmentation for further processing.
"""

import os
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from ..models import AudioFile, Transcription, TranscriptSegment, ProcessingResult
from ..config import SystemConfig
from ..utils import get_logger, TranscriptionError
from ..utils.logging import LoggerMixin

# Optional imports - handle gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    whisper = None
    WHISPER_AVAILABLE = False


class TranscriberService(LoggerMixin):
    """Service for transcribing audio to text with timestamps."""
    
    def __init__(self, config: SystemConfig):
        """Initialize the transcriber service.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.model = None
        self.device = None
        self._model_cache = {}
        self._initialize_device()
        self._load_whisper_model()
    
    def _initialize_device(self) -> None:
        """Initialize and configure the compute device (GPU/CPU)."""
        if not TORCH_AVAILABLE:
            self.device = "cpu"
            self.logger.warning("PyTorch not available, using CPU")
            return
            
        try:
            if self.config.enable_gpu and torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU acceleration enabled: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.device = "cpu"
                if self.config.enable_gpu:
                    self.logger.warning("GPU requested but not available, falling back to CPU")
                else:
                    self.logger.info("Using CPU for transcription")
        except Exception as e:
            self.logger.error(f"Error initializing device: {e}")
            self.device = "cpu"
            self.logger.info("Falling back to CPU due to device initialization error")
    
    def _load_whisper_model(self) -> None:
        """Load and cache the Whisper model."""
        if not WHISPER_AVAILABLE:
            self.logger.error("Whisper not available - please install openai-whisper")
            raise TranscriptionError("Whisper not available - please install openai-whisper")
            
        model_size = self.config.whisper_model_size
        cache_key = f"{model_size}_{self.device}"
        
        try:
            # Check if model is already cached
            if cache_key in self._model_cache:
                self.model = self._model_cache[cache_key]
                self.logger.info(f"Using cached Whisper model: {model_size}")
                return
            
            # Load model with performance optimization
            start_time = time.time()
            self.logger.info(f"Loading Whisper model: {model_size} on {self.device}")
            
            # Load model with appropriate settings
            self.model = whisper.load_model(
                model_size, 
                device=self.device,
                download_root=os.path.join(self.config.cache_dir, "whisper_models")
            )
            
            # Cache the model for reuse
            if self.config.cache_enabled:
                self._model_cache[cache_key] = self.model
            
            load_time = time.time() - start_time
            self.logger.info(f"Whisper model loaded successfully in {load_time:.2f}s")
            
            # Log model information
            self._log_model_info()
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model {model_size}: {e}")
            # Try fallback to smaller model
            if model_size != "base":
                self.logger.info("Attempting fallback to base model")
                self.config.whisper_model_size = "base"
                self._load_whisper_model()
            else:
                raise TranscriptionError(f"Failed to load Whisper model: {e}")
    
    def _log_model_info(self) -> None:
        """Log information about the loaded model."""
        if self.model is None or not TORCH_AVAILABLE:
            return
            
        try:
            # Get model parameters count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            self.logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            # Log memory usage if on GPU
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
        except Exception as e:
            self.logger.warning(f"Could not retrieve model info: {e}")
    
    def _get_model_cache_path(self, model_size: str) -> Path:
        """Get the cache path for a specific model size.
        
        Args:
            model_size: Size of the Whisper model
            
        Returns:
            Path to model cache directory
        """
        cache_dir = Path(self.config.cache_dir) / "whisper_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def reload_model(self, model_size: Optional[str] = None) -> None:
        """Reload the Whisper model with optional size change.
        
        Args:
            model_size: Optional new model size to load
        """
        if model_size and model_size != self.config.whisper_model_size:
            self.config.whisper_model_size = model_size
            self.logger.info(f"Changing model size to: {model_size}")
        
        # Clear current model
        self.model = None
        
        # Force garbage collection if on GPU
        if self.device == "cuda" and TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        
        # Reload model
        self._load_whisper_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_size": self.config.whisper_model_size,
            "device": self.device,
            "cache_enabled": self.config.cache_enabled
        }
        
        try:
            info["total_parameters"] = sum(p.numel() for p in self.model.parameters())
            
            if self.device == "cuda" and TORCH_AVAILABLE:
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
                
        except Exception as e:
            self.logger.warning(f"Could not retrieve detailed model info: {e}")
        
        return info
    
    def transcribe_with_speakers(self, audio_file: AudioFile) -> Transcription:
        """Transcribe audio with speaker diarization for multi-speaker content.
        
        Args:
            audio_file: AudioFile object to transcribe
            
        Returns:
            Transcription object with speaker-labeled segments
            
        Raises:
            TranscriptionError: If transcription or diarization fails
        """
        try:
            self.logger.info(f"Starting transcription with speaker diarization for {audio_file.file_path}")
            
            # First, get regular transcription
            transcription = self.transcribe_audio(audio_file)
            
            # Perform speaker diarization
            speaker_segments = self._perform_speaker_diarization(audio_file)
            
            # Assign speakers to transcription segments
            transcription_with_speakers = self._assign_speakers_to_segments(
                transcription, speaker_segments
            )
            
            self.logger.info(f"Speaker diarization completed with {len(set(s.speaker_id for s in transcription_with_speakers.segments if s.speaker_id))} unique speakers")
            
            return transcription_with_speakers
            
        except Exception as e:
            self.logger.error(f"Transcription with speaker diarization failed: {e}")
            # Fallback to regular transcription without speakers
            self.logger.warning("Falling back to transcription without speaker diarization")
            return self.transcribe_audio(audio_file)
    
    def _perform_speaker_diarization(self, audio_file: AudioFile) -> List[Tuple[float, float, str]]:
        """Perform speaker diarization on audio file.
        
        Args:
            audio_file: AudioFile object
            
        Returns:
            List of tuples (start_time, end_time, speaker_id)
        """
        try:
            # Simple speaker diarization using audio energy and spectral features
            # This is a basic implementation - for production, consider using pyannote.audio
            
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_file.file_path, sr=None)
            
            # Parameters for speaker change detection
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Extract features for speaker change detection
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, 
                                      hop_length=hop_length, 
                                      n_fft=frame_length*2)
            
            # Calculate spectral centroid for voice characteristics
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                                hop_length=hop_length)[0]
            
            # Detect speaker changes using MFCC similarity
            speaker_changes = self._detect_speaker_changes(mfcc, hop_length, sr)
            
            # Assign speaker IDs based on voice characteristics
            speaker_segments = self._assign_speaker_ids(speaker_changes, spectral_centroid, 
                                                      hop_length, sr, audio_file.duration)
            
            return speaker_segments
            
        except ImportError:
            self.logger.warning("librosa not available, using simple speaker detection")
            return self._simple_speaker_detection(audio_file)
        except Exception as e:
            self.logger.warning(f"Speaker diarization failed: {e}, using simple detection")
            return self._simple_speaker_detection(audio_file)
    
    def _detect_speaker_changes(self, mfcc: np.ndarray, hop_length: int, sr: int) -> List[float]:
        """Detect speaker changes using MFCC features.
        
        Args:
            mfcc: MFCC feature matrix
            hop_length: Hop length in samples
            sr: Sample rate
            
        Returns:
            List of speaker change timestamps
        """
        # Calculate frame-to-frame distance
        distances = []
        for i in range(1, mfcc.shape[1]):
            # Euclidean distance between consecutive MFCC frames
            dist = np.linalg.norm(mfcc[:, i] - mfcc[:, i-1])
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Smooth the distances
        window_size = 5
        smoothed_distances = np.convolve(distances, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks (potential speaker changes)
        threshold = np.mean(smoothed_distances) + 1.5 * np.std(smoothed_distances)
        
        speaker_changes = [0.0]  # Always start with a speaker change at the beginning
        
        for i, dist in enumerate(smoothed_distances):
            if dist > threshold:
                # Convert frame index to time
                time_stamp = (i + 1) * hop_length / sr
                # Avoid changes too close together (minimum 2 seconds)
                if not speaker_changes or time_stamp - speaker_changes[-1] > 2.0:
                    speaker_changes.append(time_stamp)
        
        return speaker_changes
    
    def _assign_speaker_ids(self, speaker_changes: List[float], spectral_centroid: np.ndarray,
                          hop_length: int, sr: int, total_duration: float) -> List[Tuple[float, float, str]]:
        """Assign speaker IDs based on voice characteristics.
        
        Args:
            speaker_changes: List of speaker change timestamps
            spectral_centroid: Spectral centroid features
            hop_length: Hop length in samples
            sr: Sample rate
            total_duration: Total audio duration
            
        Returns:
            List of speaker segments (start_time, end_time, speaker_id)
        """
        segments = []
        
        # Add final timestamp if not present
        if not speaker_changes or speaker_changes[-1] < total_duration - 1.0:
            speaker_changes.append(total_duration)
        
        # Analyze spectral characteristics for each segment
        segment_features = []
        for i in range(len(speaker_changes) - 1):
            start_time = speaker_changes[i]
            end_time = speaker_changes[i + 1]
            
            # Convert to frame indices
            start_frame = int(start_time * sr / hop_length)
            end_frame = int(end_time * sr / hop_length)
            
            # Calculate average spectral centroid for this segment
            if start_frame < len(spectral_centroid) and end_frame <= len(spectral_centroid):
                avg_centroid = np.mean(spectral_centroid[start_frame:end_frame])
                segment_features.append((start_time, end_time, avg_centroid))
        
        # Cluster segments by spectral centroid (simple 2-means clustering)
        if len(segment_features) > 1:
            centroids = [f[2] for f in segment_features]
            
            # Simple clustering: high vs low spectral centroid
            centroid_threshold = np.median(centroids)
            
            for start_time, end_time, centroid in segment_features:
                if centroid > centroid_threshold:
                    speaker_id = "speaker_1"  # Higher pitch/frequency
                else:
                    speaker_id = "speaker_2"  # Lower pitch/frequency
                
                segments.append((start_time, end_time, speaker_id))
        else:
            # Single speaker
            for start_time, end_time, _ in segment_features:
                segments.append((start_time, end_time, "speaker_1"))
        
        return segments
    
    def _simple_speaker_detection(self, audio_file: AudioFile) -> List[Tuple[float, float, str]]:
        """Simple speaker detection fallback when advanced methods fail.
        
        Args:
            audio_file: AudioFile object
            
        Returns:
            List of speaker segments with single speaker
        """
        # Fallback: assume single speaker for entire duration
        return [(0.0, audio_file.duration, "speaker_1")]
    
    def _assign_speakers_to_segments(self, transcription: Transcription, 
                                   speaker_segments: List[Tuple[float, float, str]]) -> Transcription:
        """Assign speaker IDs to transcription segments.
        
        Args:
            transcription: Original transcription without speakers
            speaker_segments: List of speaker segments (start_time, end_time, speaker_id)
            
        Returns:
            Transcription with speaker IDs assigned
        """
        updated_segments = []
        
        for segment in transcription.segments:
            # Find the speaker segment that overlaps most with this transcription segment
            best_speaker = "speaker_1"  # Default
            max_overlap = 0.0
            
            for start_time, end_time, speaker_id in speaker_segments:
                # Calculate overlap
                overlap_start = max(segment.start_time, start_time)
                overlap_end = min(segment.end_time, end_time)
                overlap_duration = max(0.0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker_id
            
            # Create new segment with speaker ID
            updated_segment = TranscriptSegment(
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                speaker_id=best_speaker,
                language=segment.language
            )
            
            updated_segments.append(updated_segment)
        
        # Create new transcription with speaker information
        return Transcription(
            segments=updated_segments,
            language=transcription.language,
            total_duration=transcription.total_duration,
            confidence_avg=transcription.confidence_avg
        )
    
    def get_speaker_statistics(self, transcription: Transcription) -> Dict[str, Any]:
        """Get statistics about speakers in the transcription.
        
        Args:
            transcription: Transcription with speaker information
            
        Returns:
            Dictionary containing speaker statistics
        """
        if not transcription.segments:
            return {"total_speakers": 0}
        
        speaker_stats = {}
        speaker_durations = {}
        speaker_word_counts = {}
        
        for segment in transcription.segments:
            speaker_id = segment.speaker_id or "unknown"
            
            if speaker_id not in speaker_durations:
                speaker_durations[speaker_id] = 0.0
                speaker_word_counts[speaker_id] = 0
            
            speaker_durations[speaker_id] += segment.duration
            speaker_word_counts[speaker_id] += len(segment.text.split())
        
        # Calculate statistics
        total_speakers = len(speaker_durations)
        total_duration = sum(speaker_durations.values())
        
        for speaker_id in speaker_durations:
            speaker_stats[speaker_id] = {
                "duration": speaker_durations[speaker_id],
                "duration_percentage": (speaker_durations[speaker_id] / total_duration * 100) if total_duration > 0 else 0,
                "word_count": speaker_word_counts[speaker_id],
                "words_per_minute": (speaker_word_counts[speaker_id] / (speaker_durations[speaker_id] / 60)) if speaker_durations[speaker_id] > 0 else 0
            }
        
        return {
            "total_speakers": total_speakers,
            "total_duration": total_duration,
            "speaker_details": speaker_stats
        }
    
    def transcribe_audio(self, audio_file: AudioFile) -> Transcription:
        """Transcribe audio file to text with timestamps.
        
        Args:
            audio_file: AudioFile object to transcribe
            
        Returns:
            Transcription object with segments and metadata
            
        Raises:
            TranscriptionError: If transcription fails
        """
        if self.model is None:
            raise TranscriptionError("Whisper model not loaded")
        
        try:
            start_time = time.time()
            self.logger.info(f"Starting transcription of {audio_file.file_path}")
            
            # Validate audio file exists
            if not os.path.exists(audio_file.file_path):
                raise TranscriptionError(f"Audio file not found: {audio_file.file_path}")
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_file.file_path,
                word_timestamps=True,
                verbose=False,
                temperature=0.0,  # Deterministic output
                best_of=1,        # Single pass for speed
                beam_size=1,      # Greedy decoding for speed
                fp16=self.device == "cuda"  # Use FP16 on GPU for speed
            )
            
            # Extract language and confidence
            detected_language = result.get("language", "unknown")
            
            # Convert segments to our format
            segments = []
            total_confidence = 0.0
            
            for segment in result["segments"]:
                # Skip segments with empty or whitespace-only text
                segment_text = segment["text"].strip()
                if not segment_text:
                    continue
                
                # Calculate segment confidence (average of word confidences if available)
                segment_confidence = 0.0
                word_count = 0
                
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        if "probability" in word:
                            segment_confidence += word["probability"]
                            word_count += 1
                    
                    if word_count > 0:
                        segment_confidence /= word_count
                else:
                    # Fallback to segment-level confidence if available
                    segment_confidence = segment.get("avg_logprob", 0.0)
                    # Convert log probability to probability (approximate)
                    segment_confidence = max(0.0, min(1.0, (segment_confidence + 1.0) / 2.0))
                
                transcript_segment = TranscriptSegment(
                    text=segment_text,
                    start_time=segment["start"],
                    end_time=segment["end"],
                    confidence=segment_confidence,
                    language=detected_language
                )
                
                segments.append(transcript_segment)
                total_confidence += segment_confidence
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(segments) if segments else 0.0
            
            # Handle case where no speech was detected
            if not segments:
                self.logger.warning("No speech segments detected in audio")
                # Create a minimal transcription indicating no speech
                segments = [TranscriptSegment(
                    text="[No speech detected]",
                    start_time=0.0,
                    end_time=min(audio_file.duration, 1.0),
                    confidence=0.0,
                    language=detected_language
                )]
                avg_confidence = 0.0
            
            # Create transcription object
            transcription = Transcription(
                segments=segments,
                language=detected_language,
                total_duration=audio_file.duration,
                confidence_avg=avg_confidence
            )
            
            processing_time = time.time() - start_time
            self.logger.info(
                f"Transcription completed in {processing_time:.2f}s: "
                f"{len(segments)} segments, language: {detected_language}, "
                f"avg confidence: {avg_confidence:.3f}"
            )
            
            return transcription
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise TranscriptionError(f"Failed to transcribe audio: {e}")
    
    def detect_language(self, audio_file: AudioFile) -> str:
        """Detect the language of the audio file.
        
        Args:
            audio_file: AudioFile object
            
        Returns:
            Language code (e.g., 'en', 'hi', 'bn')
            
        Raises:
            TranscriptionError: If language detection fails
        """
        if self.model is None:
            raise TranscriptionError("Whisper model not loaded")
        
        if not WHISPER_AVAILABLE:
            raise TranscriptionError("Whisper not available - please install openai-whisper")
        
        try:
            self.logger.info(f"Detecting language for {audio_file.file_path}")
            
            # Validate audio file exists
            if not os.path.exists(audio_file.file_path):
                raise TranscriptionError(f"Audio file not found: {audio_file.file_path}")
            
            # Load audio and detect language
            audio = whisper.load_audio(audio_file.file_path)
            
            # Use only first 30 seconds for language detection (faster)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect the spoken language
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            confidence = probs[detected_language]
            
            self.logger.info(f"Detected language: {detected_language} (confidence: {confidence:.3f})")
            
            # Log top 3 language predictions
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            self.logger.debug(f"Top language predictions: {top_languages}")
            
            return detected_language
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            raise TranscriptionError(f"Failed to detect language: {e}")
    
    def segment_transcription(self, transcription: Transcription) -> List[TranscriptSegment]:
        """Segment transcription into logical chunks for processing.
        
        Args:
            transcription: Transcription object
            
        Returns:
            List of TranscriptSegment objects
            
        Raises:
            TranscriptionError: If segmentation fails
        """
        try:
            self.logger.info(f"Segmenting transcription with {len(transcription.segments)} segments")
            
            if not transcription.segments:
                return []
            
            # Configuration for segmentation
            max_segment_duration = 30.0  # Maximum segment duration in seconds
            min_segment_duration = 2.0   # Minimum segment duration in seconds
            max_words_per_segment = 50   # Maximum words per segment
            
            segmented_chunks = []
            current_chunk_segments = []
            current_chunk_duration = 0.0
            current_chunk_words = 0
            
            for segment in transcription.segments:
                segment_duration = segment.duration
                segment_words = len(segment.text.split())
                
                # Check if adding this segment would exceed limits
                would_exceed_duration = (current_chunk_duration + segment_duration) > max_segment_duration
                would_exceed_words = (current_chunk_words + segment_words) > max_words_per_segment
                
                # If we have segments and would exceed limits, finalize current chunk
                if current_chunk_segments and (would_exceed_duration or would_exceed_words):
                    # Only create chunk if it meets minimum duration
                    if current_chunk_duration >= min_segment_duration:
                        chunk = self._create_merged_segment(current_chunk_segments)
                        segmented_chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk_segments = [segment]
                    current_chunk_duration = segment_duration
                    current_chunk_words = segment_words
                else:
                    # Add to current chunk
                    current_chunk_segments.append(segment)
                    current_chunk_duration += segment_duration
                    current_chunk_words += segment_words
            
            # Handle remaining segments
            if current_chunk_segments:
                chunk = self._create_merged_segment(current_chunk_segments)
                segmented_chunks.append(chunk)
            
            self.logger.info(f"Segmentation completed: {len(segmented_chunks)} chunks created")
            
            return segmented_chunks
            
        except Exception as e:
            self.logger.error(f"Transcription segmentation failed: {e}")
            raise TranscriptionError(f"Failed to segment transcription: {e}")
    
    def _create_merged_segment(self, segments: List[TranscriptSegment]) -> TranscriptSegment:
        """Create a merged segment from multiple segments.
        
        Args:
            segments: List of segments to merge
            
        Returns:
            Merged TranscriptSegment
        """
        if not segments:
            raise ValueError("Cannot merge empty segment list")
        
        if len(segments) == 1:
            return segments[0]
        
        # Merge text with proper spacing
        merged_text = " ".join(segment.text.strip() for segment in segments)
        
        # Use timing from first and last segments
        start_time = segments[0].start_time
        end_time = segments[-1].end_time
        
        # Calculate weighted average confidence
        total_duration = 0.0
        weighted_confidence = 0.0
        
        for segment in segments:
            duration = segment.duration
            total_duration += duration
            weighted_confidence += segment.confidence * duration
        
        avg_confidence = weighted_confidence / total_duration if total_duration > 0 else 0.0
        
        # Use language from first segment (should be consistent)
        language = segments[0].language
        
        # Use speaker_id from first segment if available
        speaker_id = segments[0].speaker_id
        
        return TranscriptSegment(
            text=merged_text,
            start_time=start_time,
            end_time=end_time,
            confidence=avg_confidence,
            speaker_id=speaker_id,
            language=language
        )