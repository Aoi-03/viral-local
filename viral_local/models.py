"""
Core data models for the Viral-Local system.

This module defines the primary data structures used throughout the video localization pipeline,
including video files, transcription segments, viral analysis results, and voice configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json


@dataclass
class VideoFile:
    """Represents a video file with metadata and processing information."""
    file_path: str
    duration: float
    resolution: Tuple[int, int]
    format: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate video file data after initialization."""
        if self.duration <= 0:
            raise ValueError("Video duration must be positive")
        if len(self.resolution) != 2 or any(r <= 0 for r in self.resolution):
            raise ValueError("Resolution must be a tuple of two positive integers")
        if not self.file_path:
            raise ValueError("File path cannot be empty")


@dataclass
class AudioFile:
    """Represents an audio file extracted from video."""
    file_path: str
    duration: float
    sample_rate: int
    channels: int
    format: str = "wav"
    
    def __post_init__(self):
        """Validate audio file data after initialization."""
        if self.duration <= 0:
            raise ValueError("Audio duration must be positive")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.channels <= 0:
            raise ValueError("Number of channels must be positive")


@dataclass
class VideoMetadata:
    """Metadata extracted from YouTube videos."""
    title: str
    description: str
    duration: float
    upload_date: str
    uploader: str
    view_count: int
    like_count: Optional[int] = None
    thumbnail_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with timing and confidence information."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None
    language: Optional[str] = None
    
    def __post_init__(self):
        """Validate transcript segment data after initialization."""
        if not self.text.strip():
            raise ValueError("Transcript text cannot be empty")
        if self.start_time < 0:
            raise ValueError("Start time cannot be negative")
        if self.end_time <= self.start_time:
            raise ValueError("End time must be greater than start time")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def duration(self) -> float:
        """Calculate the duration of this segment."""
        return self.end_time - self.start_time


@dataclass
class Transcription:
    """Complete transcription of a video with segments and metadata."""
    segments: List[TranscriptSegment]
    language: str
    total_duration: float
    confidence_avg: float
    
    def __post_init__(self):
        """Validate transcription data after initialization."""
        if not self.segments:
            raise ValueError("Transcription must contain at least one segment")
        if self.total_duration <= 0:
            raise ValueError("Total duration must be positive")
        if not 0 <= self.confidence_avg <= 1:
            raise ValueError("Average confidence must be between 0 and 1")


@dataclass
class ViralSegment:
    """A segment identified as having high viral potential."""
    segment: TranscriptSegment
    viral_score: float
    engagement_factors: List[str]
    priority_level: int
    analysis_rationale: str = ""
    
    def __post_init__(self):
        """Validate viral segment data after initialization."""
        if not 0 <= self.viral_score <= 1:
            raise ValueError("Viral score must be between 0 and 1")
        if self.priority_level < 1:
            raise ValueError("Priority level must be at least 1")
        if not self.engagement_factors:
            raise ValueError("Engagement factors cannot be empty")


@dataclass
class TranslatedSegment:
    """A translated segment with quality metrics and cultural adaptations."""
    original_segment: TranscriptSegment
    translated_text: str
    target_language: str
    quality_score: float
    cultural_adaptations: List[str] = field(default_factory=list)
    technical_terms_preserved: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate translated segment data after initialization."""
        if not self.translated_text.strip():
            raise ValueError("Translated text cannot be empty")
        if not self.target_language:
            raise ValueError("Target language cannot be empty")
        if not 0 <= self.quality_score <= 1:
            raise ValueError("Quality score must be between 0 and 1")


@dataclass
class VoiceConfig:
    """Configuration for text-to-speech voice synthesis."""
    language: str
    gender: str
    age_range: str
    speaking_rate: float = 1.0
    pitch_adjustment: float = 0.0
    voice_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate voice configuration after initialization."""
        if not self.language:
            raise ValueError("Language cannot be empty")
        if self.gender not in ["male", "female", "neutral"]:
            raise ValueError("Gender must be 'male', 'female', or 'neutral'")
        if self.age_range not in ["child", "young", "adult", "elderly"]:
            raise ValueError("Age range must be 'child', 'young', 'adult', or 'elderly'")
        if not 0.5 <= self.speaking_rate <= 2.0:
            raise ValueError("Speaking rate must be between 0.5 and 2.0")
        if not -1.0 <= self.pitch_adjustment <= 1.0:
            raise ValueError("Pitch adjustment must be between -1.0 and 1.0")


@dataclass
class TimingData:
    """Timing information for audio-video synchronization."""
    original_segments: List[TranscriptSegment]
    target_segments: List[TranslatedSegment]
    sync_points: List[Tuple[float, float]]  # (original_time, target_time) pairs
    
    def __post_init__(self):
        """Validate timing data after initialization."""
        if len(self.original_segments) != len(self.target_segments):
            raise ValueError("Original and target segments must have the same length")
        if not self.sync_points:
            raise ValueError("Sync points cannot be empty")


@dataclass
class ProcessingResult:
    """Result of a processing operation with status and error information."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingResult":
        """Create ProcessingResult from dictionary."""
        return cls(**data)