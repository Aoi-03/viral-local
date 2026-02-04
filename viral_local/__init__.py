"""
Viral-Local: Automated Video Localization Pipeline

A Python-based system for transforming YouTube content into multi-language dubbed videos
using AI-powered transcription, translation, and text-to-speech technologies.
"""

__version__ = "0.1.0"
__author__ = "Viral-Local Team"

from .models import VideoFile, TranscriptSegment, ViralSegment, TranslatedSegment, VoiceConfig
from .config import SystemConfig, ConfigManager
from .services import DownloaderService, TranscriberService, LocalizationEngine, DubbingStudio

__all__ = [
    "VideoFile",
    "TranscriptSegment", 
    "ViralSegment",
    "TranslatedSegment",
    "VoiceConfig",
    "SystemConfig",
    "ConfigManager",
    "DownloaderService",
    "TranscriberService",
    "LocalizationEngine",
    "DubbingStudio",
]