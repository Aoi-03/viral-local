"""
Tests for LocalizationEngine.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from viral_local.services.localization import LocalizationEngine
from viral_local.models import Transcription, TranscriptSegment, ViralSegment, TranslatedSegment
from viral_local.config import SystemConfig
from viral_local.utils import TranslationError, APIError


class TestLocalizationEngine:
    """Test LocalizationEngine functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        retur