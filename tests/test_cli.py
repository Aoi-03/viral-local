"""
Tests for command-line interface functionality.
"""

import pytest
from unittest.mock import Mock, patch
from viral_local.main import _is_valid_youtube_url, ProgressTracker


class TestCLI:
    """Test CLI functionality."""
    
    def test_youtube_url_validation_valid_urls(self):
        """Test validation of valid YouTube URLs."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/v/dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        
        for url in valid_urls:
            assert _is_valid_youtube_url(url), f"URL should be valid: {url}"
    
    def test_youtube_url_validation_invalid_urls(self):
        """Test validation of invalid YouTube URLs."""
        invalid_urls = [
            "https://vimeo.com/123456",
            "https://example.com/video",
            "not_a_url",
            "",
            "https://youtube.com/",
            "https://youtube.com/watch",
            "ftp://youtube.com/watch?v=123",
        ]
        
        for url in invalid_urls:
            assert not _is_valid_youtube_url(url), f"URL should be invalid: {url}"
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker()
        
        assert tracker.start_time is None
        assert tracker.stage_times == {}
        assert tracker.current_stage is None
        assert "download" in tracker.stage_estimates
        assert "transcription" in tracker.stage_estimates
    
    def test_progress_tracker_stage_management(self):
        """Test progress tracker stage management."""
        tracker = ProgressTracker()
        
        # Start a stage
        tracker.start_stage("download", "Downloading video")
        assert tracker.current_stage == "download"
        assert "download" in tracker.stage_times
        
        # Complete the stage
        tracker.complete_stage("Download completed")
        # Current stage should still be set until next stage starts
        assert tracker.current_stage == "download"
    
    @patch('viral_local.main.Console')
    def test_progress_tracker_with_rich(self, mock_console):
        """Test progress tracker with rich console."""
        # Mock rich being available
        with patch('viral_local.main.RICH_AVAILABLE', True):
            tracker = ProgressTracker()
            tracker.start_processing("https://youtube.com/watch?v=test", "hi")
            
            # Should have created a console instance
            assert tracker.console is not None
    
    def test_progress_tracker_without_rich(self):
        """Test progress tracker fallback without rich."""
        with patch('viral_local.main.RICH_AVAILABLE', False):
            tracker = ProgressTracker()
            
            # Should work without rich
            tracker.start_processing("https://youtube.com/watch?v=test", "hi")
            tracker.start_stage("download", "Downloading video")
            tracker.update_progress("Processing...")
            tracker.complete_stage("Done")
            
            # Should not crash
            assert tracker.console is None


if __name__ == "__main__":
    pytest.main([__file__])