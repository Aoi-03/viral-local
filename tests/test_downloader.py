"""
Tests for DownloaderService functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from viral_local.services.downloader import DownloaderService
from viral_local.config import SystemConfig
from viral_local.models import VideoFile, AudioFile, VideoMetadata
from viral_local.utils import ValidationError, DownloadError, ProcessingError


class TestDownloaderService:
    """Test DownloaderService functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SystemConfig(
            gemini_api_key="test_key",
            max_video_duration=1800,
            temp_dir="test_temp",
            audio_sample_rate=22050
        )
    
    @pytest.fixture
    def downloader(self, config):
        """Create DownloaderService instance."""
        return DownloaderService(config)
    
    def test_validate_url_valid_standard(self, downloader):
        """Test URL validation with standard YouTube URL."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmRdnEQy6nuLMHjMZOz59Oq8VGLrG"
        ]
        
        for url in valid_urls:
            assert downloader.validate_url(url) is True
    
    def test_validate_url_valid_shortened(self, downloader):
        """Test URL validation with shortened YouTube URLs."""
        valid_urls = [
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=10"
        ]
        
        for url in valid_urls:
            assert downloader.validate_url(url) is True
    
    def test_validate_url_valid_mobile(self, downloader):
        """Test URL validation with mobile YouTube URLs."""
        valid_urls = [
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            assert downloader.validate_url(url) is True
    
    def test_validate_url_valid_playlist(self, downloader):
        """Test URL validation with playlist URLs."""
        valid_urls = [
            "https://www.youtube.com/playlist?list=PLrAXtmRdnEQy6nuLMHjMZOz59Oq8VGLrG"
        ]
        
        for url in valid_urls:
            assert downloader.validate_url(url) is True
    
    def test_validate_url_invalid(self, downloader):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "",
            None,
            "not_a_url",
            "https://vimeo.com/123456",
            "https://example.com",
            "youtube.com/watch?v=dQw4w9WgXcQ",  # Missing protocol
            "https://youtube.com/watch",  # Missing video ID
        ]
        
        for url in invalid_urls:
            assert downloader.validate_url(url) is False
    
    @patch('viral_local.services.downloader.yt_dlp.YoutubeDL')
    def test_get_video_metadata_success(self, mock_ydl_class, downloader):
        """Test successful metadata extraction."""
        # Mock yt-dlp response
        mock_info = {
            'title': 'Test Video',
            'description': 'Test description',
            'duration': 120,
            'upload_date': '20231201',
            'uploader': 'Test Channel',
            'view_count': 1000,
            'like_count': 50,
            'thumbnail': 'https://example.com/thumb.jpg',
            'tags': ['test', 'video']
        }
        
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        metadata = downloader.get_video_metadata(url)
        
        assert isinstance(metadata, VideoMetadata)
        assert metadata.title == 'Test Video'
        assert metadata.duration == 120
        assert metadata.uploader == 'Test Channel'
        assert metadata.view_count == 1000
        assert 'test' in metadata.tags
    
    def test_get_video_metadata_invalid_url(self, downloader):
        """Test metadata extraction with invalid URL."""
        with pytest.raises(ValidationError, match="Invalid YouTube URL format"):
            downloader.get_video_metadata("invalid_url")
    
    @patch('viral_local.services.downloader.yt_dlp.YoutubeDL')
    def test_get_video_metadata_duration_exceeded(self, mock_ydl_class, downloader):
        """Test metadata extraction with duration exceeding limit."""
        mock_info = {
            'title': 'Long Video',
            'duration': 3600,  # 1 hour, exceeds 30 min limit
            'uploader': 'Test Channel',
            'view_count': 1000
        }
        
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        with pytest.raises(ValidationError, match="Video duration.*exceeds maximum"):
            downloader.get_video_metadata(url)
    
    @patch('viral_local.services.downloader.yt_dlp.YoutubeDL')
    def test_get_video_metadata_video_unavailable(self, mock_ydl_class, downloader):
        """Test metadata extraction with unavailable video."""
        import yt_dlp
        
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = yt_dlp.DownloadError("Video unavailable")
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        with pytest.raises(DownloadError, match="Video is unavailable or private"):
            downloader.get_video_metadata(url)


class TestDownloaderServiceIntegration:
    """Integration tests for DownloaderService (require network access)."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SystemConfig(
            gemini_api_key="test_key",
            max_video_duration=1800,
            temp_dir="test_temp",
            audio_sample_rate=22050
        )
    
    @pytest.fixture
    def downloader(self, config):
        """Create DownloaderService instance."""
        return DownloaderService(config)
    
    @pytest.mark.integration
    def test_validate_url_real_video(self, downloader):
        """Test URL validation with a real YouTube video (requires network)."""
        # Using a well-known stable video
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert downloader.validate_url(url) is True
    
    @pytest.mark.integration
    @patch('viral_local.services.downloader.yt_dlp.YoutubeDL')
    def test_metadata_extraction_real_video(self, mock_ydl_class, downloader):
        """Test metadata extraction with real video data structure."""
        # Mock realistic YouTube metadata
        mock_info = {
            'id': 'dQw4w9WgXcQ',
            'title': 'Rick Astley - Never Gonna Give You Up (Official Video)',
            'description': 'The official video for "Never Gonna Give You Up" by Rick Astley',
            'duration': 212,
            'upload_date': '20091025',
            'uploader': 'Rick Astley',
            'uploader_id': 'RickAstleyVEVO',
            'view_count': 1000000000,
            'like_count': 10000000,
            'thumbnail': 'https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
            'tags': ['Rick', 'Astley', 'Never', 'Gonna', 'Give', 'You', 'Up'],
            'width': 1920,
            'height': 1080,
            'ext': 'mp4'
        }
        
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        metadata = downloader.get_video_metadata(url)
        
        assert metadata.title == 'Rick Astley - Never Gonna Give You Up (Official Video)'
        assert metadata.duration == 212
        assert metadata.uploader == 'Rick Astley'
        assert metadata.view_count == 1000000000
        assert len(metadata.tags) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])