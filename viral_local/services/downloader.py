"""
DownloaderService for YouTube video processing.

This module handles YouTube video acquisition, validation, and preprocessing
including video download, metadata extraction, and audio extraction.
"""

import re
import time
from typing import Optional, Dict, Any
from pathlib import Path
import yt_dlp
from ..models import VideoFile, AudioFile, VideoMetadata, ProcessingResult
from ..config import SystemConfig
from ..utils import get_logger, ViralLocalError, DownloadError, ValidationError, ProcessingError
from ..utils.logging import LoggerMixin


class DownloaderService(LoggerMixin):
    """Service for downloading and processing YouTube videos."""
    
    def __init__(self, config: SystemConfig):
        """Initialize the downloader service.
        
        Args:
            config: System configuration
        """
        self.config = config
    
    def download_video(self, url: str) -> VideoFile:
        """Download video from YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            VideoFile object with downloaded video information
            
        Raises:
            DownloadError: If download fails
            ValidationError: If URL is invalid
        """
        if not self.validate_url(url):
            raise ValidationError(
                "Invalid YouTube URL format",
                field_name="url", 
                field_value=url
            )
        
        # First get metadata to validate duration
        metadata = self.get_video_metadata(url)
        
        self.logger.info(f"Starting download for: {metadata.title}")
        
        # Create temp directory if it doesn't exist
        temp_dir = Path(self.config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp options for optimal quality and compatibility
        output_template = str(temp_dir / "%(title)s.%(ext)s")
        
        ydl_opts = {
            'format': self._get_optimal_format(),
            'outtmpl': output_template,
            'writeinfojson': True,  # Save metadata
            'writethumbnail': True,  # Save thumbnail
            'ignoreerrors': False,
            'no_warnings': False,
            'extractaudio': False,  # We'll extract audio separately
            'retries': self.config.max_retries,
            'fragment_retries': self.config.max_retries,
            'socket_timeout': 30,
            'progress_hooks': [self._progress_hook],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download the video
                info = ydl.extract_info(url, download=True)
                
                if not info:
                    raise DownloadError(
                        "Failed to download video",
                        url=url,
                        error_code="DOWNLOAD_FAILED"
                    )
                
                # Find the downloaded file
                downloaded_file = self._find_downloaded_file(temp_dir, info)
                
                if not downloaded_file or not downloaded_file.exists():
                    raise DownloadError(
                        "Downloaded file not found",
                        url=url,
                        error_code="FILE_NOT_FOUND"
                    )
                
                # Get video resolution and format
                resolution = self._extract_resolution(info)
                video_format = downloaded_file.suffix[1:]  # Remove the dot
                
                # Create VideoFile object
                video_file = VideoFile(
                    file_path=str(downloaded_file),
                    duration=metadata.duration,
                    resolution=resolution,
                    format=video_format,
                    metadata={
                        'title': metadata.title,
                        'uploader': metadata.uploader,
                        'upload_date': metadata.upload_date,
                        'view_count': metadata.view_count,
                        'url': url,
                        'file_size': downloaded_file.stat().st_size
                    }
                )
                
                self.logger.info(f"Successfully downloaded video: {downloaded_file}")
                return video_file
                
        except yt_dlp.DownloadError as e:
            error_msg = str(e)
            if "HTTP Error 403" in error_msg:
                raise DownloadError(
                    "Access denied - video may be private or geo-blocked",
                    url=url,
                    error_code="ACCESS_DENIED"
                )
            elif "HTTP Error 404" in error_msg:
                raise DownloadError(
                    "Video not found - it may have been deleted",
                    url=url,
                    error_code="VIDEO_NOT_FOUND"
                )
            else:
                raise DownloadError(
                    f"Download failed: {error_msg}",
                    url=url,
                    error_code="DOWNLOAD_ERROR"
                )
        except Exception as e:
            self.logger.error(f"Unexpected error during download: {e}")
            raise DownloadError(
                f"Unexpected error during download: {str(e)}",
                url=url,
                error_code="UNEXPECTED_ERROR"
            )
    
    def _get_optimal_format(self) -> str:
        """Get optimal video format string for yt-dlp.
        
        Returns:
            Format string for yt-dlp
        """
        # Select best quality video with audio, fallback to best available
        # Prefer mp4 format for compatibility
        return "best[ext=mp4]/best[height<=1080]/best"
    
    def _progress_hook(self, d: Dict[str, Any]) -> None:
        """Progress hook for yt-dlp download progress.
        
        Args:
            d: Progress dictionary from yt-dlp
        """
        if d['status'] == 'downloading':
            if 'total_bytes' in d and d['total_bytes']:
                percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                self.logger.info(f"Download progress: {percent:.1f}%")
            elif '_percent_str' in d:
                self.logger.info(f"Download progress: {d['_percent_str']}")
        elif d['status'] == 'finished':
            self.logger.info(f"Download completed: {d['filename']}")
        elif d['status'] == 'error':
            self.logger.error(f"Download error: {d.get('error', 'Unknown error')}")
    
    def _find_downloaded_file(self, temp_dir: Path, info: Dict[str, Any]) -> Optional[Path]:
        """Find the downloaded video file.
        
        Args:
            temp_dir: Temporary directory where file was downloaded
            info: Video info dictionary from yt-dlp
            
        Returns:
            Path to downloaded file or None if not found
        """
        # Try to get filename from info
        if 'filepath' in info:
            return Path(info['filepath'])
        
        # Search for video files in temp directory
        video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov']
        
        # First try to find by title/filename match
        if 'title' in info:
            title = info['title']
            for file_path in temp_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                    if title in file_path.stem:
                        return file_path
        
        # Fallback: find any video file (for reused downloads)
        for file_path in temp_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                return file_path
        
        return None
    
    def _extract_resolution(self, info: Dict[str, Any]) -> tuple[int, int]:
        """Extract video resolution from info dictionary.
        
        Args:
            info: Video info dictionary from yt-dlp
            
        Returns:
            Tuple of (width, height)
        """
        width = info.get('width', 1920)
        height = info.get('height', 1080)
        
        # Fallback to common resolutions if not available
        if not width or not height:
            width, height = 1920, 1080
        
        return (width, height)
    
    def extract_audio(self, video_file: VideoFile) -> AudioFile:
        """Extract audio track from video file.
        
        Args:
            video_file: VideoFile object
            
        Returns:
            AudioFile object with extracted audio information
            
        Raises:
            ProcessingError: If audio extraction fails
        """
        if not Path(video_file.file_path).exists():
            raise ProcessingError(
                "Video file not found",
                stage="audio_extraction",
                input_data={"file_path": video_file.file_path}
            )
        
        self.logger.info(f"Extracting audio from: {video_file.file_path}")
        
        # Create output path for audio file
        video_path = Path(video_file.file_path)
        audio_path = video_path.parent / f"{video_path.stem}.wav"
        
        try:
            # Use direct ffmpeg extraction since we already have the video file
            extracted_audio = self._extract_audio_with_ffmpeg(video_file, audio_path)
            
            # Validate and get audio properties
            audio_info = self._get_audio_info(extracted_audio)
            
            # Create AudioFile object
            audio_file = AudioFile(
                file_path=str(extracted_audio),
                duration=audio_info['duration'],
                sample_rate=audio_info['sample_rate'],
                channels=audio_info['channels'],
                format='wav'
            )
            
            self.logger.info(f"Successfully extracted audio: {extracted_audio}")
            return audio_file
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise ProcessingError(
                f"Failed to extract audio: {str(e)}",
                stage="audio_extraction",
                input_data={"video_file": video_file.file_path}
            )
    
    def _find_extracted_audio(self, search_dir: Path, base_name: str) -> Optional[Path]:
        """Find extracted audio file.
        
        Args:
            search_dir: Directory to search in
            base_name: Base name of the audio file
            
        Returns:
            Path to audio file or None if not found
        """
        audio_extensions = ['.wav', '.mp3', '.m4a', '.ogg']
        
        for ext in audio_extensions:
            audio_path = search_dir / f"{base_name}{ext}"
            if audio_path.exists():
                return audio_path
        
        return None
    
    def _extract_audio_with_ffmpeg(self, video_file: VideoFile, output_path: Path) -> Path:
        """Extract audio using ffmpeg directly as fallback.
        
        Args:
            video_file: VideoFile object
            output_path: Output path for audio file
            
        Returns:
            Path to extracted audio file
        """
        import subprocess
        
        # FFmpeg command for audio extraction optimized for Whisper
        cmd = [
            'ffmpeg',
            '-i', video_file.file_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', str(self.config.audio_sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise ProcessingError(
                    f"FFmpeg failed: {result.stderr}",
                    stage="audio_extraction"
                )
            
            return output_path
            
        except subprocess.TimeoutExpired:
            raise ProcessingError(
                "Audio extraction timed out",
                stage="audio_extraction"
            )
        except FileNotFoundError:
            raise ProcessingError(
                "FFmpeg not found. Please install FFmpeg.",
                stage="audio_extraction"
            )
    
    def _get_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        """Get audio file information.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            import librosa
            
            # Load audio file to get properties
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = len(y) / sr
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if y.ndim == 1 else y.shape[0]
            }
            
        except ImportError:
            # Fallback using subprocess if librosa not available
            return self._get_audio_info_with_ffprobe(audio_path)
    
    def _get_audio_info_with_ffprobe(self, audio_path: Path) -> Dict[str, Any]:
        """Get audio info using ffprobe as fallback.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        import subprocess
        import json
        
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Fallback to basic info
                return {
                    'duration': 0.0,
                    'sample_rate': self.config.audio_sample_rate,
                    'channels': 1
                }
            
            info = json.loads(result.stdout)
            audio_stream = None
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if audio_stream:
                return {
                    'duration': float(audio_stream.get('duration', 0)),
                    'sample_rate': int(audio_stream.get('sample_rate', self.config.audio_sample_rate)),
                    'channels': int(audio_stream.get('channels', 1))
                }
            else:
                return {
                    'duration': float(info.get('format', {}).get('duration', 0)),
                    'sample_rate': self.config.audio_sample_rate,
                    'channels': 1
                }
                
        except (subprocess.SubprocessError, json.JSONDecodeError, ValueError):
            # Final fallback
            return {
                'duration': 0.0,
                'sample_rate': self.config.audio_sample_rate,
                'channels': 1
            }
    
    def validate_url(self, url: str) -> bool:
        """Validate YouTube URL format and accessibility.
        
        Args:
            url: YouTube video URL
            
        Returns:
            True if URL is valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
        
        # YouTube URL patterns
        youtube_patterns = [
            r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',  # Standard format
            r'^https?://(?:www\.)?youtube\.com/watch\?.*v=[\w-]+',  # With additional params
            r'^https?://youtu\.be/[\w-]+',  # Shortened format
            r'^https?://(?:www\.)?youtube\.com/playlist\?list=[\w-]+',  # Playlist format
            r'^https?://(?:m\.)?youtube\.com/watch\?v=[\w-]+',  # Mobile format
        ]
        
        # Check if URL matches any YouTube pattern
        for pattern in youtube_patterns:
            if re.match(pattern, url.strip()):
                return True
        
        return False
    
    def get_video_metadata(self, url: str) -> VideoMetadata:
        """Extract metadata from YouTube video.
        
        Args:
            url: YouTube video URL
            
        Returns:
            VideoMetadata object with video information
            
        Raises:
            ValidationError: If URL is invalid
            DownloadError: If metadata extraction fails
        """
        if not self.validate_url(url):
            raise ValidationError(
                "Invalid YouTube URL format",
                field_name="url",
                field_value=url
            )
        
        self.logger.info(f"Extracting metadata for URL: {url}")
        
        # Configure yt-dlp for metadata extraction only
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,  # Only extract metadata, don't download
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video information
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    raise DownloadError(
                        "Failed to extract video information",
                        url=url,
                        error_code="METADATA_EXTRACTION_FAILED"
                    )
                
                # Validate video duration against limit
                duration = info.get('duration', 0)
                if duration > self.config.max_video_duration:
                    raise ValidationError(
                        f"Video duration ({duration}s) exceeds maximum allowed duration ({self.config.max_video_duration}s)",
                        field_name="duration",
                        field_value=duration
                    )
                
                # Create VideoMetadata object
                metadata = VideoMetadata(
                    title=info.get('title', 'Unknown Title'),
                    description=info.get('description', ''),
                    duration=duration,
                    upload_date=info.get('upload_date', ''),
                    uploader=info.get('uploader', 'Unknown'),
                    view_count=info.get('view_count', 0),
                    like_count=info.get('like_count'),
                    thumbnail_url=info.get('thumbnail'),
                    tags=info.get('tags', [])
                )
                
                self.logger.info(f"Successfully extracted metadata: {metadata.title} ({metadata.duration}s)")
                return metadata
                
        except ValidationError:
            # Re-raise validation errors as they are expected
            raise
        except yt_dlp.DownloadError as e:
            error_msg = str(e)
            if "Video unavailable" in error_msg:
                raise DownloadError(
                    "Video is unavailable or private",
                    url=url,
                    error_code="VIDEO_UNAVAILABLE"
                )
            elif "age-restricted" in error_msg.lower():
                raise DownloadError(
                    "Video is age-restricted and cannot be processed",
                    url=url,
                    error_code="AGE_RESTRICTED"
                )
            else:
                raise DownloadError(
                    f"Failed to extract video metadata: {error_msg}",
                    url=url,
                    error_code="METADATA_EXTRACTION_ERROR"
                )
        except Exception as e:
            self.logger.error(f"Unexpected error during metadata extraction: {e}")
            raise DownloadError(
                f"Unexpected error during metadata extraction: {str(e)}",
                url=url,
                error_code="UNEXPECTED_ERROR"
            )