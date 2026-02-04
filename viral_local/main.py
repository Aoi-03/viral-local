"""
Main entry point for the Viral-Local application.

This module provides the main pipeline orchestrator and command-line interface
for the video localization system.
"""

import time
import sys
from typing import Optional
from pathlib import Path

from .config import ConfigManager, SystemConfig
from .utils import setup_logging, get_logger, ViralLocalError, handle_error, create_user_friendly_message
from .services import DownloaderService, TranscriberService, LocalizationEngine, DubbingStudio
from .models import ProcessingResult

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback progress indicator
    class SimpleProgress:
        def __init__(self):
            self.current_stage = ""
        
        def update_stage(self, stage_name: str, message: str = ""):
            self.current_stage = stage_name
            print(f"[{stage_name}] {message}")
        
        def update_progress(self, message: str):
            print(f"  → {message}")
        
        def complete_stage(self, message: str = ""):
            print(f"  ✓ {message}")
        
        def error(self, message: str):
            print(f"  ✗ {message}")


class ProgressTracker:
    """Enhanced progress tracking with rich formatting when available."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.start_time = None
        self.stage_times = {}
        self.current_stage = None
        
        # Stage duration estimates (in seconds) based on typical processing times
        self.stage_estimates = {
            "download": 30,
            "transcription": 60,
            "translation": 45,
            "speech_generation": 90,
            "video_assembly": 30
        }
    
    def start_processing(self, url: str, target_language: str):
        """Start the processing with initial information."""
        self.start_time = time.time()
        
        if RICH_AVAILABLE:
            self.console.print(Panel.fit(
                f"[bold blue]Viral-Local Video Localization[/bold blue]\n"
                f"URL: {url}\n"
                f"Target Language: {target_language}",
                title="Processing Started"
            ))
        else:
            print("=" * 60)
            print("Viral-Local Video Localization")
            print(f"URL: {url}")
            print(f"Target Language: {target_language}")
            print("=" * 60)
    
    def start_stage(self, stage_name: str, description: str):
        """Start a new processing stage."""
        if self.current_stage:
            self.complete_stage()
        
        self.current_stage = stage_name
        self.stage_times[stage_name] = time.time()
        
        estimated_time = self.stage_estimates.get(stage_name, 30)
        
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold yellow]Stage: {description}[/bold yellow]")
            self.console.print(f"[dim]Estimated time: {estimated_time}s[/dim]")
        else:
            print(f"\n[{stage_name.upper()}] {description}")
            print(f"  Estimated time: {estimated_time}s")
    
    def update_progress(self, message: str, detail: str = ""):
        """Update progress within current stage."""
        if RICH_AVAILABLE:
            if detail:
                self.console.print(f"  → {message}: [dim]{detail}[/dim]")
            else:
                self.console.print(f"  → {message}")
        else:
            if detail:
                print(f"    → {message}: {detail}")
            else:
                print(f"    → {message}")
    
    def complete_stage(self, message: str = ""):
        """Complete the current stage."""
        if not self.current_stage:
            return
        
        elapsed = time.time() - self.stage_times[self.current_stage]
        
        if RICH_AVAILABLE:
            self.console.print(f"  [green]✓ Completed in {elapsed:.1f}s[/green] {message}")
        else:
            print(f"    ✓ Completed in {elapsed:.1f}s {message}")
    
    def show_error(self, error: Exception, stage: str = ""):
        """Display error information with suggestions."""
        if isinstance(error, ViralLocalError):
            user_message = create_user_friendly_message(error)
        else:
            user_message = f"An unexpected error occurred: {str(error)}"
        
        if RICH_AVAILABLE:
            self.console.print(Panel(
                user_message,
                title="[red]Error[/red]",
                border_style="red"
            ))
        else:
            print("\n" + "=" * 60)
            print("ERROR:")
            print(user_message)
            print("=" * 60)
    
    def show_completion(self, result_path: str):
        """Show completion information."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        if RICH_AVAILABLE:
            self.console.print(Panel.fit(
                f"[bold green]Processing Completed Successfully![/bold green]\n"
                f"Output file: {result_path}\n"
                f"Total time: {total_time:.1f}s",
                title="Success",
                border_style="green"
            ))
        else:
            print("\n" + "=" * 60)
            print("✓ Processing Completed Successfully!")
            print(f"Output file: {result_path}")
            print(f"Total time: {total_time:.1f}s")
            print("=" * 60)
    
    def show_validation_results(self, results: dict):
        """Display system validation results."""
        if RICH_AVAILABLE:
            table = Table(title="System Validation Results")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="bold")
            
            for component, status in results.items():
                status_text = "[green]✓ PASS[/green]" if status else "[red]✗ FAIL[/red]"
                table.add_row(component.replace("_", " ").title(), status_text)
            
            self.console.print(table)
        else:
            print("\nSystem Validation Results:")
            print("-" * 40)
            for component, status in results.items():
                status_str = "✓ PASS" if status else "✗ FAIL"
                print(f"  {component.replace('_', ' ').title()}: {status_str}")
            print("-" * 40)


class ViralLocalPipeline:
    """Main pipeline orchestrator for the Viral-Local system."""
    
    def __init__(self, config_path: Optional[str] = None, progress_tracker: Optional[ProgressTracker] = None):
        """Initialize the pipeline with configuration.
        
        Args:
            config_path: Optional path to configuration file
            progress_tracker: Optional progress tracker for UI updates
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file,
            enable_file_logging=self.config.enable_file_logging
        )
        
        # Progress tracking
        self.progress = progress_tracker or ProgressTracker()
        
        # Initialize services
        self.downloader = DownloaderService(self.config)
        self.transcriber = TranscriberService(self.config)
        self.localizer = LocalizationEngine(self.config)
        self.dubber = DubbingStudio(self.config)
        
        self.logger.info("Viral-Local pipeline initialized successfully")
    
    def process_video(self, youtube_url: str, target_language: str) -> ProcessingResult:
        """Process a YouTube video through the complete localization pipeline.
        
        Args:
            youtube_url: YouTube video URL to process
            target_language: Target language code for localization
            
        Returns:
            ProcessingResult with final video file or error information
        """
        try:
            self.progress.start_processing(youtube_url, target_language)
            self.logger.info(f"Starting video processing: {youtube_url} -> {target_language}")
            
            # Stage 1: Download video
            self.progress.start_stage("download", "Downloading and extracting audio")
            self.logger.info("Stage 1: Downloading video...")
            
            self.progress.update_progress("Validating YouTube URL")
            video_file = self.downloader.download_video(youtube_url)
            
            self.progress.update_progress("Extracting audio track")
            audio_file = self.downloader.extract_audio(video_file)
            
            self.progress.complete_stage(f"Downloaded {video_file.duration:.1f}s video")
            
            # Stage 2: Transcribe audio
            self.progress.start_stage("transcription", "Transcribing audio with Whisper")
            self.logger.info("Stage 2: Transcribing audio...")
            
            self.progress.update_progress("Loading Whisper model")
            transcription = self.transcriber.transcribe_audio(audio_file)
            
            self.progress.complete_stage(f"Transcribed {len(transcription.segments)} segments")
            
            # Stage 3: Analyze and translate
            self.progress.start_stage("translation", "Analyzing content and translating")
            self.logger.info("Stage 3: Analyzing viral segments and translating...")
            
            self.progress.update_progress("Analyzing viral segments")
            viral_segments = self.localizer.analyze_viral_segments(transcription)
            
            self.progress.update_progress(f"Translating to {target_language}")
            translated_segments = self.localizer.translate_content(
                transcription.segments, target_language
            )
            
            self.progress.complete_stage(f"Translated {len(translated_segments)} segments")
            
            # Stage 4: Generate speech and assemble video
            self.progress.start_stage("speech_generation", "Generating localized speech")
            self.logger.info("Stage 4: Generating speech...")
            
            # Voice configuration will be determined based on original audio characteristics
            voice_config = self._determine_voice_config(transcription, target_language)
            self.progress.update_progress(f"Using voice: {voice_config.gender} {voice_config.age_range}")
            
            localized_audio = self.dubber.generate_speech(translated_segments, voice_config)
            self.progress.complete_stage(f"Generated {localized_audio.duration:.1f}s of speech")
            
            # Stage 5: Video assembly
            self.progress.start_stage("video_assembly", "Synchronizing and assembling final video")
            self.logger.info("Stage 5: Assembling video...")
            
            self.progress.update_progress("Synchronizing audio timing")
            timing_data = self._create_timing_data(transcription.segments, translated_segments)
            synchronized_audio = self.dubber.synchronize_audio(localized_audio, timing_data)
            
            self.progress.update_progress("Merging audio with video")
            final_video = self.dubber.merge_audio_video(video_file, synchronized_audio)
            
            self.progress.complete_stage()
            self.progress.show_completion(final_video.file_path)
            
            self.logger.info(f"Video processing completed successfully: {final_video.file_path}")
            
            return ProcessingResult(
                success=True,
                data=final_video,
                processing_time=None  # Will be calculated by performance logging
            )
            
        except ViralLocalError as e:
            self.logger.error(f"Processing failed: {e}")
            self.progress.show_error(e, self.progress.current_stage or "unknown")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                error_code=e.error_code
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during processing: {e}")
            self.progress.show_error(e, self.progress.current_stage or "unknown")
            error_info = handle_error(e, self.logger)
            return ProcessingResult(
                success=False,
                error_message=str(e),
                error_code="UNEXPECTED_ERROR"
            )
    
    def _determine_voice_config(self, transcription, target_language):
        """Determine appropriate voice configuration based on original audio.
        
        Args:
            transcription: Transcription object with speaker information
            target_language: Target language code
            
        Returns:
            VoiceConfig object with appropriate settings
        """
        from .models import VoiceConfig
        
        # Analyze speaker characteristics from transcription
        speaker_stats = self.transcriber.get_speaker_statistics(transcription)
        
        # Default configuration
        gender = "neutral"
        age_range = "adult"
        speaking_rate = 1.0
        pitch_adjustment = 0.0
        
        # Determine gender based on speaker analysis or default to neutral
        if speaker_stats.get("total_speakers", 0) > 0:
            # For simplicity, use first speaker's characteristics
            # In a more advanced system, this would analyze audio features
            primary_speaker = list(speaker_stats.get("speaker_details", {}).keys())[0]
            
            # Basic heuristic: if speaker talks fast, adjust rate
            speaker_info = speaker_stats["speaker_details"][primary_speaker]
            words_per_minute = speaker_info.get("words_per_minute", 150)
            
            if words_per_minute > 180:
                speaking_rate = 1.1  # Slightly faster
            elif words_per_minute < 120:
                speaking_rate = 0.9  # Slightly slower
        
        # Language-specific adjustments
        if target_language == "hi":
            # Hindi speakers often have a slightly different rhythm
            speaking_rate *= 0.95
        elif target_language == "ta":
            # Tamil has different prosodic patterns
            speaking_rate *= 1.05
        
        return VoiceConfig(
            language=target_language,
            gender=gender,
            age_range=age_range,
            speaking_rate=speaking_rate,
            pitch_adjustment=pitch_adjustment
        )
    
    def _create_timing_data(self, original_segments, translated_segments):
        """Create timing data for audio-video synchronization.
        
        Args:
            original_segments: List of original transcript segments
            translated_segments: List of translated segments
            
        Returns:
            TimingData object for synchronization
        """
        from .models import TimingData
        
        # Create sync points based on segment boundaries
        sync_points = []
        
        for orig_seg, trans_seg in zip(original_segments, translated_segments):
            # Add sync point at segment start
            sync_points.append((orig_seg.start_time, orig_seg.start_time))
            
            # Add sync point at segment end if there's a significant duration difference
            orig_duration = orig_seg.end_time - orig_seg.start_time
            if orig_duration > 5.0:  # For segments longer than 5 seconds
                sync_points.append((orig_seg.end_time, orig_seg.end_time))
        
        # Remove duplicate sync points and sort
        sync_points = sorted(list(set(sync_points)))
        
        return TimingData(
            original_segments=original_segments,
            target_segments=translated_segments,
            sync_points=sync_points
        )
    
    def validate_setup(self) -> dict:
        """Validate that the system is properly configured and ready to use.
        
        Returns:
            Dictionary with validation results for different components
        """
        validation_results = {}
        
        # Check API keys
        api_validation = self.config_manager.validate_api_keys()
        validation_results.update(api_validation)
        
        # Check directories
        validation_results["directories"] = all([
            Path(self.config.temp_dir).exists(),
            Path(self.config.output_dir).exists(),
            Path(self.config.cache_dir).exists()
        ])
        
        # Check supported languages
        validation_results["languages"] = len(self.config.supported_languages) > 0
        
        self.logger.info(f"System validation results: {validation_results}")
        return validation_results


def main():
    """Main entry point for command-line usage."""
    import argparse
    import sys
    
    # Create argument parser with better help and descriptions
    parser = argparse.ArgumentParser(
        description="Viral-Local: Automated Video Localization System",
        epilog="""
Examples:
  %(prog)s "https://youtube.com/watch?v=example" --target-lang hi
  %(prog)s "https://youtu.be/example" -t bn --config my_config.yaml
  %(prog)s --validate --config production.yaml
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional arguments
    parser.add_argument(
        "url", 
        nargs="?",
        help="YouTube video URL to process (required unless using --validate)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--target-lang", "-t", 
        default="hi",
        choices=["hi", "bn", "ta"],
        help="Target language code: hi (Hindi), bn (Bengali), ta (Tamil) [default: hi]"
    )
    
    parser.add_argument(
        "--config", "-c", 
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate system setup and configuration, then exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose logging output"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Override output directory for processed videos"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators (useful for scripting)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Viral-Local v1.0.0"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not args.validate and not args.url:
        parser.error("YouTube URL is required unless using --validate")
    
    try:
        # Initialize progress tracker
        progress = None if args.no_progress else ProgressTracker()
        
        # Initialize pipeline
        if progress and not args.no_progress:
            if RICH_AVAILABLE:
                progress.console.print("[dim]Initializing Viral-Local pipeline...[/dim]")
            else:
                print("Initializing Viral-Local pipeline...")
        
        pipeline = ViralLocalPipeline(args.config, progress)
        
        # Override output directory if specified
        if args.output_dir:
            pipeline.config.output_dir = args.output_dir
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set verbose logging if requested
        if args.verbose:
            pipeline.config.log_level = "DEBUG"
            # Re-setup logging with new level
            setup_logging(
                log_level="DEBUG",
                log_file=pipeline.config.log_file,
                enable_file_logging=pipeline.config.enable_file_logging
            )
        
        if args.validate:
            # Validate setup
            if progress:
                if RICH_AVAILABLE:
                    progress.console.print("[bold]Validating system setup...[/bold]")
                else:
                    print("Validating system setup...")
            
            results = pipeline.validate_setup()
            
            if progress:
                progress.show_validation_results(results)
            
            # Check if all validations passed
            all_passed = all(results.values())
            if not all_passed:
                if progress:
                    if RICH_AVAILABLE:
                        progress.console.print("\n[red]Some validations failed. Please check your configuration.[/red]")
                    else:
                        print("\nSome validations failed. Please check your configuration.")
                return 1
            else:
                if progress:
                    if RICH_AVAILABLE:
                        progress.console.print("\n[green]All validations passed! System is ready.[/green]")
                    else:
                        print("\nAll validations passed! System is ready.")
                return 0
        
        # Validate URL format
        if args.url:
            if not _is_valid_youtube_url(args.url):
                if progress:
                    progress.show_error(
                        ValueError("Invalid YouTube URL format. Please provide a valid YouTube URL."),
                        "validation"
                    )
                else:
                    print("Error: Invalid YouTube URL format.")
                return 1
        
        # Process video
        result = pipeline.process_video(args.url, args.target_lang)
        
        if result.success:
            return 0
        else:
            return 1
    
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            Console().print("\n[yellow]Processing interrupted by user.[/yellow]")
        else:
            print("\nProcessing interrupted by user.")
        return 130  # Standard exit code for SIGINT
    
    except Exception as e:
        if RICH_AVAILABLE:
            Console().print(f"\n[red]Application error: {e}[/red]")
        else:
            print(f"\nApplication error: {e}")
        return 1


def _is_valid_youtube_url(url: str) -> bool:
    """Validate YouTube URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL appears to be a valid YouTube URL
    """
    import re
    
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/v/[\w-]+',
    ]
    
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def create_cli_config():
    """Create a sample configuration file for CLI usage."""
    import yaml
    
    sample_config = {
        'gemini_api_key': 'your_gemini_api_key_here',
        'groq_api_key': 'your_groq_api_key_here',  # Optional
        'whisper_model_size': 'base',
        'tts_engine': 'edge-tts',
        'max_video_duration': 1800,  # 30 minutes
        'target_audio_quality': 'high',
        'video_output_format': 'mp4',
        'supported_languages': ['hi', 'bn', 'ta'],
        'default_target_language': 'hi',
        'temp_dir': 'temp',
        'output_dir': 'output',
        'cache_dir': 'cache',
        'log_level': 'INFO',
        'enable_file_logging': True,
        'enable_gpu': True,
        'cache_enabled': True
    }
    
    config_path = Path('config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration created: {config_path}")
    print("Please edit the file and add your API keys before running the application.")


if __name__ == "__main__":
    exit(main())