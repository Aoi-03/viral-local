"""
Streamlit web interface for Viral-Local video localization.

This module provides a user-friendly web interface for processing YouTube videos
through the Viral-Local localization pipeline.
"""

import streamlit as st
import time
import tempfile
import os
from pathlib import Path
from typing import Optional
import threading
import queue

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viral_local.main import ViralLocalPipeline, ProgressTracker
from viral_local.config import ConfigManager
from viral_local.utils import ViralLocalError


class StreamlitProgressTracker(ProgressTracker):
    """Progress tracker adapted for Streamlit interface."""
    
    def __init__(self):
        super().__init__()
        self.progress_bar = None
        self.status_text = None
        self.current_progress = 0
        self.total_stages = 5  # download, transcription, translation, speech_generation, video_assembly
        
    def start_processing(self, url: str, target_language: str):
        """Start processing with Streamlit UI elements."""
        st.success(f"Processing video: {url}")
        st.info(f"Target language: {self._get_language_name(target_language)}")
        
        # Create progress elements
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
        super().start_processing(url, target_language)
    
    def start_stage(self, stage_name: str, description: str):
        """Start a new processing stage with Streamlit updates."""
        super().start_stage(stage_name, description)
        
        # Update progress
        stage_mapping = {
            "download": 0,
            "transcription": 1,
            "translation": 2,
            "speech_generation": 3,
            "video_assembly": 4
        }
        
        if stage_name in stage_mapping:
            self.current_progress = stage_mapping[stage_name]
            progress_value = self.current_progress / self.total_stages
            if self.progress_bar:
                self.progress_bar.progress(progress_value)
        
        if self.status_text:
            self.status_text.text(f"Processing: {description}")
    
    def update_progress(self, message: str, detail: str = ""):
        """Update progress within current stage."""
        if self.status_text:
            display_message = f"Processing: {message}"
            if detail:
                display_message += f": {detail}"
            self.status_text.text(display_message)
    
    def complete_stage(self, message: str = ""):
        """Complete the current stage."""
        super().complete_stage(message)
        
        # Update progress bar
        progress_value = (self.current_progress + 1) / self.total_stages
        if self.progress_bar:
            self.progress_bar.progress(min(progress_value, 1.0))
        
        if self.status_text and message:
            self.status_text.text(f"Completed: {message}")
    
    def show_error(self, error: Exception, stage: str = ""):
        """Display error in Streamlit."""
        if isinstance(error, ViralLocalError):
            st.error(f"Error in {stage}: {error.message}")
            if hasattr(error, 'details') and error.details:
                with st.expander("Error Details"):
                    st.json(error.details)
        else:
            st.error(f"Unexpected error in {stage}: {str(error)}")
    
    def show_completion(self, result_path: str):
        """Show completion in Streamlit."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        
        if self.status_text:
            self.status_text.text("Processing completed successfully!")
        
        st.success(f"Video processing completed in {total_time:.1f} seconds!")
        
        # Provide download link
        if os.path.exists(result_path):
            with open(result_path, "rb") as file:
                st.download_button(
                    label="Download Localized Video",
                    data=file.read(),
                    file_name=os.path.basename(result_path),
                    mime="video/mp4"
                )
    
    def _get_language_name(self, lang_code: str) -> str:
        """Get full language name from code."""
        language_names = {
            "hi": "Hindi",
            "bn": "Bengali",
            "ta": "Tamil"
        }
        return language_names.get(lang_code, lang_code)


def create_config_form():
    """Create a form for configuration settings."""
    st.sidebar.header("Configuration")
    
    with st.sidebar.expander("API Settings", expanded=False):
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Required for content analysis and translation"
        )
        
        groq_key = st.text_input(
            "Groq API Key (Optional)",
            type="password",
            help="Alternative API for faster processing"
        )
    
    with st.sidebar.expander("Processing Settings", expanded=False):
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        audio_quality = st.selectbox(
            "Audio Quality",
            ["low", "medium", "high"],
            index=2,
            help="Higher quality produces better results but takes longer"
        )
        
        enable_gpu = st.checkbox(
            "Enable GPU Acceleration",
            value=True,
            help="Use GPU for faster processing if available"
        )
    
    return {
        "gemini_api_key": gemini_key,
        "groq_api_key": groq_key if groq_key else None,
        "whisper_model_size": whisper_model,
        "target_audio_quality": audio_quality,
        "enable_gpu": enable_gpu
    }


def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format."""
    import re
    
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/v/[\w-]+',
    ]
    
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Viral-Local",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("Viral-Local")
    st.markdown("### Automated Video Localization for Indian Languages")
    
    # Description
    st.markdown("""
    Transform your YouTube videos into multiple Indian languages with AI-powered dubbing!
    
    **Features:**
    - Intelligent viral segment detection
    - Natural text-to-speech generation
    - Audio-video synchronization
    - Support for Hindi, Bengali, and Tamil
    """)
    
    # Configuration sidebar
    config_data = create_config_form()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Video Processing")
        
        # URL input
        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://youtube.com/watch?v=...",
            help="Paste the YouTube video URL you want to localize"
        )
        
        # Language selection
        target_language = st.selectbox(
            "Target Language",
            options=["hi", "bn", "ta"],
            format_func=lambda x: {
                "hi": "Hindi",
                "bn": "Bengali", 
                "ta": "Tamil"
            }[x],
            help="Select the language for localization"
        )
        
        # Process button
        process_button = st.button(
            "Start Processing",
            type="primary",
            disabled=not youtube_url or not config_data.get("gemini_api_key")
        )
        
        # Validation messages
        if youtube_url and not validate_youtube_url(youtube_url):
            st.error("Please enter a valid YouTube URL")
        
        if not config_data.get("gemini_api_key"):
            st.warning("Please configure your Gemini API key in the sidebar")
    
    with col2:
        st.header("Information")
        
        st.info("""
        **Processing Steps:**
        1. Download video and extract audio
        2. Transcribe speech with Whisper
        3. Analyze viral segments with AI
        4. Translate to target language
        5. Generate localized speech
        6. Assemble final video
        """)
        
        st.warning("""
        **Requirements:**
        - Valid YouTube URL
        - Gemini API key
        - Video duration < 30 minutes
        """)
    
    # Processing logic
    if process_button and youtube_url and config_data.get("gemini_api_key"):
        try:
            # Create temporary config
            temp_config = {
                **config_data,
                "supported_languages": ["hi", "bn", "ta"],
                "default_target_language": target_language,
                "temp_dir": "temp",
                "output_dir": "output",
                "cache_dir": "cache",
                "log_level": "INFO",
                "enable_file_logging": False,  # Disable file logging in web mode
                "max_video_duration": 1800,
                "video_output_format": "mp4"
            }
            
            # Save temporary config
            import yaml
            config_path = "temp_web_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(temp_config, f)
            
            # Initialize pipeline with Streamlit progress tracker
            progress_tracker = StreamlitProgressTracker()
            pipeline = ViralLocalPipeline(config_path, progress_tracker)
            
            # Process video
            with st.spinner("Initializing pipeline..."):
                result = pipeline.process_video(youtube_url, target_language)
            
            # Clean up temporary config
            if os.path.exists(config_path):
                os.remove(config_path)
            
            if not result.success:
                st.error(f"Processing failed: {result.error_message}")
                if result.error_code:
                    st.code(f"Error Code: {result.error_code}")
        
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built by Aoi | Powered by ARIES</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()