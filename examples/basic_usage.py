#!/usr/bin/env python3
"""
Basic usage example for Viral-Local video localization.

This script demonstrates the simplest way to use Viral-Local to process
a YouTube video and generate a localized version.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import viral_local
sys.path.insert(0, str(Path(__file__).parent.parent))

from viral_local.main import ViralLocalPipeline
from viral_local.config import ConfigManager


def main():
    """Basic usage example."""
    print("🎬 Viral-Local Basic Usage Example")
    print("=" * 50)
    
    # Example YouTube URL (replace with your own)
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll as example
    target_language = "hi"  # Hindi
    
    try:
        # Initialize the pipeline with default configuration
        print("📋 Initializing Viral-Local pipeline...")
        pipeline = ViralLocalPipeline()
        
        # Validate system setup
        print("🔍 Validating system setup...")
        validation_results = pipeline.validate_setup()
        
        if not all(validation_results.values()):
            print("❌ System validation failed:")
            for component, status in validation_results.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {component.replace('_', ' ').title()}")
            print("\n💡 Please check your configuration and API keys.")
            return 1
        
        print("✅ System validation passed!")
        
        # Process the video
        print(f"\n🚀 Processing video: {youtube_url}")
        print(f"🌍 Target language: {target_language}")
        
        result = pipeline.process_video(youtube_url, target_language)
        
        if result.success:
            print(f"\n🎉 Success! Localized video saved to: {result.data.file_path}")
            print(f"📊 Video duration: {result.data.duration:.1f} seconds")
            print(f"📐 Resolution: {result.data.resolution[0]}x{result.data.resolution[1]}")
            return 0
        else:
            print(f"\n❌ Processing failed: {result.error_message}")
            if result.error_code:
                print(f"🔍 Error code: {result.error_code}")
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user.")
        return 130
    
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())