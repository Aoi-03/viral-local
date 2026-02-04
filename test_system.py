#!/usr/bin/env python3
"""
System validation script for Viral-Local.
Tests the complete pipeline with a short video.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viral_local.main import ViralLocalPipeline, ProgressTracker
from viral_local.config import ConfigManager

class TestProgressTracker(ProgressTracker):
    """Simple progress tracker for testing."""
    
    def start_processing(self, url: str, target_language: str):
        print(f"🎬 Starting processing: {url}")
        print(f"🌍 Target language: {target_language}")
        super().start_processing(url, target_language)
    
    def start_stage(self, stage_name: str, description: str):
        print(f"🔄 {stage_name}: {description}")
        super().start_stage(stage_name, description)
    
    def update_progress(self, message: str, detail: str = ""):
        if detail:
            print(f"   📝 {message}: {detail}")
        else:
            print(f"   📝 {message}")
    
    def complete_stage(self, message: str = ""):
        if message:
            print(f"✅ {message}")
        super().complete_stage(message)
    
    def show_error(self, error: Exception, stage: str = ""):
        print(f"❌ Error in {stage}: {str(error)}")
    
    def show_completion(self, result_path: str):
        total_time = time.time() - self.start_time if self.start_time else 0
        print(f"🎉 Processing completed in {total_time:.1f} seconds!")
        print(f"📁 Output file: {result_path}")

def test_system():
    """Test the complete system with a short video."""
    
    # Check if config exists
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("❌ config.yaml not found. Please copy config.yaml.example and add your API keys.")
        return False
    
    try:
        # Load config to check API key
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        if not config.gemini_api_key or config.gemini_api_key == "your_gemini_api_key_here":
            print("❌ Please configure your Gemini API key in config.yaml")
            return False
        
        print("🔧 Configuration loaded successfully")
        
        # Test with a very short video (YouTube's first video)
        test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - 19 seconds
        target_language = "hi"  # Hindi
        
        print(f"🧪 Testing with video: {test_url}")
        print(f"🎯 Target language: {target_language}")
        print("-" * 50)
        
        # Initialize pipeline
        progress_tracker = TestProgressTracker()
        pipeline = ViralLocalPipeline(config_path, progress_tracker)
        
        # Process video
        result = pipeline.process_video(test_url, target_language)
        
        if result.success:
            print("✅ System validation PASSED!")
            print(f"📁 Output: {result.output_file}")
            
            # Check if output file exists
            if os.path.exists(result.output_file):
                file_size = os.path.getsize(result.output_file) / (1024 * 1024)  # MB
                print(f"📊 File size: {file_size:.2f} MB")
            
            return True
        else:
            print("❌ System validation FAILED!")
            print(f"🔍 Error: {result.error_message}")
            if result.error_code:
                print(f"🔍 Error code: {result.error_code}")
            return False
            
    except Exception as e:
        print(f"❌ System validation FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🚀 Viral-Local System Validation")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    
    print(f"🐍 Python version: {sys.version}")
    
    # Check dependencies
    try:
        import torch
        import whisper
        import streamlit
        import moviepy
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return
    
    # Run system test
    success = test_system()
    
    if success:
        print("\n🎉 System is ready for use!")
        print("🌐 Start web interface: python run_web_app.py")
        print("💻 Use CLI: python -m viral_local.main <youtube_url> <language>")
    else:
        print("\n🔧 System needs attention. Check the errors above.")

if __name__ == "__main__":
    main()