#!/usr/bin/env python3
"""
Offline system test for Viral-Local.
Tests the system with fallback mechanisms when API quota is exceeded.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viral_local.main import ViralLocalPipeline, ProgressTracker

class OfflineTestProgressTracker(ProgressTracker):
    """Progress tracker for offline testing."""
    
    def start_processing(self, url: str, target_language: str):
        print(f"🎬 Starting OFFLINE processing: {url}")
        print(f"🌍 Target language: {target_language}")
        print("📝 Note: Using fallback mechanisms due to API quota limits")
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
        if "quota" in str(error).lower() or "429" in str(error):
            print(f"⚠️  API quota exceeded in {stage} - using fallback")
        else:
            print(f"❌ Error in {stage}: {str(error)}")
    
    def show_completion(self, result_path: str):
        total_time = time.time() - self.start_time if self.start_time else 0
        print(f"🎉 OFFLINE processing completed in {total_time:.1f} seconds!")
        print(f"📁 Output file: {result_path}")
        print("📝 Note: This video uses fallback translation and TTS due to API limits")

def test_offline_system():
    """Test the system with fallback mechanisms."""
    
    # Check if config exists
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("❌ config.yaml not found. Please copy config.yaml.example and add your API keys.")
        return False
    
    try:
        print("🔧 Testing with fallback mechanisms enabled")
        
        # Test with a very short video
        test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - 19 seconds
        target_language = "hi"  # Hindi
        
        print(f"🧪 Testing with video: {test_url}")
        print(f"🎯 Target language: {target_language}")
        print("📝 Expected: API quota errors will trigger fallback mechanisms")
        print("-" * 50)
        
        # Initialize pipeline
        progress_tracker = OfflineTestProgressTracker()
        pipeline = ViralLocalPipeline(config_path, progress_tracker)
        
        # Process video
        result = pipeline.process_video(test_url, target_language)
        
        if result.success:
            print("✅ Offline system test PASSED!")
            print(f"📁 Output: {result.data.file_path if hasattr(result.data, 'file_path') else result.data}")
            
            # Check if output file exists
            output_path = result.data.file_path if hasattr(result.data, 'file_path') else result.data
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"📊 File size: {file_size:.2f} MB")
                print("🔊 Audio: Generated using fallback TTS (silence/basic synthesis)")
                print("🌐 Translation: Basic fallback translation applied")
            
            return True
        else:
            print("❌ Offline system test FAILED!")
            print(f"🔍 Error: {result.error_message}")
            if result.error_code:
                print(f"🔍 Error code: {result.error_code}")
            return False
            
    except Exception as e:
        print(f"❌ Offline system test FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🚀 Viral-Local Offline System Test")
    print("=" * 50)
    print("This test validates that the system works with fallback mechanisms")
    print("when API quotas are exceeded or services are unavailable.")
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    
    print(f"🐍 Python version: {sys.version}")
    
    # Check dependencies
    try:
        import torch
        import whisper
        import moviepy
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return
    
    # Run offline system test
    success = test_offline_system()
    
    if success:
        print("\n🎉 Offline system is working!")
        print("📝 The system can handle API quota limits gracefully")
        print("🔄 Fallback mechanisms ensure continuous operation")
        print("🌐 Start web interface: python run_web_app.py")
    else:
        print("\n🔧 System needs attention. Check the errors above.")
        print("💡 Try waiting for API quota to reset or check your API key")

if __name__ == "__main__":
    main()