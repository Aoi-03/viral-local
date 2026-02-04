#!/usr/bin/env python3
"""
Advanced usage example for Viral-Local video localization.

This script demonstrates advanced features including custom configuration,
batch processing, and detailed progress monitoring.
"""

import sys
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path so we can import viral_local
sys.path.insert(0, str(Path(__file__).parent.parent))

from viral_local.main import ViralLocalPipeline, ProgressTracker
from viral_local.config import ConfigManager, SystemConfig
from viral_local.utils import get_logger


class DetailedProgressTracker(ProgressTracker):
    """Enhanced progress tracker with detailed logging."""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.metrics = {}
    
    def start_processing(self, url: str, target_language: str):
        super().start_processing(url, target_language)
        self.metrics['start_time'] = self.start_time
        self.metrics['url'] = url
        self.metrics['target_language'] = target_language
    
    def start_stage(self, stage_name: str, description: str):
        super().start_stage(stage_name, description)
        self.metrics[f'{stage_name}_start'] = self.stage_times[stage_name]
        print(f"🔄 [{stage_name.upper()}] {description}")
    
    def complete_stage(self, message: str = ""):
        super().complete_stage(message)
        if self.current_stage:
            elapsed = self.stage_times[self.current_stage]
            self.metrics[f'{self.current_stage}_duration'] = elapsed
            print(f"✅ [{self.current_stage.upper()}] Completed in {elapsed:.1f}s")
    
    def show_completion(self, result_path: str):
        super().show_completion(result_path)
        self.metrics['total_duration'] = self.start_time
        self.metrics['output_file'] = result_path
        
        # Print detailed metrics
        print("\n📊 Processing Metrics:")
        print("-" * 40)
        for key, value in self.metrics.items():
            if key.endswith('_duration'):
                stage = key.replace('_duration', '')
                print(f"  {stage.title()}: {value:.2f}s")


def create_custom_config() -> str:
    """Create a custom configuration file for advanced usage."""
    config_data = {
        # API Configuration
        'gemini_api_key': os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here'),
        'groq_api_key': os.getenv('GROQ_API_KEY'),  # Optional
        
        # Model Settings - Use larger model for better quality
        'whisper_model_size': 'small',  # Better accuracy than base
        'tts_engine': 'edge-tts',
        
        # Processing Limits
        'max_video_duration': 1800,  # 30 minutes
        'max_concurrent_requests': 2,  # Conservative for stability
        
        # Quality Settings - High quality for advanced usage
        'target_audio_quality': 'high',
        'video_output_format': 'mp4',
        'audio_sample_rate': 22050,
        
        # Language Support
        'supported_languages': ['hi', 'bn', 'ta'],
        'default_target_language': 'hi',
        
        # Directory Settings
        'temp_dir': 'temp_advanced',
        'output_dir': 'output_advanced',
        'cache_dir': 'cache_advanced',
        
        # Logging Configuration - Detailed logging
        'log_level': 'DEBUG',
        'log_file': 'viral_local_advanced.log',
        'enable_file_logging': True,
        
        # Performance Settings
        'enable_gpu': True,
        'cache_enabled': True,
        'cache_max_size_mb': 2000,  # Larger cache
        
        # Retry Settings - More aggressive retries
        'max_retries': 5,
        'retry_delay': 2.0,
        'exponential_backoff': True,
        
        # Rate Limiting
        'api_requests_per_minute': 30,  # Conservative rate limiting
        'api_requests_per_hour': 500,
    }
    
    config_path = Path(__file__).parent / "advanced_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    return str(config_path)


def process_single_video(pipeline: ViralLocalPipeline, url: str, target_lang: str) -> Dict[str, Any]:
    """Process a single video with detailed result tracking."""
    print(f"\n🎬 Processing: {url}")
    print(f"🌍 Target Language: {target_lang}")
    
    result = pipeline.process_video(url, target_lang)
    
    return {
        'url': url,
        'target_language': target_lang,
        'success': result.success,
        'output_file': result.data.file_path if result.success else None,
        'error_message': result.error_message if not result.success else None,
        'error_code': result.error_code if not result.success else None
    }


def batch_process_videos(pipeline: ViralLocalPipeline, video_configs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Process multiple videos in batch."""
    print(f"\n🔄 Starting batch processing of {len(video_configs)} videos...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, config in enumerate(video_configs, 1):
        print(f"\n📹 Processing video {i}/{len(video_configs)}")
        
        try:
            result = process_single_video(
                pipeline, 
                config['url'], 
                config['target_language']
            )
            results.append(result)
            
            if result['success']:
                successful += 1
                print(f"✅ Success: {result['output_file']}")
            else:
                failed += 1
                print(f"❌ Failed: {result['error_message']}")
        
        except Exception as e:
            failed += 1
            error_result = {
                'url': config['url'],
                'target_language': config['target_language'],
                'success': False,
                'output_file': None,
                'error_message': str(e),
                'error_code': 'BATCH_PROCESSING_ERROR'
            }
            results.append(error_result)
            print(f"💥 Exception: {e}")
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {(successful / len(video_configs) * 100):.1f}%")
    
    return results


def demonstrate_viral_analysis(pipeline: ViralLocalPipeline, url: str):
    """Demonstrate viral segment analysis capabilities."""
    print(f"\n🧠 Demonstrating viral analysis for: {url}")
    
    try:
        # Download and transcribe first
        video_file = pipeline.downloader.download_video(url)
        audio_file = pipeline.downloader.extract_audio(video_file)
        transcription = pipeline.transcriber.transcribe_audio(audio_file)
        
        print(f"📝 Transcription completed: {len(transcription.segments)} segments")
        
        # Analyze viral segments
        viral_segments = pipeline.localizer.analyze_viral_segments(transcription)
        
        print(f"🔥 Found {len(viral_segments)} viral segments:")
        
        for i, segment in enumerate(viral_segments[:3], 1):  # Show top 3
            print(f"\n  🎯 Viral Segment #{i}:")
            print(f"    Score: {segment.viral_score:.3f}")
            print(f"    Time: {segment.segment.start_time:.1f}s - {segment.segment.end_time:.1f}s")
            print(f"    Text: {segment.segment.text[:100]}...")
            print(f"    Factors: {', '.join(segment.engagement_factors[:3])}")
    
    except Exception as e:
        print(f"❌ Viral analysis failed: {e}")


def main():
    """Advanced usage demonstration."""
    print("🚀 Viral-Local Advanced Usage Example")
    print("=" * 60)
    
    try:
        # Create custom configuration
        print("⚙️ Creating custom configuration...")
        config_path = create_custom_config()
        print(f"📄 Configuration saved to: {config_path}")
        
        # Initialize pipeline with custom config and detailed progress tracking
        print("\n🔧 Initializing advanced pipeline...")
        progress_tracker = DetailedProgressTracker()
        pipeline = ViralLocalPipeline(config_path, progress_tracker)
        
        # Validate setup
        print("\n🔍 Validating system setup...")
        validation_results = pipeline.validate_setup()
        
        if not all(validation_results.values()):
            print("❌ System validation failed. Please check configuration.")
            return 1
        
        print("✅ System validation passed!")
        
        # Example 1: Single video processing with detailed tracking
        print("\n" + "="*60)
        print("📹 EXAMPLE 1: Single Video Processing")
        print("="*60)
        
        single_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = process_single_video(pipeline, single_video_url, "hi")
        
        # Example 2: Batch processing
        print("\n" + "="*60)
        print("📚 EXAMPLE 2: Batch Processing")
        print("="*60)
        
        batch_configs = [
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "target_language": "hi"},
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "target_language": "bn"},
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "target_language": "ta"},
        ]
        
        batch_results = batch_process_videos(pipeline, batch_configs)
        
        # Example 3: Viral analysis demonstration
        print("\n" + "="*60)
        print("🧠 EXAMPLE 3: Viral Analysis")
        print("="*60)
        
        demonstrate_viral_analysis(pipeline, single_video_url)
        
        # Save results summary
        results_summary = {
            'single_video_result': result,
            'batch_results': batch_results,
            'configuration_used': config_path
        }
        
        summary_path = Path(__file__).parent / "processing_results.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(results_summary, f, default_flow_style=False, indent=2)
        
        print(f"\n📊 Results summary saved to: {summary_path}")
        print("\n🎉 Advanced usage demonstration completed!")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user.")
        return 130
    
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())