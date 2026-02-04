#!/usr/bin/env python3
"""
Performance benchmarking utility for Viral-Local.

This script provides comprehensive performance testing and benchmarking
capabilities to measure system performance under different conditions.
"""

import sys
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add the parent directory to the path so we can import viral_local
sys.path.insert(0, str(Path(__file__).parent.parent))

from viral_local.main import ViralLocalPipeline
from viral_local.config import SystemConfig


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    video_url: str
    target_language: str
    video_duration: float
    processing_time: float
    success: bool
    error_message: str = ""
    stage_timings: Dict[str, float] = None
    memory_usage: Dict[str, float] = None
    
    def __post_init__(self):
        if self.stage_timings is None:
            self.stage_timings = {}
        if self.memory_usage is None:
            self.memory_usage = {}


class PerformanceBenchmark:
    """Performance benchmarking utility."""
    
    def __init__(self, config_path: str = None):
        """Initialize benchmark utility."""
        self.pipeline = ViralLocalPipeline(config_path)
        self.results: List[BenchmarkResult] = []
    
    def run_single_benchmark(
        self, 
        test_name: str, 
        video_url: str, 
        target_language: str
    ) -> BenchmarkResult:
        """Run a single benchmark test."""
        print(f"🏃 Running benchmark: {test_name}")
        
        start_time = time.time()
        
        try:
            result = self.pipeline.process_video(video_url, target_language)
            processing_time = time.time() - start_time
            
            benchmark_result = BenchmarkResult(
                test_name=test_name,
                video_url=video_url,
                target_language=target_language,
                video_duration=result.data.duration if result.success else 0,
                processing_time=processing_time,
                success=result.success,
                error_message=result.error_message or ""
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            benchmark_result = BenchmarkResult(
                test_name=test_name,
                video_url=video_url,
                target_language=target_language,
                video_duration=0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        print("🚀 Starting comprehensive benchmark suite...")
        
        # Test configurations
        test_configs = [
            {
                "name": "Short Video - Hindi",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "language": "hi"
            },
            {
                "name": "Short Video - Bengali", 
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "language": "bn"
            },
            {
                "name": "Short Video - Tamil",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", 
                "language": "ta"
            }
        ]
        
        # Run benchmarks
        for config in test_configs:
            self.run_single_benchmark(
                config["name"],
                config["url"], 
                config["language"]
            )
        
        # Analyze results
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate statistics."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if successful_results:
            processing_times = [r.processing_time for r in successful_results]
            video_durations = [r.video_duration for r in successful_results]
            
            analysis = {
                "summary": {
                    "total_tests": len(self.results),
                    "successful": len(successful_results),
                    "failed": len(failed_results),
                    "success_rate": len(successful_results) / len(self.results) * 100
                },
                "performance_metrics": {
                    "avg_processing_time": statistics.mean(processing_times),
                    "min_processing_time": min(processing_times),
                    "max_processing_time": max(processing_times),
                    "median_processing_time": statistics.median(processing_times),
                    "avg_video_duration": statistics.mean(video_durations),
                    "processing_speed_ratio": statistics.mean([
                        r.processing_time / r.video_duration 
                        for r in successful_results if r.video_duration > 0
                    ])
                },
                "detailed_results": [asdict(r) for r in self.results]
            }
        else:
            analysis = {
                "summary": {
                    "total_tests": len(self.results),
                    "successful": 0,
                    "failed": len(failed_results),
                    "success_rate": 0
                },
                "errors": [r.error_message for r in failed_results],
                "detailed_results": [asdict(r) for r in self.results]
            }
        
        return analysis
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        results_path = Path(__file__).parent / filename
        analysis = self.analyze_results()
        
        with open(results_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return str(results_path)
    
    def print_summary(self):
        """Print benchmark summary to console."""
        analysis = self.analyze_results()
        
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        summary = analysis.get("summary", {})
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Successful: {summary.get('successful', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        if "performance_metrics" in analysis:
            metrics = analysis["performance_metrics"]
            print(f"\n⏱️  Performance Metrics:")
            print(f"  Average Processing Time: {metrics.get('avg_processing_time', 0):.2f}s")
            print(f"  Fastest Processing: {metrics.get('min_processing_time', 0):.2f}s")
            print(f"  Slowest Processing: {metrics.get('max_processing_time', 0):.2f}s")
            print(f"  Processing Speed Ratio: {metrics.get('processing_speed_ratio', 0):.2f}x")
        
        if analysis.get("errors"):
            print(f"\n❌ Errors encountered:")
            for error in set(analysis["errors"]):
                print(f"  • {error}")


def main():
    """Main benchmarking function."""
    print("⚡ Viral-Local Performance Benchmark")
    print("=" * 50)
    
    try:
        # Initialize benchmark
        benchmark = PerformanceBenchmark()
        
        # Run comprehensive benchmark
        print("🔍 Running comprehensive benchmark suite...")
        results = benchmark.run_comprehensive_benchmark()
        
        # Print summary
        benchmark.print_summary()
        
        # Save results
        results_file = benchmark.save_results()
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Performance recommendations
        print("\n💡 Performance Recommendations:")
        if results.get("performance_metrics"):
            speed_ratio = results["performance_metrics"].get("processing_speed_ratio", 0)
            if speed_ratio > 3:
                print("  ⚠️  Processing is slower than expected. Consider:")
                print("     • Using a smaller Whisper model")
                print("     • Enabling GPU acceleration")
                print("     • Reducing video quality settings")
            elif speed_ratio < 1.5:
                print("  ✅ Excellent performance! System is well optimized.")
            else:
                print("  👍 Good performance within expected range.")
        
        return 0
    
    except Exception as e:
        print(f"💥 Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())