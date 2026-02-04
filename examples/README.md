# Viral-Local Examples

This directory contains example scripts and usage patterns for Viral-Local video localization system.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)
**Purpose**: Demonstrates the simplest way to use Viral-Local

**Features:**
- Single video processing
- Default configuration
- Basic error handling
- System validation

**Usage:**
```bash
python examples/basic_usage.py
```

**What it does:**
1. Initializes Viral-Local pipeline
2. Validates system setup
3. Processes a sample YouTube video
4. Outputs localized video

### 2. Advanced Usage (`advanced_usage.py`)
**Purpose**: Shows advanced features and customization options

**Features:**
- Custom configuration creation
- Batch processing multiple videos
- Detailed progress tracking
- Viral segment analysis demonstration
- Performance metrics collection

**Usage:**
```bash
python examples/advanced_usage.py
```

**What it demonstrates:**
1. Creating custom configuration files
2. Processing multiple videos in different languages
3. Analyzing viral segments in content
4. Collecting detailed performance metrics

### 3. Web Interface (`web_interface_example.py`)
**Purpose**: Demonstrates web-based interface setup

**Features:**
- Streamlit web interface
- Docker deployment setup
- Web-optimized configuration
- User-friendly interface

**Usage:**
```bash
python examples/web_interface_example.py
```

**Options:**
1. Run web interface locally
2. Create Docker deployment files
3. Both options combined

### 4. Performance Benchmark (`performance_benchmark.py`)
**Purpose**: Comprehensive performance testing and benchmarking

**Features:**
- Multiple test scenarios
- Performance metrics collection
- Statistical analysis
- Results export
- Performance recommendations

**Usage:**
```bash
python examples/performance_benchmark.py
```

**Metrics measured:**
- Processing time per video
- Memory usage patterns
- Success/failure rates
- Speed ratios (processing time vs video duration)

## Configuration Examples

### Basic Configuration (`config.yaml`)
```yaml
# Minimal configuration for getting started
gemini_api_key: "your_api_key_here"
whisper_model_size: "base"
target_audio_quality: "medium"
supported_languages: ["hi", "bn", "ta"]
```

### Advanced Configuration (`advanced_config.yaml`)
```yaml
# Advanced configuration with all options
gemini_api_key: "your_gemini_key"
groq_api_key: "your_groq_key"
whisper_model_size: "small"
tts_engine: "edge-tts"
max_video_duration: 1800
target_audio_quality: "high"
enable_gpu: true
cache_enabled: true
log_level: "DEBUG"
```

### Web Configuration (`web_config.yaml`)
```yaml
# Optimized for web interface
gemini_api_key: "your_api_key"
whisper_model_size: "base"
max_video_duration: 900  # 15 minutes
target_audio_quality: "medium"
enable_gpu: false
log_level: "WARNING"
```

## Usage Patterns

### 1. Single Video Processing
```python
from viral_local.main import ViralLocalPipeline

pipeline = ViralLocalPipeline()
result = pipeline.process_video(
    "https://youtube.com/watch?v=VIDEO_ID",
    "hi"  # Hindi
)

if result.success:
    print(f"Success: {result.data.file_path}")
else:
    print(f"Error: {result.error_message}")
```

### 2. Batch Processing
```python
videos = [
    {"url": "https://youtube.com/watch?v=ID1", "lang": "hi"},
    {"url": "https://youtube.com/watch?v=ID2", "lang": "bn"},
    {"url": "https://youtube.com/watch?v=ID3", "lang": "ta"},
]

for video in videos:
    result = pipeline.process_video(video["url"], video["lang"])
    # Handle result...
```

### 3. Custom Progress Tracking
```python
from viral_local.main import ViralLocalPipeline, ProgressTracker

class CustomTracker(ProgressTracker):
    def start_stage(self, stage_name, description):
        print(f"Starting: {stage_name}")
    
    def complete_stage(self, message=""):
        print(f"Completed: {message}")

pipeline = ViralLocalPipeline(progress_tracker=CustomTracker())
```

### 4. Error Handling
```python
from viral_local.utils import ViralLocalError, DownloadError, TranscriptionError

try:
    result = pipeline.process_video(url, language)
except DownloadError as e:
    print(f"Download failed: {e.message}")
    print(f"URL: {e.details.get('url')}")
except TranscriptionError as e:
    print(f"Transcription failed: {e.message}")
    print(f"Audio file: {e.details.get('audio_file')}")
except ViralLocalError as e:
    print(f"General error: {e.message}")
    print(f"Error code: {e.error_code}")
```

## Testing Your Setup

### 1. Quick Validation
```bash
python -c "from viral_local.main import ViralLocalPipeline; print(ViralLocalPipeline().validate_setup())"
```

### 2. Component Testing
```python
# Test individual components
from viral_local.services import DownloaderService
from viral_local.config import SystemConfig

config = SystemConfig(gemini_api_key="your_key")
downloader = DownloaderService(config)

# Test URL validation
is_valid = downloader.validate_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
print(f"URL valid: {is_valid}")
```

### 3. Performance Testing
```bash
# Run benchmark suite
python examples/performance_benchmark.py

# Check results
cat benchmark_results_*.json
```

## Customization Examples

### 1. Custom Voice Configuration
```python
from viral_local.models import VoiceConfig

voice_config = VoiceConfig(
    language="hi",
    gender="female",
    age_range="young",
    speaking_rate=1.1,
    pitch_adjustment=0.1
)
```

### 2. Custom Translation Prompts
```python
# Extend LocalizationEngine for custom prompts
from viral_local.services import LocalizationEngine

class CustomLocalizer(LocalizationEngine):
    def _create_translation_prompt(self, segments, target_lang):
        # Your custom prompt logic
        return custom_prompt
```

### 3. Custom Progress Tracking
```python
class DatabaseProgressTracker(ProgressTracker):
    def start_processing(self, url, target_language):
        # Log to database
        self.db.log_start(url, target_language)
    
    def show_completion(self, result_path):
        # Update database with results
        self.db.log_completion(result_path)
```

## Docker Deployment

### 1. Basic Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "viral_local"]
```

### 2. Docker Compose
```yaml
version: '3.8'
services:
  viral-local:
    build: .
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./output:/app/output
```

## Best Practices

### 1. Configuration Management
- Use environment variables for sensitive data
- Keep separate configs for dev/staging/production
- Version control your configuration templates

### 2. Error Handling
- Always check result.success before using result.data
- Log errors with sufficient context
- Implement retry logic for transient failures

### 3. Performance Optimization
- Use appropriate Whisper model size for your use case
- Enable caching for repeated processing
- Monitor memory usage and clean up temporary files

### 4. Security
- Never commit API keys to version control
- Use secure methods to store and retrieve credentials
- Regularly rotate API keys

## Troubleshooting

If examples don't work:

1. **Check Prerequisites:**
   ```bash
   python --version  # Should be 3.8+
   pip list | grep viral-local
   ```

2. **Validate Setup:**
   ```bash
   python -m viral_local --validate
   ```

3. **Check Logs:**
   ```bash
   tail -f viral_local.log
   ```

4. **Test Components:**
   ```python
   # Test API connectivity
   import google.generativeai as genai
   genai.configure(api_key="your_key")
   ```

For more help, see [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) or create an issue on GitHub.