# Viral-Local Setup Guide

This guide will help you set up and configure Viral-Local for video localization.

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for models and temporary files
- **Internet**: Stable connection for API calls and video downloads

### Required Software

1. **FFmpeg**: Required for video/audio processing
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS (with Homebrew)
   brew install ffmpeg
   
   # Windows (with Chocolatey)
   choco install ffmpeg
   ```

2. **Git**: For cloning the repository
   ```bash
   # Ubuntu/Debian
   sudo apt install git
   
   # macOS (with Homebrew)
   brew install git
   
   # Windows: Download from https://git-scm.com/
   ```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/viral-local.git
cd viral-local
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For web interface (optional)
pip install streamlit

# For development (optional)
pip install -r requirements-dev.txt
```

### 4. Install System Package

```bash
# Install in development mode
pip install -e .
```

## Configuration

### 1. API Keys Setup

Viral-Local requires API keys for AI services:

#### Gemini API Key (Required)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key for configuration

#### Groq API Key (Optional, for faster processing)
1. Go to [Groq Console](https://console.groq.com/)
2. Create an account and generate an API key
3. Copy the key for configuration

### 2. Configuration File

Create a configuration file `config.yaml`:

```yaml
# API Configuration
gemini_api_key: "your_gemini_api_key_here"
groq_api_key: "your_groq_api_key_here"  # Optional

# Model Settings
whisper_model_size: "base"  # tiny, base, small, medium, large
tts_engine: "edge-tts"

# Processing Limits
max_video_duration: 1800  # 30 minutes in seconds
max_concurrent_requests: 3

# Quality Settings
target_audio_quality: "high"  # low, medium, high
video_output_format: "mp4"

# Language Support
supported_languages: ["hi", "bn", "ta"]
default_target_language: "hi"

# Directory Settings
temp_dir: "temp"
output_dir: "output"
cache_dir: "cache"

# Logging
log_level: "INFO"
enable_file_logging: true

# Performance
enable_gpu: true
cache_enabled: true
```

### 3. Environment Variables (Alternative)

You can also use environment variables:

```bash
export VIRAL_LOCAL_GEMINI_API_KEY="your_gemini_api_key"
export VIRAL_LOCAL_GROQ_API_KEY="your_groq_api_key"
export VIRAL_LOCAL_WHISPER_MODEL="base"
export VIRAL_LOCAL_LOG_LEVEL="INFO"
```

## Verification

### 1. Test Installation

```bash
# Test CLI
python -m viral_local --validate

# Test basic functionality
python examples/basic_usage.py
```

### 2. Run System Validation

```python
from viral_local.main import ViralLocalPipeline

pipeline = ViralLocalPipeline()
results = pipeline.validate_setup()
print(results)
```

Expected output:
```
{
    'gemini': True,
    'groq': True,  # If configured
    'directories': True,
    'languages': True
}
```

## Troubleshooting

### Common Issues

#### 1. FFmpeg Not Found
```
Error: FFmpeg not found
```
**Solution**: Install FFmpeg and ensure it's in your PATH.

#### 2. API Key Issues
```
Error: Gemini API key is required
```
**Solution**: Check your configuration file or environment variables.

#### 3. Memory Issues
```
Error: Out of memory
```
**Solution**: 
- Use a smaller Whisper model (`tiny` or `base`)
- Reduce `max_concurrent_requests`
- Close other applications

#### 4. GPU Issues
```
Warning: GPU requested but not available
```
**Solution**: 
- Install PyTorch with CUDA support
- Or set `enable_gpu: false` in config

### Getting Help

1. **Check Logs**: Look at the log file for detailed error information
2. **Validate Setup**: Run `python -m viral_local --validate`
3. **GitHub Issues**: Report bugs at [GitHub Issues](https://github.com/your-username/viral-local/issues)

## Next Steps

1. **Basic Usage**: Try `examples/basic_usage.py`
2. **Web Interface**: Run `python examples/web_interface_example.py`
3. **Advanced Features**: Explore `examples/advanced_usage.py`
4. **Performance Testing**: Use `examples/performance_benchmark.py`

## Security Notes

- Keep your API keys secure and never commit them to version control
- Use environment variables or secure configuration management in production
- Regularly rotate your API keys
- Monitor API usage to avoid unexpected charges

## Performance Optimization

### For Better Speed:
- Use smaller Whisper models (`tiny`, `base`)
- Enable GPU acceleration
- Use SSD storage for temporary files
- Increase available RAM

### For Better Quality:
- Use larger Whisper models (`small`, `medium`, `large`)
- Set `target_audio_quality: "high"`
- Use both Gemini and Groq APIs for redundancy

### For Production:
- Set up proper logging and monitoring
- Use Docker for consistent deployment
- Implement rate limiting and error handling
- Set up backup and recovery procedures