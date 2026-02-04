# Viral-Local Troubleshooting Guide

This guide helps you diagnose and resolve common issues with Viral-Local.

## Quick Diagnostics

### System Validation
Run the built-in validation to check your setup:

```bash
python -m viral_local --validate
```

This will check:
- ✅ API keys configuration
- ✅ Required directories
- ✅ Language support
- ✅ System dependencies

## Common Issues and Solutions

### 1. Installation Issues

#### Problem: `pip install` fails
```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**
```bash
# Try with --user flag
pip install --user -r requirements.txt

# Or upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt

# On Windows, try:
python -m pip install --upgrade pip setuptools wheel
```

#### Problem: FFmpeg not found
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### 2. Configuration Issues

#### Problem: API key errors
```
ValueError: Gemini API key is required
```

**Solutions:**
1. Check your `config.yaml` file:
   ```yaml
   gemini_api_key: "your_actual_api_key_here"
   ```

2. Or set environment variable:
   ```bash
   export VIRAL_LOCAL_GEMINI_API_KEY="your_api_key"
   ```

3. Verify API key is valid:
   ```python
   import google.generativeai as genai
   genai.configure(api_key="your_key")
   model = genai.GenerativeModel('gemini-1.5-flash')
   response = model.generate_content("Hello")
   print(response.text)
   ```

#### Problem: Invalid configuration
```
ValueError: Invalid Whisper model size
```

**Solution:** Check valid options in config:
```yaml
whisper_model_size: "base"  # Valid: tiny, base, small, medium, large
tts_engine: "edge-tts"      # Valid: edge-tts, kokoro-82m
target_audio_quality: "high"  # Valid: low, medium, high
```

### 3. Processing Issues

#### Problem: YouTube download fails
```
DownloadError: Failed to download video
```

**Solutions:**
1. **Check URL format:**
   ```python
   # Valid formats:
   "https://www.youtube.com/watch?v=VIDEO_ID"
   "https://youtu.be/VIDEO_ID"
   "https://youtube.com/watch?v=VIDEO_ID"
   ```

2. **Update yt-dlp:**
   ```bash
   pip install --upgrade yt-dlp
   ```

3. **Check video accessibility:**
   - Video might be private or geo-blocked
   - Video might be age-restricted
   - Video might have been deleted

4. **Network issues:**
   ```bash
   # Test internet connection
   ping google.com
   
   # Test YouTube access
   curl -I https://www.youtube.com
   ```

#### Problem: Transcription fails
```
TranscriptionError: Failed to transcribe audio
```

**Solutions:**
1. **Check audio quality:**
   - Ensure video has clear audio
   - Check if audio track exists

2. **Try smaller Whisper model:**
   ```yaml
   whisper_model_size: "tiny"  # Faster, less memory
   ```

3. **Check available memory:**
   ```python
   import psutil
   print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
   ```

4. **Disable GPU if causing issues:**
   ```yaml
   enable_gpu: false
   ```

#### Problem: Translation fails
```
TranslationError: Translation to hi failed
```

**Solutions:**
1. **Check API quotas:**
   - Verify you haven't exceeded API limits
   - Check billing status for paid APIs

2. **Try alternative API:**
   ```yaml
   groq_api_key: "your_groq_key"  # Fallback API
   ```

3. **Check language support:**
   ```yaml
   supported_languages: ["hi", "bn", "ta"]  # Supported languages
   ```

#### Problem: Audio generation fails
```
AudioGenerationError: Failed to generate audio
```

**Solutions:**
1. **Check TTS engine:**
   ```bash
   # Test Edge-TTS
   edge-tts --text "Hello" --voice "hi-IN-SwaraNeural" --write-media test.wav
   ```

2. **Try different voice:**
   ```python
   from viral_local.models import VoiceConfig
   voice_config = VoiceConfig(
       language="hi",
       gender="female",  # Try: male, female, neutral
       age_range="adult"  # Try: young, adult, elderly
   )
   ```

### 4. Performance Issues

#### Problem: Processing is very slow
```
Processing taking longer than expected
```

**Solutions:**
1. **Use smaller models:**
   ```yaml
   whisper_model_size: "tiny"  # Fastest
   target_audio_quality: "medium"  # Faster processing
   ```

2. **Enable GPU acceleration:**
   ```yaml
   enable_gpu: true
   ```
   
   Install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Increase cache size:**
   ```yaml
   cache_enabled: true
   cache_max_size_mb: 2000
   ```

4. **Reduce concurrent requests:**
   ```yaml
   max_concurrent_requests: 1
   ```

#### Problem: Out of memory errors
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Reduce batch size:**
   ```yaml
   batch_size: 8  # Reduce from default 16
   ```

2. **Use CPU instead of GPU:**
   ```yaml
   enable_gpu: false
   ```

3. **Use smaller Whisper model:**
   ```yaml
   whisper_model_size: "base"  # Instead of large
   ```

4. **Clear cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### 5. Web Interface Issues

#### Problem: Streamlit not starting
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install streamlit
```

#### Problem: Web interface crashes
```
StreamlitAPIException: ...
```

**Solutions:**
1. **Check port availability:**
   ```bash
   netstat -an | grep 8501
   ```

2. **Use different port:**
   ```bash
   streamlit run viral_local/web_app.py --server.port 8502
   ```

3. **Clear Streamlit cache:**
   ```bash
   streamlit cache clear
   ```

## Advanced Debugging

### Enable Debug Logging
```yaml
log_level: "DEBUG"
enable_file_logging: true
log_file: "debug.log"
```

### Memory Profiling
```python
import tracemalloc
tracemalloc.start()

# Your code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

### Performance Profiling
```python
import cProfile
import pstats

# Profile your code
cProfile.run('your_function()', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

## Getting Help

### 1. Check Logs
Look for detailed error information in log files:
```bash
tail -f viral_local.log
```

### 2. Run Diagnostics
```bash
python examples/performance_benchmark.py
```

### 3. System Information
```python
import sys
import platform
import torch

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 4. Create Issue Report
When reporting issues, include:

1. **System Information:**
   - Operating system and version
   - Python version
   - Viral-Local version

2. **Configuration:**
   - Relevant config.yaml sections
   - Environment variables used

3. **Error Details:**
   - Complete error message
   - Stack trace
   - Log file excerpts

4. **Reproduction Steps:**
   - Exact commands used
   - Input data (YouTube URL, etc.)
   - Expected vs actual behavior

### 5. Community Support
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-username/viral-local/issues)
- **Discussions**: [Community discussions](https://github.com/your-username/viral-local/discussions)
- **Documentation**: [Full documentation](https://viral-local.readthedocs.io)

## Prevention Tips

1. **Regular Updates:**
   ```bash
   pip install --upgrade viral-local
   ```

2. **Monitor Resources:**
   - Check disk space regularly
   - Monitor memory usage
   - Clean temporary files

3. **Backup Configuration:**
   - Keep backup of working config.yaml
   - Document custom settings

4. **Test Changes:**
   - Use validation before production
   - Test with small videos first
   - Keep rollback plan ready