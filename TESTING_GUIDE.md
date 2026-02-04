# 🧪 Viral-Local Testing Guide

## Quick Start - How to Test the System

The Viral-Local system provides **two interfaces** for testing and usage:

### 🖥️ 1. Command Line Interface (CLI)
### 🌐 2. Web Interface (Streamlit)

---

## 🚀 Setup Instructions

### Step 1: Install Dependencies
```bash
# Install required Python packages
pip install -r requirements.txt

# Install additional dependencies for full functionality
pip install openai-whisper yt-dlp streamlit
```

### Step 2: Configure API Keys
Create a `config.yaml` file with your API keys:

```yaml
# Required API key
gemini_api_key: "your_gemini_api_key_here"

# Optional API key for faster processing
groq_api_key: "your_groq_api_key_here"

# System settings
whisper_model_size: "base"
tts_engine: "edge-tts"
supported_languages: ["hi", "bn", "ta"]
default_target_language: "hi"
enable_gpu: false  # Set to true if you have GPU
cache_enabled: true
```

---

## 🖥️ Testing the CLI Interface

### Basic Usage
```bash
# Process a YouTube video to Hindi
python -m viral_local.main "https://youtube.com/watch?v=VIDEO_ID" --target-lang hi

# Process to Bengali with custom config
python -m viral_local.main "https://youtu.be/VIDEO_ID" -t bn --config my_config.yaml

# Process to Tamil with verbose output
python -m viral_local.main "https://youtube.com/watch?v=VIDEO_ID" -t ta --verbose
```

### System Validation
```bash
# Validate system setup (recommended first step)
python -m viral_local.main --validate --config config.yaml

# Check if everything is configured correctly
python -m viral_local.main --validate --verbose
```

### CLI Options
```bash
# See all available options
python -m viral_local.main --help

# Available options:
# --target-lang, -t    : Target language (hi, bn, ta)
# --config, -c         : Configuration file path
# --validate           : Validate system setup
# --verbose, -v        : Enable verbose logging
# --output-dir, -o     : Override output directory
# --no-progress        : Disable progress indicators
```

### Example CLI Session
```bash
# 1. First, validate your setup
python -m viral_local.main --validate --config config.yaml

# 2. Process a short test video
python -m viral_local.main "https://youtube.com/watch?v=dQw4w9WgXcQ" --target-lang hi --verbose

# 3. Check the output directory for results
ls output/
```

---

## 🌐 Testing the Web Interface

### Launch the Web App
```bash
# Method 1: Using the launcher script
python run_web_app.py

# Method 2: Direct Streamlit command
streamlit run viral_local/web_app.py

# Method 3: With custom port
streamlit run viral_local/web_app.py --server.port 8502
```

### Web Interface Features
- **📹 Video URL Input**: Paste YouTube URLs directly
- **🌍 Language Selection**: Choose from Hindi, Bengali, Tamil
- **⚙️ Configuration Panel**: Set API keys and processing options
- **📊 Real-time Progress**: Visual progress bars and status updates
- **📥 Download Results**: Direct download of processed videos
- **🔧 Settings**: Adjust Whisper model, audio quality, GPU usage

### Web Interface Testing Steps
1. **Open Browser**: Navigate to `http://localhost:8501`
2. **Configure API Keys**: Enter your Gemini API key in the sidebar
3. **Enter Video URL**: Paste a YouTube URL (start with short videos)
4. **Select Language**: Choose Hindi, Bengali, or Tamil
5. **Start Processing**: Click "🚀 Start Processing"
6. **Monitor Progress**: Watch real-time progress updates
7. **Download Result**: Use the download button when complete

---

## 🧪 Test Cases for Validation

### Test Case 1: System Validation
```bash
# Test system setup
python -m viral_local.main --validate --config config.yaml

# Expected: All validations should pass
# ✅ Gemini API: Valid
# ✅ Directories: Created
# ✅ Languages: Configured
```

### Test Case 2: Short Video Processing (CLI)
```bash
# Test with a 1-2 minute video
python -m viral_local.main "https://youtube.com/watch?v=SHORT_VIDEO" --target-lang hi

# Expected: Complete processing in under 2 minutes
# Output: MP4 file in output/ directory
```

### Test Case 3: Language Support Testing
```bash
# Test Hindi
python -m viral_local.main "URL" --target-lang hi

# Test Bengali  
python -m viral_local.main "URL" --target-lang bn

# Test Tamil
python -m viral_local.main "URL" --target-lang ta
```

### Test Case 4: Error Handling
```bash
# Test invalid URL
python -m viral_local.main "invalid_url" --target-lang hi

# Expected: Clear error message about invalid URL

# Test without API key
python -m viral_local.main "valid_url" --target-lang hi

# Expected: Error about missing API key
```

### Test Case 5: Web Interface Testing
1. Launch web app: `python run_web_app.py`
2. Test without API key (should show warning)
3. Configure API key in sidebar
4. Test with invalid URL (should show error)
5. Test with valid short video URL
6. Monitor progress and download result

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Whisper not available" Error
```bash
# Install Whisper
pip install openai-whisper

# Or install with GPU support
pip install openai-whisper[gpu]
```

#### 2. "Gemini API key required" Error
- Add your API key to `config.yaml`
- Or set environment variable: `export VIRAL_LOCAL_GEMINI_API_KEY="your_key"`

#### 3. "YouTube video unavailable" Error
- Check if the video is public and accessible
- Try a different video URL
- Ensure the video is under 30 minutes

#### 4. Streamlit Import Error
```bash
# Install Streamlit
pip install streamlit

# Launch web app
python run_web_app.py
```

#### 5. Permission/Directory Errors
```bash
# Create required directories
mkdir -p temp output cache

# Check permissions
ls -la temp/ output/ cache/
```

### Performance Tips
- **Start Small**: Test with 1-2 minute videos first
- **GPU Acceleration**: Enable GPU if available for faster processing
- **Model Selection**: Use "base" Whisper model for speed, "small" for accuracy
- **Cache**: Enable caching for repeated processing

---

## 📊 Expected Results

### Successful Processing Should Produce:
- **Input**: YouTube video URL + target language
- **Output**: Localized MP4 video file
- **Features**: 
  - Original video quality preserved
  - Audio dubbed in target language
  - Synchronized timing
  - Cultural adaptations applied

### Processing Time Expectations:
- **1-minute video**: ~30-60 seconds processing
- **5-minute video**: ~2-3 minutes processing  
- **10-minute video**: ~4-6 minutes processing

### File Locations:
- **Output Videos**: `output/` directory
- **Temporary Files**: `temp/` directory (auto-cleaned)
- **Cache**: `cache/` directory
- **Logs**: `viral_local.log` file

---

## 🎯 Recommended Testing Sequence

### For First-Time Users:
1. **Install dependencies** and create config file
2. **Run system validation**: `python -m viral_local.main --validate`
3. **Test CLI with short video** (1-2 minutes)
4. **Launch web interface**: `python run_web_app.py`
5. **Test web interface** with same short video
6. **Try different languages** (hi, bn, ta)
7. **Test with longer video** (5-10 minutes)

### For Developers:
1. **Run test suite**: `python -m pytest tests/ -v`
2. **Test CLI interface** with various scenarios
3. **Test web interface** functionality
4. **Performance testing** with different video lengths
5. **Error handling testing** with invalid inputs

---

## 📞 Getting Help

If you encounter issues:

1. **Check logs**: Look at `viral_local.log` for detailed error information
2. **Run validation**: Use `--validate` flag to check system setup
3. **Test with simple case**: Start with a short, public YouTube video
4. **Check API keys**: Ensure your Gemini API key is valid and has quota
5. **Review requirements**: Make sure all dependencies are installed

The system is designed to provide clear error messages and guidance for troubleshooting common issues.

---

**Happy Testing! 🎬✨**