# Viral-Local

Automated Video Localization System for Indian Languages.

Transform your YouTube videos into multiple Indian languages with AI-powered dubbing!

## Features

- **Intelligent Viral Segment Detection** - AI identifies the most engaging parts of your content
- **Natural Text-to-Speech Generation** - High-quality voice synthesis with Edge-TTS
- **Audio-Video Synchronization** - Precise timing alignment for natural playback
- **Multi-Language Support** - Hindi, Bengali, and Tamil localization
- **Dual Interface** - Both command-line and web interfaces available
- **GPU Acceleration** - Faster processing with CUDA support

## Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **API Keys** (required):
   - Gemini API key (required) - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Groq API key (optional) - Get from [Groq Console](https://console.groq.com/)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd viral-local
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml and add your API keys
```

## Usage

### Command Line Interface

Process a video directly from the command line:

```bash
# Basic usage
python -m viral_local.main "https://youtube.com/watch?v=VIDEO_ID" --target-lang hi

# With custom configuration
python -m viral_local.main "https://youtu.be/VIDEO_ID" -t bn --config my_config.yaml

# Validate system setup
python -m viral_local.main --validate

# Verbose output
python -m viral_local.main "https://youtube.com/watch?v=VIDEO_ID" -t ta --verbose
```

**CLI Options:**
- `--target-lang, -t`: Target language (`hi`, `bn`, `ta`)
- `--config, -c`: Path to configuration file
- `--output-dir, -o`: Override output directory
- `--verbose, -v`: Enable verbose logging
- `--validate`: Validate system setup
- `--no-progress`: Disable progress indicators

### Web Interface

Launch the user-friendly web interface:

```bash
python run_web_app.py
```

Or directly with Streamlit:

```bash
streamlit run viral_local/web_app.py
```

The web interface provides:
- Intuitive URL input and language selection
- Real-time progress tracking
- Configuration management
- Direct video download

## Supported Languages

| Language | Code | Script |
|----------|------|--------|
| Hindi | `hi` | हिन्दी |
| Bengali | `bn` | বাংলা |
| Tamil | `ta` | தமிழ் |

## Configuration

The system uses YAML configuration files. Key settings:

```yaml
# API Keys
gemini_api_key: "your_key_here"
groq_api_key: "optional_key_here"

# Model Settings
whisper_model_size: "base"  # tiny, base, small, medium, large
tts_engine: "edge-tts"

# Processing Limits
max_video_duration: 1800  # 30 minutes
target_audio_quality: "high"

# Performance
enable_gpu: true
cache_enabled: true
```

## System Requirements

### Minimum Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for temporary files
- **Internet**: Stable connection for API calls and video download

### Recommended for Best Performance
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 32GB for processing longer videos
- **CPU**: 8+ cores for faster transcription

## Processing Pipeline

1. **Download & Extract** - Downloads YouTube video and extracts audio
2. **Transcription** - Uses Whisper AI for speech-to-text conversion
3. **Viral Analysis** - AI identifies high-engagement segments
4. **Translation** - Gemini/Groq API translates content with cultural adaptation
5. **Speech Generation** - Edge-TTS creates natural-sounding localized audio
6. **Video Assembly** - MoviePy merges localized audio with original video

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_cli.py
pytest tests/test_dubbing.py

# Run with coverage
pytest --cov=viral_local
```

### Project Structure

```
viral_local/
├── services/          # Core processing services
│   ├── downloader.py  # YouTube video download
│   ├── transcriber.py # Speech-to-text processing
│   ├── localization.py # AI translation & analysis
│   └── dubbing.py     # TTS and video assembly
├── models.py          # Data models
├── config.py          # Configuration management
├── main.py           # CLI interface
├── web_app.py        # Streamlit web interface
└── utils/            # Utilities and error handling
```

## Troubleshooting

### Common Issues

**"Gemini API key is required"**
- Ensure your API key is correctly set in `config.yaml`
- Verify the key is valid and has sufficient quota

**"Invalid YouTube URL"**
- Check the URL format is correct
- Ensure the video is publicly accessible
- Try with a different video

**"Video duration exceeds limit"**
- Videos must be under 30 minutes
- Consider processing shorter clips

**Memory errors**
- Reduce `whisper_model_size` to "base" or "small"
- Close other applications to free RAM
- Process shorter videos

### Getting Help

1. Check the logs in `viral_local.log`
2. Run with `--verbose` flag for detailed output
3. Use `--validate` to check system configuration
4. Ensure all dependencies are properly installed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Google Gemini for AI-powered translation
- Microsoft Edge-TTS for speech synthesis
- MoviePy for video processing
- Streamlit for the web interface

---

Built with ❤️ for the Indian creator community
