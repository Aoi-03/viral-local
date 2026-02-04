#!/usr/bin/env python3
"""
Web interface example for Viral-Local video localization.

This script demonstrates how to run the Streamlit web interface
and customize it for different use cases.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the parent directory to the path so we can import viral_local
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_streamlit_installation():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def install_streamlit():
    """Install Streamlit if not available."""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Streamlit: {e}")
        return False


def create_custom_web_config():
    """Create a configuration file optimized for web usage."""
    import yaml
    
    config_data = {
        # API Configuration
        'gemini_api_key': os.getenv('GEMINI_API_KEY', ''),
        'groq_api_key': os.getenv('GROQ_API_KEY', ''),
        
        # Model Settings - Balanced for web usage
        'whisper_model_size': 'base',  # Good balance of speed and accuracy
        'tts_engine': 'edge-tts',
        
        # Processing Limits - Conservative for web
        'max_video_duration': 900,  # 15 minutes for web usage
        'max_concurrent_requests': 1,  # Single request for web
        
        # Quality Settings
        'target_audio_quality': 'medium',  # Faster processing
        'video_output_format': 'mp4',
        
        # Language Support
        'supported_languages': ['hi', 'bn', 'ta'],
        'default_target_language': 'hi',
        
        # Directory Settings
        'temp_dir': 'temp_web',
        'output_dir': 'output_web',
        'cache_dir': 'cache_web',
        
        # Logging Configuration - Minimal for web
        'log_level': 'WARNING',
        'enable_file_logging': False,  # Disable file logging for web
        
        # Performance Settings
        'enable_gpu': False,  # Disable GPU for web compatibility
        'cache_enabled': True,
        'cache_max_size_mb': 500,  # Smaller cache for web
        
        # Retry Settings
        'max_retries': 2,  # Fewer retries for faster feedback
        'retry_delay': 1.0,
    }
    
    config_path = Path(__file__).parent / "web_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    return str(config_path)


def run_web_interface():
    """Run the Streamlit web interface."""
    print("🌐 Starting Viral-Local Web Interface...")
    print("=" * 50)
    
    # Create web-optimized configuration
    config_path = create_custom_web_config()
    print(f"⚙️ Created web configuration: {config_path}")
    
    # Set environment variable for the config
    os.environ['VIRAL_LOCAL_CONFIG'] = config_path
    
    # Get the path to the web app module
    web_app_path = Path(__file__).parent.parent / "viral_local" / "web_app.py"
    
    if not web_app_path.exists():
        print(f"❌ Web app not found at: {web_app_path}")
        return 1
    
    try:
        # Run Streamlit
        print("🚀 Launching web interface...")
        print("📱 The web interface will open in your default browser")
        print("🔗 URL: http://localhost:8501")
        print("\n💡 Tips for web usage:")
        print("  • Use shorter videos (< 15 minutes) for better performance")
        print("  • Ensure your API keys are configured")
        print("  • Check the sidebar for configuration options")
        print("\n⏹️  Press Ctrl+C to stop the server")
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(web_app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
        return 0
    
    except KeyboardInterrupt:
        print("\n⏹️ Web server stopped by user.")
        return 0
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start web interface: {e}")
        return 1


def create_docker_setup():
    """Create Docker setup files for web deployment."""
    print("🐳 Creating Docker setup for web deployment...")
    
    # Create Dockerfile
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional web dependencies
RUN pip install streamlit

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the web interface
CMD ["python", "-m", "streamlit", "run", "viral_local/web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    # Create docker-compose.yml
    docker_compose_content = """version: '3.8'

services:
  viral-local-web:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./output_web:/app/output_web
      - ./temp_web:/app/temp_web
      - ./cache_web:/app/cache_web
    restart: unless-stopped
"""
    
    compose_path = Path(__file__).parent / "docker-compose.yml"
    with open(compose_path, 'w') as f:
        f.write(docker_compose_content)
    
    # Create .env template
    env_template = """# Copy this file to .env and fill in your API keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
"""
    
    env_path = Path(__file__).parent / ".env.template"
    with open(env_path, 'w') as f:
        f.write(env_template)
    
    print(f"✅ Docker setup created:")
    print(f"  📄 Dockerfile: {dockerfile_path}")
    print(f"  📄 docker-compose.yml: {compose_path}")
    print(f"  📄 .env.template: {env_path}")
    print("\n🚀 To deploy with Docker:")
    print("  1. Copy .env.template to .env and add your API keys")
    print("  2. Run: docker-compose up --build")
    print("  3. Access at: http://localhost:8501")


def main():
    """Main function for web interface example."""
    print("🌐 Viral-Local Web Interface Example")
    print("=" * 50)
    
    # Check if Streamlit is installed
    if not check_streamlit_installation():
        print("❌ Streamlit is not installed.")
        install_choice = input("📦 Would you like to install it? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_streamlit():
                return 1
        else:
            print("💡 You can install Streamlit manually with: pip install streamlit")
            return 1
    
    print("✅ Streamlit is available!")
    
    # Show options
    print("\n🎯 Choose an option:")
    print("  1. Run web interface locally")
    print("  2. Create Docker deployment setup")
    print("  3. Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        return run_web_interface()
    
    elif choice == "2":
        create_docker_setup()
        return 0
    
    elif choice == "3":
        create_docker_setup()
        print("\n" + "="*50)
        return run_web_interface()
    
    else:
        print("❌ Invalid choice. Please run the script again.")
        return 1


if __name__ == "__main__":
    exit(main())