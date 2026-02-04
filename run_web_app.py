#!/usr/bin/env python3
"""
Simple script to run the Viral-Local web application.
This fixes the import issues by running the web app as a module.
"""

import sys
import os
import subprocess

def main():
    """Run the Streamlit web application."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to Python path
    sys.path.insert(0, script_dir)
    
    # Run streamlit with the web app
    web_app_path = os.path.join(script_dir, "viral_local", "web_app.py")
    
    try:
        # Run streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            web_app_path, 
            "--server.port", "8502",
            "--server.headless", "true"
        ]
        
        print("🚀 Starting Viral-Local Web Interface...")
        print(f"📍 URL: http://localhost:8502")
        print("🔧 Make sure you have configured your Gemini API key in config.yaml")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Viral-Local Web Interface...")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify your config.yaml file has valid API keys")

if __name__ == "__main__":
    main()