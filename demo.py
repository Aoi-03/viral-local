#!/usr/bin/env python3
"""
Demo script for Viral-Local system testing.
"""

import os
import sys
from pathlib import Path

def main():
    print("🎬 Viral-Local Demo & Testing Guide")
    print("=" * 50)
    
    print("\n📋 Available Testing Options:")
    print("1. 🖥️  Command Line Interface (CLI)")
    print("2. 🌐 Web Interface (Streamlit)")
    print("3. 🧪 Run Test Suite")
    print("4. ✅ Validate System Setup")
    print("5. 📖 View Documentation")
    
    while True:
        choice = input("\n👉 Choose an option (1-5, or 'q' to quit): ").strip()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            demo_cli()
        elif choice == '2':
            demo_web()
        elif choice == '3':
            run_tests()
        elif choice == '4':
            validate_system()
        elif choice == '5':
            show_docs()
        else:
            print("❌ Invalid choice. Please enter 1-5 or 'q'.")

def demo_cli():
    """Demo CLI interface."""
    print("\n🖥️ Command Line Interface Demo")
    print("-" * 30)
    
    print("📝 Example CLI commands:")
    print()
    print("# 1. Validate system setup:")
    print("python -m viral_local.main --validate --config config.yaml")
    print()
    print("# 2. Process a video to Hindi:")
    print('python -m viral_local.main "https://youtube.com/watch?v=VIDEO_ID" --target-lang hi')
    print()
    print("# 3. Process to Bengali with verbose output:")
    print('python -m viral_local.main "https://youtu.be/VIDEO_ID" -t bn --verbose')
    print()
    print("# 4. Get help:")
    print("python -m viral_local.main --help")
    
    if input("\n🚀 Run validation now? (y/n): ").lower() == 'y':
        print("\n⚡ Running system validation...")
        os.system("python -m viral_local.main --validate")

def demo_web():
    """Demo web interface."""
    print("\n🌐 Web Interface Demo")
    print("-" * 25)
    
    print("🚀 To launch the web interface:")
    print("1. Run: python run_web_app.py")
    print("2. Open browser to: http://localhost:8501")
    print("3. Configure API keys in sidebar")
    print("4. Enter YouTube URL and select language")
    print("5. Click 'Start Processing'")
    
    if input("\n🌐 Launch web interface now? (y/n): ").lower() == 'y':
        print("\n🚀 Starting web interface...")
        print("📱 Opening in your default browser...")
        print("🛑 Press Ctrl+C to stop the server")
        os.system("python run_web_app.py")

def run_tests():
    """Run the test suite."""
    print("\n🧪 Test Suite")
    print("-" * 15)
    
    print("Available test categories:")
    print("1. All tests (recommended)")
    print("2. Core functionality tests")
    print("3. System validation tests")
    print("4. Performance tests")
    print("5. Language support tests")
    
    choice = input("\n👉 Choose test category (1-5): ").strip()
    
    test_commands = {
        '1': "python -m pytest tests/ -v",
        '2': "python -m pytest tests/test_*.py -v -k 'not (validation or performance or language)'",
        '3': "python -m pytest tests/test_system_validation.py -v",
        '4': "python -m pytest tests/test_performance_validation.py -v",
        '5': "python -m pytest tests/test_language_validation.py -v"
    }
    
    if choice in test_commands:
        print(f"\n⚡ Running tests...")
        os.system(test_commands[choice])
    else:
        print("❌ Invalid choice.")

def validate_system():
    """Validate system setup."""
    print("\n✅ System Validation")
    print("-" * 20)
    
    print("🔍 Checking system requirements...")
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check if config exists
    if Path("config.yaml").exists():
        print("✅ Configuration file found")
        if input("🚀 Run full system validation? (y/n): ").lower() == 'y':
            os.system("python -m viral_local.main --validate --config config.yaml")
    else:
        print("⚠️  Configuration file not found")
        print("💡 Copy config.yaml.example to config.yaml and add your API keys")
        
        if input("📋 Create config file now? (y/n): ").lower() == 'y':
            import shutil
            shutil.copy("config.yaml.example", "config.yaml")
            print("✅ Created config.yaml - please edit it and add your API keys")

def show_docs():
    """Show documentation."""
    print("\n📖 Documentation")
    print("-" * 17)
    
    docs = {
        '1': ("Testing Guide", "TESTING_GUIDE.md"),
        '2': ("Validation Report", "VALIDATION_REPORT.md"),
        '3': ("Requirements", ".kiro/specs/viral-local/requirements.md"),
        '4': ("Design Document", ".kiro/specs/viral-local/design.md"),
        '5': ("Task List", ".kiro/specs/viral-local/tasks.md")
    }
    
    print("Available documentation:")
    for key, (name, _) in docs.items():
        print(f"{key}. {name}")
    
    choice = input("\n👉 Choose document to view (1-5): ").strip()
    
    if choice in docs:
        _, filepath = docs[choice]
        if Path(filepath).exists():
            print(f"\n📄 Opening {filepath}...")
            # Try to open with default system viewer
            if sys.platform.startswith('win'):
                os.system(f'start "" "{filepath}"')
            elif sys.platform.startswith('darwin'):
                os.system(f'open "{filepath}"')
            else:
                os.system(f'xdg-open "{filepath}"')
        else:
            print(f"❌ File not found: {filepath}")
    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main()