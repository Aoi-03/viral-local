#!/usr/bin/env python3
"""
Test script to check available Gemini models.
"""

import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="your_gemini_api_key_here")

print("Available Gemini models:")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")

# Test a simple generation
print("\nTesting model access:")
try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, can you translate 'Hello world' to Hindi?")
    print("✅ gemini-pro works!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ gemini-pro failed: {e}")

try:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content("Hello, can you translate 'Hello world' to Hindi?")
    print("✅ gemini-1.5-pro-latest works!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ gemini-1.5-pro-latest failed: {e}")