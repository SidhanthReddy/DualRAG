#!/usr/bin/env python3
"""
Quick API key setup for State RAG testing
"""

import os

print("🔑 State RAG API Key Setup")
print("=" * 60)
print("\nWhich LLM provider do you want to use?")
print("1. OpenAI (GPT-4o-mini) - Fast and cheap")
print("2. Google Gemini (gemini-2.5-flash) - Free tier available")
print("3. Mock (No API key needed, for testing only)")

choice = input("\nEnter choice (1/2/3): ").strip()

if choice == "1":
    print("\n📝 OpenAI Setup")
    print("Get your API key from: https://platform.openai.com/api-keys")
    api_key = input("Enter your OpenAI API key: ").strip()
    
    with open(".env", "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print("✅ OpenAI configured!")
    print("\nTo run tests: python test_state_rag.py")

elif choice == "2":
    print("\n📝 Google Gemini Setup")
    print("Get your API key from: https://makersuite.google.com/app/apikey")
    api_key = input("Enter your Gemini API key: ").strip()
    
    with open(".env", "w") as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")
    
    print("✅ Gemini configured!")
    print("\nTo run tests: python test_state_rag.py")

else:
    print("\n✅ Using mock provider (no API key needed)")
    print("Note: Mock provider returns hardcoded responses for testing")
    print("\nTo run tests: python test_state_rag.py")