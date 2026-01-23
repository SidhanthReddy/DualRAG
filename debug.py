#!/usr/bin/env python3
"""
Debug Full Integration Test Failure
"""

import os
import sys

print("🔍 Debugging Full Integration Test")
print("=" * 60)

# Check .env file
print("\n1. Checking .env configuration...")
if not os.path.exists(".env"):
    print("   ❌ .env file not found!")
    sys.exit(1)

with open(".env", "r") as f:
    env_content = f.read()
    
if "OPENAI_API_KEY" in env_content:
    provider = "openai"
    key_present = any(line.startswith("OPENAI_API_KEY=") and len(line.split("=", 1)[1].strip()) > 0 
                     for line in env_content.split("\n"))
elif "GEMINI_API_KEY" in env_content:
    provider = "gemini"
    key_present = any(line.startswith("GEMINI_API_KEY=") and len(line.split("=", 1)[1].strip()) > 0 
                     for line in env_content.split("\n"))
else:
    provider = "mock"
    key_present = False

print(f"   Provider: {provider}")
print(f"   API Key Present: {'✅' if key_present else '❌'}")

if not key_present and provider != "mock":
    print(f"\n   ⚠️  No API key found for {provider}")
    print(f"   Run: python setup_api.py")
    sys.exit(1)

# Test imports
print("\n2. Checking imports...")
try:
    from orchestrator import Orchestrator
    print("   ✅ orchestrator imported")
except Exception as e:
    print(f"   ❌ orchestrator import failed: {e}")
    sys.exit(1)

try:
    from llm_adapter import LLMAdapter
    print("   ✅ llm_adapter imported")
except Exception as e:
    print(f"   ❌ llm_adapter import failed: {e}")
    sys.exit(1)

# Test LLM adapter directly
print("\n3. Testing LLM adapter...")
try:
    llm = LLMAdapter(provider=provider)
    print(f"   ✅ LLM adapter created ({provider})")
except Exception as e:
    print(f"   ❌ LLM adapter creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test basic LLM call
print("\n4. Testing basic LLM call...")
try:
    prompt = "Say 'Hello World' in exactly those two words, nothing else."
    print(f"   Prompt: {prompt}")
    print("   ⏳ Calling LLM...")
    
    response = llm.generate(prompt)
    print(f"   ✅ LLM responded")
    print(f"   Response length: {len(response)} chars")
    print(f"   Response preview: {response[:100]}...")
    
except Exception as e:
    print(f"   ❌ LLM call failed: {e}")
    print(f"\n   Error type: {type(e).__name__}")
    print(f"   Error details: {str(e)}")
    
    if provider == "openai":
        print("\n   Troubleshooting OpenAI:")
        print("   - Check API key is valid: https://platform.openai.com/api-keys")
        print("   - Check you have credits: https://platform.openai.com/account/usage")
        print("   - Try: export OPENAI_API_KEY=sk-...")
    elif provider == "gemini":
        print("\n   Troubleshooting Gemini:")
        print("   - Check API key is valid: https://makersuite.google.com/app/apikey")
        print("   - Check free tier limits (15 req/min, 1500 req/day)")
        print("   - Try: export GEMINI_API_KEY=...")
    
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test orchestrator creation
print("\n5. Testing orchestrator...")
try:
    orchestrator = Orchestrator(llm_provider=provider)
    print(f"   ✅ Orchestrator created")
except Exception as e:
    print(f"   ❌ Orchestrator creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test full request flow
print("\n6. Testing full request flow...")
try:
    print("   ⏳ Sending request to orchestrator...")
    results = orchestrator.handle_request(
        user_request="Create a simple React component named HelloWorld in components/HelloWorld.tsx that returns a div with text 'Hello World'",
        allowed_paths=["components/HelloWorld.tsx"],
    )
    
    print(f"   ✅ Request succeeded!")
    print(f"   Created {len(results)} artifact(s):")
    for r in results:
        print(f"      - {r.file_path} ({len(r.content)} chars)")
        print(f"        Preview: {r.content[:80]}...")
    
except Exception as e:
    print(f"   ❌ Full request failed: {e}")
    print(f"\n   Error type: {type(e).__name__}")
    print(f"   Error details: {str(e)}")
    
    import traceback
    traceback.print_exc()
    
    print("\n   Possible causes:")
    print("   1. LLM output parsing failed (check llm_output_parser.py)")
    print("   2. Validation failed (check validator.py)")
    print("   3. LLM returned unexpected format")
    print("   4. API rate limit hit")
    
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All checks passed! Integration should work.")
print("=" * 60)