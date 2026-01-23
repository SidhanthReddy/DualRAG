#!/usr/bin/env python3
"""
Simplified Integration Test with Detailed Error Reporting
"""

import os
import sys

print("🧪 Simplified Integration Test")
print("=" * 60)

# Detect provider
with open(".env", "r") as f:
    env_content = f.read()
    
if "OPENAI_API_KEY" in env_content and len(env_content.split("OPENAI_API_KEY=")[1].split()[0].strip()) > 10:
    provider = "openai"
elif "GEMINI_API_KEY" in env_content and len(env_content.split("GEMINI_API_KEY=")[1].split()[0].strip()) > 10:
    provider = "gemini"
else:
    provider = "mock"

print(f"Provider: {provider}\n")

from orchestrator import Orchestrator
from llm_output_parser import parse_llm_output, LLMOutputParseError

# Test 1: LLM Adapter
print("Step 1: Testing LLM Adapter")
print("-" * 60)

try:
    from llm_adapter import LLMAdapter
    llm = LLMAdapter(provider=provider)
    
    test_prompt = """SYSTEM:
You are a code generator. Output ONLY in this format:

FILE: components/Test.tsx
export default function Test() { return <div>Hello</div>; }

USER REQUEST: Create a test component
"""
    
    print("⏳ Calling LLM...")
    response = llm.generate(test_prompt)
    print(f"✅ LLM responded ({len(response)} chars)")
    print(f"\nRaw LLM Output:\n{'-'*40}")
    print(response)
    print(f"{'-'*40}\n")
    
except Exception as e:
    print(f"❌ LLM Adapter failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Possible fixes:")
    if provider == "openai":
        print("- Verify OpenAI API key in .env")
        print("- Check account has credits")
    elif provider == "gemini":
        print("- Verify Gemini API key in .env")
        print("- Check rate limits (15/min, 1500/day)")
    sys.exit(1)

# Test 2: LLM Output Parser
print("\nStep 2: Testing LLM Output Parser")
print("-" * 60)

try:
    artifacts = parse_llm_output(response)
    print(f"✅ Parsed {len(artifacts)} artifact(s)")
    
    for i, artifact in enumerate(artifacts, 1):
        print(f"\n  Artifact {i}:")
        print(f"    Path: {artifact.file_path}")
        print(f"    Language: {artifact.language}")
        print(f"    Content: {len(artifact.content)} chars")
        print(f"    Preview: {artifact.content[:60]}...")
    
except LLMOutputParseError as e:
    print(f"❌ Parser failed: {e}")
    print("\n💡 LLM output format is wrong. Expected format:")
    print("FILE: path/to/file.tsx")
    print("<file content>")
    print("\nActual output was shown above.")
    sys.exit(1)

except Exception as e:
    print(f"❌ Unexpected parser error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Full Orchestrator
print("\nStep 3: Testing Full Orchestrator")
print("-" * 60)

try:
    orchestrator = Orchestrator(llm_provider=provider)
    
    print("⏳ Sending request...")
    results = orchestrator.handle_request(
        user_request="Create a simple button component",
        allowed_paths=["components/Button.tsx"],
    )
    
    print(f"✅ Success! Created {len(results)} artifact(s)")
    
    for artifact in results:
        print(f"\n  📄 {artifact.file_path}")
        print(f"     Language: {artifact.language}")
        print(f"     Source: {artifact.source}")
        print(f"     Version: {artifact.version}")
        print(f"     Content: {len(artifact.content)} chars")
    
    print("\n" + "=" * 60)
    print("🎉 Full Integration Test PASSED!")
    print("=" * 60)

except Exception as e:
    print(f"❌ Orchestrator failed: {e}")
    print(f"\nError type: {type(e).__name__}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
    print("\n💡 Debugging steps:")
    print("1. Check if LLM output format matches expected format")
    print("2. Verify validation rules aren't too strict")
    print("3. Check file paths are in allowed_paths")
    print("4. Review full error traceback above")
    
    sys.exit(1)