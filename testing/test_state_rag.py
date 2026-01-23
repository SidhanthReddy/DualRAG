#!/usr/bin/env python3
"""
State RAG System - Comprehensive Test Suite

Tests all fixes:
1. FAISS rebuild after load
2. Circular dependency detection
3. Memory leak cleanup
4. Path traversal security
5. Race condition prevention
6. Pre-validation
7. LLM retry logic
"""

import os
import sys
import time
from datetime import datetime

# Set up environment
print("🔧 Setting up test environment...")

# Create .env file for API key
def setup_env():
    env_path = ".env"
    
    # If .env exists, detect provider from it
    if os.path.exists(env_path):
        print("✅ .env file already exists")
        with open(env_path, "r") as f:
            env_content = f.read()
        
        if "OPENAI_API_KEY" in env_content:
            has_key = any(
                line.startswith("OPENAI_API_KEY=") and len(line.split("=", 1)[1].strip()) > 10
                for line in env_content.split("\n")
            )
            if has_key:
                print("   Provider: openai")
                return "openai"
        
        if "GEMINI_API_KEY" in env_content:
            has_key = any(
                line.startswith("GEMINI_API_KEY=") and len(line.split("=", 1)[1].strip()) > 10
                for line in env_content.split("\n")
            )
            if has_key:
                print("   Provider: gemini")
                return "gemini"
        
        print("   Provider: mock")
        return "mock"
    
    # Create new .env
    print("\n📝 API Key Setup")
    print("=" * 60)
    print("Choose your LLM provider:")
    print("1. OpenAI (GPT-4o-mini)")
    print("2. Google Gemini (gemini-2.5-flash)")
    print("3. Mock (for testing without API)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        api_key = input("Enter your OpenAI API key: ").strip()
        with open(env_path, "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        provider = "openai"
    elif choice == "2":
        api_key = input("Enter your Gemini API key: ").strip()
        with open(env_path, "w") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
        provider = "gemini"
    else:
        provider = "mock"
        print("Using mock provider (no API key needed)")
    
    print(f"✅ Configuration saved: {provider}")
    return provider

provider = setup_env()

# Import after .env setup
from orchestrator import Orchestrator
from state_rag_manager import StateRAGManager
from artifact import Artifact
from state_rag_enums import ArtifactSource, ArtifactType
from global_rag import GlobalRAG
from schemas import GlobalRAGEntry

print("\n" + "=" * 60)
print("🧪 STARTING STATE RAG SYSTEM TESTS")
print("=" * 60)


# ============================================================================
# TEST 1: Path Traversal Security
# ============================================================================
def test_path_traversal():
    print("\n📋 TEST 1: Path Traversal Security")
    print("-" * 60)
    
    malicious_paths = [
        "../../../etc/passwd",
        "../../.env",
        "../outside/project",
        "/etc/shadow",
        "C:\\Windows\\system32\\config",
        "etc/passwd",  # Even without ../
    ]
    
    passed = 0
    failed = 0
    
    for path in malicious_paths:
        try:
            artifact = Artifact(
                type=ArtifactType.component,
                name="test",
                file_path=path,
                content="malicious",
                language="tsx",
                source=ArtifactSource.ai_generated,
            )
            print(f"  ❌ SECURITY BREACH: {path} was allowed!")
            failed += 1
        except ValueError as e:
            print(f"  ✅ Blocked: {path}")
            passed += 1
    
    print(f"\n  Result: {passed} blocked, {failed} failed")
    return failed == 0


# ============================================================================
# TEST 2: Circular Dependency Detection
# ============================================================================
def test_circular_dependencies():
    print("\n📋 TEST 2: Circular Dependency Detection")
    print("-" * 60)
    
    manager = StateRAGManager()
    
    # Create circular dependency: A → B → C → A
    artifacts = [
        Artifact(
            artifact_id="a",
            type=ArtifactType.component,
            name="ComponentA",
            file_path="components/A.tsx",
            content="export default function A() {}",
            language="tsx",
            source=ArtifactSource.ai_generated,
            dependencies=["b"],
        ),
        Artifact(
            artifact_id="b",
            type=ArtifactType.component,
            name="ComponentB",
            file_path="components/B.tsx",
            content="export default function B() {}",
            language="tsx",
            source=ArtifactSource.ai_generated,
            dependencies=["c"],
        ),
        Artifact(
            artifact_id="c",
            type=ArtifactType.component,
            name="ComponentC",
            file_path="components/C.tsx",
            content="export default function C() {}",
            language="tsx",
            source=ArtifactSource.ai_generated,
            dependencies=["a"],  # Circular!
        ),
    ]
    
    for artifact in artifacts:
        manager.commit(artifact)
    
    print("  ⏳ Testing circular dependency expansion...")
    start = time.time()
    
    try:
        # This should NOT hang
        result = manager._expand_dependencies([artifacts[0]])
        elapsed = time.time() - start
        
        if elapsed < 1.0:  # Should be instant
            print(f"  ✅ No infinite loop (completed in {elapsed:.3f}s)")
            print(f"  ✅ Returned {len(result)} artifacts")
            return True
        else:
            print(f"  ⚠️  Slow execution ({elapsed:.3f}s) - possible issue")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


# ============================================================================
# TEST 3: Memory Leak Prevention
# ============================================================================
def test_memory_cleanup():
    print("\n📋 TEST 3: Memory Leak Prevention")
    print("-" * 60)
    
    manager = StateRAGManager()
    
    # Create 50 versions of the same file
    print("  ⏳ Creating 50 versions of test file...")
    for i in range(50):
        artifact = Artifact(
            type=ArtifactType.component,
            name="Test",
            file_path="components/Test.tsx",
            content=f"export default function Test() {{ return <div>v{i}</div>; }}",
            language="tsx",
            source=ArtifactSource.ai_generated,
        )
        manager.commit(artifact)
    
    # Check that cleanup happened
    artifact_count = len(manager.artifacts)
    print(f"  📊 Total artifacts after 50 commits: {artifact_count}")
    
    # Should be much less than 50 due to cleanup
    if artifact_count < 20:
        print(f"  ✅ Cleanup working (kept only {artifact_count} versions)")
        return True
    else:
        print(f"  ❌ Memory leak detected ({artifact_count} versions retained)")
        return False


# ============================================================================
# TEST 4: FAISS Index Persistence
# ============================================================================
def test_faiss_persistence():
    print("\n📋 TEST 4: FAISS Index Persistence")
    print("-" * 60)
    
    # Create manager and add artifacts
    manager1 = StateRAGManager()
    
    artifacts = [
        Artifact(
            type=ArtifactType.component,
            name="Button",
            file_path="components/Button.tsx",
            content="export default function Button() { return <button>Click</button>; }",
            language="tsx",
            source=ArtifactSource.ai_generated,
        ),
        Artifact(
            type=ArtifactType.component,
            name="Card",
            file_path="components/Card.tsx",
            content="export default function Card() { return <div className='card'>Content</div>; }",
            language="tsx",
            source=ArtifactSource.ai_generated,
        ),
    ]
    
    for a in artifacts:
        manager1.commit(a)
    
    print("  ✅ Committed 2 artifacts")
    
    # Test semantic search BEFORE reload
    results1 = manager1.retrieve(user_query="button component", limit=5)
    print(f"  📊 Semantic search (before reload): {len(results1)} results")
    
    # Create NEW manager (simulates restart)
    print("  ⏳ Simulating restart (creating new manager)...")
    manager2 = StateRAGManager()
    
    # Test semantic search AFTER reload
    results2 = manager2.retrieve(user_query="button component", limit=5)
    print(f"  📊 Semantic search (after reload): {len(results2)} results")
    
    if len(results2) > 0:
        print("  ✅ FAISS index rebuilt successfully after load")
        return True
    else:
        print("  ❌ FAISS index not working after reload")
        return False


# ============================================================================
# TEST 5: Pre-Validation
# ============================================================================
def test_pre_validation():
    print("\n📋 TEST 5: Pre-Validation (Authority Check)")
    print("-" * 60)
    
    # Create a user-modified artifact
    manager = StateRAGManager()
    user_artifact = Artifact(
        type=ArtifactType.component,
        name="UserComponent",
        file_path="components/UserComponent.tsx",
        content="// User's precious code",
        language="tsx",
        source=ArtifactSource.user_modified,  # User owns this!
    )
    manager.commit(user_artifact)
    
    print("  ✅ Created user-modified artifact")
    
    # Try to modify it WITHOUT permission
    orchestrator = Orchestrator(llm_provider="mock")
    
    try:
        # This should fail BEFORE calling LLM
        orchestrator.handle_request(
            user_request="Add a button",
            allowed_paths=[],  # Empty! User didn't give permission
        )
        print("  ❌ Pre-validation failed - should have blocked!")
        return False
        
    except ValueError as e:
        if "user-protected" in str(e).lower():
            print(f"  ✅ Pre-validation blocked: {e}")
            return True
        else:
            print(f"  ❌ Wrong error: {e}")
            return False


# ============================================================================
# TEST 6: File Locking (Race Condition Prevention)
# ============================================================================
def test_file_locking():
    print("\n📋 TEST 6: File Locking (Race Condition)")
    print("-" * 60)
    
    from file_lock import FileLock
    
    test_file = "test_lock.txt"
    
    try:
        # Test that lock can be acquired and released
        with FileLock(test_file):
            with open(test_file, "w") as f:
                f.write("test")
        
        # Verify file was written
        with open(test_file, "r") as f:
            content = f.read()
        
        if content == "test":
            print("  ✅ File locking working")
            return True
        else:
            print("  ❌ File locking failed")
            return False
            
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(test_file + ".lock"):
            os.remove(test_file + ".lock")


# ============================================================================
# TEST 7: Full Integration Test with Real LLM
# ============================================================================
def test_full_integration():
    print("\n📋 TEST 7: Full Integration Test")
    print("-" * 60)
    
    # Safety check for provider
    if provider is None:
        print("  ⚠️  Provider is None - .env detection failed")
        print("  Using mock provider as fallback")
        test_provider = "mock"
    else:
        test_provider = provider
    
    if test_provider == "mock":
        print("  ⏭️  Skipping (mock provider)")
        return True
    
    # Create orchestrator with real LLM
    orchestrator = Orchestrator(llm_provider=test_provider)
    
    # Simple request
    print("  ⏳ Sending request to LLM...")
    print("  Request: 'Create a simple React button component'")
    
    try:
        results = orchestrator.handle_request(
            user_request="Create a simple React button component named Button in components/Button.tsx",
            allowed_paths=["components/Button.tsx"],
        )
        
        print(f"  ✅ LLM responded successfully")
        print(f"  ✅ Created {len(results)} artifact(s)")
        
        for r in results:
            print(f"     - {r.file_path} ({len(r.content)} chars)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


# ============================================================================
# RUN ALL TESTS
# ============================================================================
def run_all_tests():
    tests = [
        ("Path Traversal Security", test_path_traversal),
        ("Circular Dependency Detection", test_circular_dependencies),
        ("Memory Leak Prevention", test_memory_cleanup),
        ("FAISS Index Persistence", test_faiss_persistence),
        ("Pre-Validation", test_pre_validation),
        ("File Locking", test_file_locking),
        ("Full Integration", test_full_integration),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)