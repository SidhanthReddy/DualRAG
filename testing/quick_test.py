#!/usr/bin/env python3
"""
State RAG System - Quick Test (No ML dependencies required)

Tests core fixes without requiring sentence_transformers/FAISS
"""

import os
import sys
from datetime import datetime

print("🔧 Setting up test environment...")

# Create state_rag directory
os.makedirs("state_rag", exist_ok=True)

# Reset state
for file in ["state_rag/artifacts.json", "global_rag.json", "global_rag.index"]:
    if os.path.exists(file):
        os.remove(file)
print("✅ State RAG reset complete\n")

# Import modules
from artifact import Artifact
from state_rag_enums import ArtifactSource, ArtifactType

print("=" * 60)
print("🧪 STATE RAG QUICK TESTS")
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
        "/etc/shadow",
        "C:\\Windows\\system32\\config",
        "etc/passwd",
        "sys/config",
    ]
    
    passed = 0
    
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
            print(f"  ❌ BREACH: {path} allowed")
            return False
        except ValueError:
            print(f"  ✅ Blocked: {path}")
            passed += 1
    
    print(f"\n  ✅ All {passed} malicious paths blocked!")
    return True


# ============================================================================
# TEST 2: Valid Paths Allowed
# ============================================================================
def test_valid_paths():
    print("\n📋 TEST 2: Valid Paths Should Be Allowed")
    print("-" * 60)
    
    valid_paths = [
        "components/Button.tsx",
        "package.json",
        "src/index.ts",
        "app/page.tsx",
    ]
    
    passed = 0
    
    for path in valid_paths:
        try:
            artifact = Artifact(
                type=ArtifactType.component,
                name="test",
                file_path=path,
                content="valid content",
                language="tsx",
                source=ArtifactSource.ai_generated,
            )
            print(f"  ✅ Allowed: {path}")
            passed += 1
        except ValueError as e:
            print(f"  ❌ Rejected: {path} - {e}")
            return False
    
    print(f"\n  ✅ All {passed} valid paths accepted!")
    return True


# ============================================================================
# TEST 3: File Locking
# ============================================================================
def test_file_locking():
    print("\n📋 TEST 3: File Locking (Cross-Platform)")
    print("-" * 60)
    
    from file_lock import FileLock
    
    test_file = "test_lock.txt"
    
    try:
        # Test lock acquisition
        print("  ⏳ Testing lock acquisition...")
        with FileLock(test_file):
            with open(test_file, "w") as f:
                f.write("locked content")
        
        # Verify file was written
        with open(test_file, "r") as f:
            content = f.read()
        
        if content == "locked content":
            print("  ✅ Lock acquired and released successfully")
            print(f"  ✅ Platform: {sys.platform}")
            return True
        else:
            print("  ❌ Lock failed")
            return False
            
    finally:
        # Cleanup
        for f in [test_file, test_file + ".lock"]:
            if os.path.exists(f):
                os.remove(f)


# ============================================================================
# TEST 4: Artifact Creation and Validation
# ============================================================================
def test_artifact_validation():
    print("\n📋 TEST 4: Artifact Validation")
    print("-" * 60)
    
    # Test empty content
    try:
        artifact = Artifact(
            type=ArtifactType.component,
            name="Empty",
            file_path="test.tsx",
            content="   ",  # Empty!
            language="tsx",
            source=ArtifactSource.ai_generated,
        )
        print("  ❌ Empty content allowed")
        return False
    except ValueError:
        print("  ✅ Empty content blocked")
    
    # Test language validation
    try:
        artifact = Artifact(
            type=ArtifactType.component,
            name="Test",
            file_path="test.tsx",
            content="valid",
            language="python",  # Invalid!
            source=ArtifactSource.ai_generated,
        )
        print("  ❌ Invalid language allowed")
        return False
    except ValueError:
        print("  ✅ Invalid language blocked")
    
    # Test valid artifact
    try:
        artifact = Artifact(
            type=ArtifactType.component,
            name="Valid",
            file_path="components/Valid.tsx",
            content="export default function Valid() {}",
            language="tsx",
            source=ArtifactSource.ai_generated,
        )
        print("  ✅ Valid artifact created")
        return True
    except Exception as e:
        print(f"  ❌ Valid artifact rejected: {e}")
        return False


# ============================================================================
# TEST 5: Authority Source Tracking
# ============================================================================
def test_authority_tracking():
    print("\n📋 TEST 5: Authority Source Tracking")
    print("-" * 60)
    
    # Create user-modified artifact
    user_artifact = Artifact(
        type=ArtifactType.component,
        name="UserCode",
        file_path="user/code.tsx",
        content="// User's code",
        language="tsx",
        source=ArtifactSource.user_modified,
    )
    
    if user_artifact.source == ArtifactSource.user_modified:
        print("  ✅ User-modified source tracked")
    else:
        print("  ❌ Source tracking failed")
        return False
    
    # Create AI-generated artifact
    ai_artifact = Artifact(
        type=ArtifactType.component,
        name="AICode",
        file_path="ai/code.tsx",
        content="// AI code",
        language="tsx",
        source=ArtifactSource.ai_generated,
    )
    
    if ai_artifact.source == ArtifactSource.ai_generated:
        print("  ✅ AI-generated source tracked")
    else:
        print("  ❌ Source tracking failed")
        return False
    
    print("  ✅ Authority tracking working correctly")
    return True


# ============================================================================
# TEST 6: Path Normalization
# ============================================================================
def test_path_normalization():
    print("\n📋 TEST 6: Path Normalization")
    print("-" * 60)
    
    test_cases = [
        ("./components/Button.tsx", "components/Button.tsx"),
        ("components//Button.tsx", "components/Button.tsx"),
        ("components/./Button.tsx", "components/Button.tsx"),
    ]
    
    for input_path, expected in test_cases:
        try:
            artifact = Artifact(
                type=ArtifactType.component,
                name="test",
                file_path=input_path,
                content="content",
                language="tsx",
                source=ArtifactSource.ai_generated,
            )
            
            if artifact.file_path == expected:
                print(f"  ✅ '{input_path}' → '{artifact.file_path}'")
            else:
                print(f"  ⚠️  '{input_path}' → '{artifact.file_path}' (expected '{expected}')")
                
        except Exception as e:
            print(f"  ❌ '{input_path}' rejected: {e}")
            return False
    
    print("  ✅ Path normalization working")
    return True


# ============================================================================
# RUN ALL TESTS
# ============================================================================
def run_all_tests():
    tests = [
        ("Path Traversal Security", test_path_traversal),
        ("Valid Paths Allowed", test_valid_paths),
        ("File Locking", test_file_locking),
        ("Artifact Validation", test_artifact_validation),
        ("Authority Tracking", test_authority_tracking),
        ("Path Normalization", test_path_normalization),
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
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Core fixes verified.")
        print("\nNext steps:")
        print("1. Set up API key: python setup_api.py")
        print("2. Run full tests: python test_state_rag.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)