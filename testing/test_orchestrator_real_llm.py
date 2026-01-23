"""
Rate-Limited Integration Test for DualRAG

This version is designed to work with Gemini's FREE TIER limits:
- 15 requests per minute (RPM)
- 1,500 requests per day (RPD)
- 1 million tokens per minute (TPM)

Strategy:
- Adds delays between tests to respect rate limits
- Uses shorter prompts to reduce token usage
- Provides clear progress indicators
- Handles 429 errors gracefully
"""

import os
import sys
import json
import time
from typing import List, Dict, Optional
from datetime import datetime

print("=" * 70)
print("DUALRAG INTEGRATION TEST - GEMINI FREE TIER EDITION")
print("=" * 70)
print("\n⏱️  This test suite respects Gemini's rate limits:")
print("   - 15 requests/minute")
print("   - 4-5 second delay between tests")
print("   - Total runtime: ~2-3 minutes\n")

# Add uploads directory to import your modules
sys.path.insert(0, '/mnt/user-data/uploads')

try:
    from orchestrator import Orchestrator
    from state_rag_manager import StateRAGManager
    from global_rag import GlobalRAG
    from artifact import Artifact
    from state_rag_enums import ArtifactType, ArtifactSource
    from llm_adapter import LLMAdapter
    from schemas import GlobalRAGEntry
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure these files exist in /mnt/user-data/uploads/:")
    print("  - orchestrator.py")
    print("  - state_rag_manager.py")
    print("  - global_rag.py")
    print("  - artifact.py")
    print("  - state_rag_enums.py")
    print("  - llm_adapter.py")
    print("  - schemas.py")
    print("  - validator.py")
    sys.exit(1)


class RateLimitedIntegrationTest:
    """Integration tests with rate limit awareness"""
    
    def __init__(self, llm_provider: str = "gemini"):
        self.llm_provider = llm_provider
        self.orchestrator = None
        self.test_results = []
        self.test_dir = "/integration_tests"
        self.delay_between_tests = 5  # 5 seconds between tests
        
        os.makedirs(self.test_dir, exist_ok=True)
    
    def setup(self):
        """Set up fresh environment"""
        print("🔧 Setting up environment...")
        
        self.orchestrator = Orchestrator(llm_provider=self.llm_provider)
        self.orchestrator.state_rag.artifacts = []
        self.orchestrator.state_rag._persist()
        
        # Add minimal global patterns
        patterns = [
            GlobalRAGEntry(
                id="navbar",
                category="component",
                title="Navbar Pattern",
                content="Sticky navbar: use 'sticky top-0 z-50 bg-white'. Include logo, links, CTA button.",
                tags=["navbar"],
                framework="react",
                styling="tailwind"
            ),
            GlobalRAGEntry(
                id="button",
                category="component",
                title="Button Pattern",
                content="Reusable button with primary/secondary variants. Use Tailwind classes.",
                tags=["button"],
                framework="react",
                styling="tailwind"
            )
        ]
        
        for pattern in patterns:
            self.orchestrator.global_rag.ingest(pattern)
        
        print("✅ Setup complete\n")
    
    def _rate_limit_delay(self, test_number: int, total_tests: int):
        """Wait between tests to respect rate limits"""
        if test_number < total_tests:
            print(f"\n⏱️  Waiting {self.delay_between_tests}s to respect rate limits...")
            time.sleep(self.delay_between_tests)
    
    def test_1_create_navbar(self):
        """Test 1: Create navbar component"""
        print("=" * 70)
        print("TEST 1/3: Create Navbar Component")
        print("=" * 70)
        
        try:
            # SHORT prompt to save tokens
            user_request = "Create a navbar with logo, Home/About links, and Hire button. Use Tailwind."
            
            print(f"📝 Request: {user_request}")
            print("🔄 Generating...")
            
            start = time.time()
            committed = self.orchestrator.handle_request(
                user_request=user_request,
                allowed_paths=["components/Navbar.tsx"]
            )
            elapsed = time.time() - start
            
            if not committed:
                raise Exception("No component generated")
            
            navbar = committed[0]
            
            # Save
            with open(f"{self.test_dir}/navbar.tsx", 'w') as f:
                f.write(navbar.content)
            
            # Basic quality checks
            has_export = "export" in navbar.content
            has_nav = "<nav" in navbar.content.lower()
            has_jsx = "<" in navbar.content and ">" in navbar.content
            
            print(f"\n✅ Generated: {navbar.file_path}")
            print(f"   Lines: {len(navbar.content.splitlines())}")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Export: {'✅' if has_export else '❌'}")
            print(f"   Has <nav>: {'✅' if has_nav else '❌'}")
            print(f"   Has JSX: {'✅' if has_jsx else '❌'}")
            
            success = has_export and has_nav and has_jsx
            
            self.test_results.append({
                "test": "Create Navbar",
                "success": success,
                "time": elapsed
            })
            
            return success
            
        except Exception as e:
            print(f"\n❌ FAILED: {str(e)}")
            self.test_results.append({
                "test": "Create Navbar",
                "success": False,
                "error": str(e)
            })
            return False
    
    def test_2_update_navbar(self):
        """Test 2: Update existing component"""
        print("=" * 70)
        print("TEST 2/3: Update Navbar Component")
        print("=" * 70)
        
        try:
            # Check existing version
            existing = self.orchestrator.state_rag.retrieve(
                file_paths=["components/Navbar.tsx"]
            )
            
            if not existing:
                raise Exception("No existing navbar found")
            
            old_version = existing[0].version
            print(f"📄 Current version: {old_version}")
            
            # SHORT update request
            user_request = "Add a dark mode toggle button to the navbar"
            
            print(f"📝 Request: {user_request}")
            print("🔄 Updating...")
            
            start = time.time()
            committed = self.orchestrator.handle_request(
                user_request=user_request,
                allowed_paths=["components/Navbar.tsx"]
            )
            elapsed = time.time() - start
            
            if not committed:
                raise Exception("No update generated")
            
            updated = committed[0]
            new_version = updated.version
            
            # Save
            with open(f"{self.test_dir}/navbar_v{new_version}.tsx", 'w') as f:
                f.write(updated.content)
            
            version_incremented = new_version == old_version + 1
            has_dark_mode = "dark" in updated.content.lower() or "theme" in updated.content.lower()
            
            print(f"\n✅ Updated: {updated.file_path}")
            print(f"   Version: {old_version} → {new_version}")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Version incremented: {'✅' if version_incremented else '❌'}")
            print(f"   Has dark mode: {'✅' if has_dark_mode else '❌'}")
            
            success = version_incremented and has_dark_mode
            
            self.test_results.append({
                "test": "Update Navbar",
                "success": success,
                "time": elapsed
            })
            
            return success
            
        except Exception as e:
            print(f"\n❌ FAILED: {str(e)}")
            self.test_results.append({
                "test": "Update Navbar",
                "success": False,
                "error": str(e)
            })
            return False
    
    def test_3_create_button(self):
        """Test 3: Create button component"""
        print("=" * 70)
        print("TEST 3/3: Create Button Component")
        print("=" * 70)
        
        try:
            # SHORT prompt
            user_request = "Create a button component with primary/secondary variants. Use Tailwind."
            
            print(f"📝 Request: {user_request}")
            print("🔄 Generating...")
            
            start = time.time()
            committed = self.orchestrator.handle_request(
                user_request=user_request,
                allowed_paths=["components/Button.tsx"]
            )
            elapsed = time.time() - start
            
            if not committed:
                raise Exception("No component generated")
            
            button = committed[0]
            
            # Save
            with open(f"{self.test_dir}/button.tsx", 'w') as f:
                f.write(button.content)
            
            # Quality checks
            has_export = "export" in button.content
            has_button = "<button" in button.content.lower()
            has_variant = "variant" in button.content.lower() or "primary" in button.content.lower()
            
            print(f"\n✅ Generated: {button.file_path}")
            print(f"   Lines: {len(button.content.splitlines())}")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Export: {'✅' if has_export else '❌'}")
            print(f"   Has <button>: {'✅' if has_button else '❌'}")
            print(f"   Has variants: {'✅' if has_variant else '❌'}")
            
            success = has_export and has_button and has_variant
            
            self.test_results.append({
                "test": "Create Button",
                "success": success,
                "time": elapsed
            })
            
            return success
            
        except Exception as e:
            print(f"\n❌ FAILED: {str(e)}")
            self.test_results.append({
                "test": "Create Button",
                "success": False,
                "error": str(e)
            })
            return False
    
    def generate_report(self):
        """Generate final report"""
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        pass_rate = passed / total if total > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"   Tests run: {total}")
        print(f"   Passed: {passed} ✅")
        print(f"   Failed: {total - passed} ❌")
        print(f"   Pass rate: {pass_rate:.0%}")
        
        print(f"\n📋 Details:")
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            time_str = f"({result.get('time', 0):.1f}s)" if 'time' in result else ""
            print(f"   {status} - {result['test']} {time_str}")
            if "error" in result:
                print(f"        Error: {result['error']}")
        
        # Save report
        report_path = f"{self.test_dir}/test_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "pass_rate": pass_rate,
                    "timestamp": datetime.now().isoformat()
                },
                "results": self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Report saved: {report_path}")
        
        if pass_rate >= 0.66:  # 2 out of 3 is acceptable
            print("\n✅ TEST SUITE PASSED")
        else:
            print("\n❌ TEST SUITE FAILED")
        
        return pass_rate
    
    def run_all(self):
        """Run all tests with rate limiting"""
        print("\n" + "=" * 70)
        print("STARTING TEST SUITE (3 tests)")
        print("=" * 70)
        print()
        
        self.setup()
        
        # Test 1
        self.test_1_create_navbar()
        self._rate_limit_delay(1, 3)
        
        # Test 2
        self.test_2_update_navbar()
        self._rate_limit_delay(2, 3)
        
        # Test 3
        self.test_3_create_button()
        
        # Report
        pass_rate = self.generate_report()
        
        return pass_rate


def main():
    """Main entry point"""
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        print("\nPlease set your API key:")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nOr run with mock LLM:")
        print("  python integration_test_lite.py mock")
        return
    
    print(f"✅ API key found: ...{api_key[-8:]}")
    print()
    
    # Confirm
    print("⚠️  This will make 3 real API calls to Gemini.")
    print("   Estimated time: ~30 seconds")
    print("   Cost: FREE (uses free tier quota)")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run tests
    suite = RateLimitedIntegrationTest(llm_provider="gemini")
    pass_rate = suite.run_all()
    
    # Exit code
    sys.exit(0 if pass_rate >= 0.66 else 1)


if __name__ == "__main__":
    main()