"""
Complete Website Build Test with Full Prompt Visualization

This test:
1. Builds a complete website (multiple components)
2. Shows EXACTLY what prompts are sent to the LLM
3. Visualizes State RAG vs Global RAG separation
4. Demonstrates iterative modifications
5. Saves all prompts and responses for analysis

Perfect for demos and understanding how DualRAG works!
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict

# Add uploads directory
sys.path.insert(0, '/mnt/user-data/uploads')

# Load environment
from dotenv import load_dotenv
load_dotenv()

from orchestrator import Orchestrator
from state_rag_manager import StateRAGManager
from global_rag import GlobalRAG
from artifact import Artifact
from state_rag_enums import ArtifactType, ArtifactSource
from schemas import GlobalRAGEntry


class WebsiteBuildDemo:
    """
    Demonstrates complete website building with full visibility
    into the DualRAG process
    """
    
    def __init__(self, llm_provider: str = "gemini"):
        self.llm_provider = llm_provider
        self.orchestrator = None
        self.demo_dir = "/testing/website_build_demo"
        self.prompts = []  # Store all prompts sent
        
        os.makedirs(self.demo_dir, exist_ok=True)
        os.makedirs(f"{self.demo_dir}/prompts", exist_ok=True)
        os.makedirs(f"{self.demo_dir}/components", exist_ok=True)
        
        print("=" * 80)
        print("DUALRAG WEBSITE BUILD DEMO - FULL PROMPT VISUALIZATION")
        print("=" * 80)
        print()
    
    def setup(self):
        """Initialize orchestrator and seed Global RAG"""
        print("🔧 Setting up DualRAG system...\n")
        
        # Create orchestrator
        self.orchestrator = Orchestrator(llm_provider=self.llm_provider)
        
        # Clear state
        self.orchestrator.state_rag.artifacts = []
        self.orchestrator.state_rag._persist()
        
        # Seed Global RAG with patterns
        patterns = [
            GlobalRAGEntry(
                id="hero_pattern",
                category="component",
                title="Hero Section Pattern",
                content="Hero section with headline, subheadline, CTA button, and background. Use h1 for headline, p for subtitle. Make it visually striking with large text and good spacing.",
                tags=["hero", "landing"],
                framework="react",
                styling="tailwind"
            ),
            GlobalRAGEntry(
                id="navbar_pattern",
                category="component",
                title="Navbar Pattern",
                content="Sticky navbar with logo on left, nav links in center, CTA on right. Use 'sticky top-0 z-50'. Include shadow for depth. Make it responsive with mobile menu.",
                tags=["navbar", "navigation"],
                framework="react",
                styling="tailwind"
            ),
            GlobalRAGEntry(
                id="carousel_pattern",
                category="component",
                title="Carousel/Slider Pattern",
                content="Image carousel with prev/next buttons and indicators. Use useState for current slide. Include smooth transitions with transform. Auto-advance optional.",
                tags=["carousel", "slider"],
                framework="react",
                styling="tailwind"
            ),
            GlobalRAGEntry(
                id="footer_pattern",
                category="component",
                title="Footer Pattern",
                content="Footer with multiple columns: company info, links, social media. Use grid layout. Include copyright notice. Dark background with light text works well.",
                tags=["footer"],
                framework="react",
                styling="tailwind"
            )
        ]
        
        for pattern in patterns:
            self.orchestrator.global_rag.ingest(pattern)
        
        print(f"✅ Global RAG seeded with {len(patterns)} patterns")
        print(f"✅ State RAG initialized (empty)")
        print()
    
    def _visualize_prompt_construction(
        self, 
        iteration: int,
        user_request: str,
        allowed_paths: List[str]
    ):
        """
        Show exactly how the prompt is constructed from:
        - User request
        - State RAG (authoritative project state)
        - Global RAG (advisory patterns)
        """
        
        print("=" * 80)
        print(f"ITERATION {iteration}: PROMPT CONSTRUCTION VISUALIZATION")
        print("=" * 80)
        print()
        
        # Step 1: Show user request
        print("📝 USER REQUEST:")
        print("─" * 80)
        print(user_request)
        print()
        
        # Step 2: Retrieve from State RAG
        print("🗄️  STATE RAG RETRIEVAL (Authoritative Project State):")
        print("─" * 80)
        
        active_artifacts = self.orchestrator.state_rag.retrieve(
            file_paths=allowed_paths
        )
        
        if active_artifacts:
            print(f"Found {len(active_artifacts)} active components:\n")
            for artifact in active_artifacts:
                print(f"   📄 {artifact.file_path}")
                print(f"      Version: {artifact.version}")
                print(f"      Source: {artifact.source}")
                print(f"      Lines: {len(artifact.content.splitlines())}")
                print(f"      Preview: {artifact.content[:100]}...")
                print()
        else:
            print("   (No existing components - starting fresh)\n")
        
        # Step 3: Retrieve from Global RAG
        print("📚 GLOBAL RAG RETRIEVAL (Advisory Best Practices):")
        print("─" * 80)
        
        global_refs = self.orchestrator.global_rag.retrieve(
            query=user_request,
            k=3
        )
        
        if global_refs:
            print(f"Retrieved {len(global_refs)} relevant patterns:\n")
            for i, ref in enumerate(global_refs, 1):
                print(f"   {i}. {ref.title}")
                print(f"      Category: {ref.category}")
                print(f"      Content: {ref.content[:100]}...")
                print()
        else:
            print("   (No relevant patterns found)\n")
        
        # Step 4: Show final prompt structure
        print("🔨 FINAL PROMPT SENT TO LLM:")
        print("─" * 80)
        
        # Build the actual prompt (using orchestrator's method)
        prompt = self.orchestrator._build_prompt(
            user_request=user_request,
            active_artifacts=active_artifacts,
            global_refs=global_refs,
            allowed_paths=allowed_paths
        )
        
        # Save prompt to file
        prompt_file = f"{self.demo_dir}/prompts/iteration_{iteration}_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        
        print(prompt)
        print()
        print(f"💾 Full prompt saved to: {prompt_file}")
        print()
        
        # Show prompt structure analysis
        print("📊 PROMPT STRUCTURE ANALYSIS:")
        print("─" * 80)
        
        lines = prompt.split('\n')
        total_lines = len(lines)
        
        # Count sections
        system_lines = len([l for l in lines if 'SYSTEM:' in l or 'stateless' in l.lower()])
        state_lines = len([l for l in lines if 'PROJECT STATE' in l or 'AUTHORITATIVE' in l])
        global_lines = len([l for l in lines if 'GLOBAL REFERENCES' in l or 'ADVISORY' in l])
        request_lines = len([l for l in lines if 'USER REQUEST' in l])
        
        print(f"   Total lines: {total_lines}")
        print(f"   System instructions: ~{system_lines} lines")
        print(f"   State RAG content: ~{len([a.content for a in active_artifacts])} artifacts")
        print(f"   Global RAG content: ~{len(global_refs)} patterns")
        print(f"   User request: ~{request_lines} lines")
        print()
        
        # Show authority boundaries
        print("🛡️  AUTHORITY BOUNDARIES:")
        print("─" * 80)
        print("   ✅ State RAG = AUTHORITATIVE (must be respected)")
        print("   ℹ️  Global RAG = ADVISORY (suggestions only)")
        print("   📝 User Request = TASK (what to accomplish)")
        print()
        
        return prompt
    
    def _execute_and_show_results(
        self,
        iteration: int,
        user_request: str,
        allowed_paths: List[str],
        prompt: str
    ):
        """Execute the request and show results"""
        
        print("⚙️  EXECUTING REQUEST...")
        print("─" * 80)
        
        start_time = time.time()
        
        # Execute
        try:
            committed_artifacts = self.orchestrator.handle_request(
                user_request=user_request,
                allowed_paths=allowed_paths
            )
            
            elapsed = time.time() - start_time
            
            print(f"✅ Execution completed in {elapsed:.2f}s")
            print()
            
            # Show results
            print("📦 RESULTS:")
            print("─" * 80)
            
            if committed_artifacts:
                print(f"Generated {len(committed_artifacts)} artifacts:\n")
                
                for artifact in committed_artifacts:
                    print(f"   ✅ {artifact.file_path}")
                    print(f"      Type: {artifact.type}")
                    print(f"      Version: {artifact.version}")
                    print(f"      Lines: {len(artifact.content.splitlines())}")
                    print(f"      Source: {artifact.source}")
                    
                    # Save to file
                    filename = artifact.file_path.replace("/", "_")
                    save_path = f"{self.demo_dir}/components/{filename}"
                    with open(save_path, 'w') as f:
                        f.write(artifact.content)
                    print(f"      Saved: {save_path}")
                    print()
                
                # Save iteration summary
                summary = {
                    "iteration": iteration,
                    "user_request": user_request,
                    "execution_time": elapsed,
                    "artifacts_created": [
                        {
                            "file_path": a.file_path,
                            "version": a.version,
                            "lines": len(a.content.splitlines()),
                            "source": str(a.source)
                        }
                        for a in committed_artifacts
                    ]
                }
                
                summary_file = f"{self.demo_dir}/prompts/iteration_{iteration}_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"💾 Summary saved to: {summary_file}")
                print()
                
                return committed_artifacts
            else:
                print("❌ No artifacts generated")
                return []
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def iteration_1_initial_website(self):
        """
        Iteration 1: Create initial website with navbar and carousel
        """
        
        print("\n" + "🌟" * 40)
        print("ITERATION 1: Create Initial Website")
        print("🌟" * 40 + "\n")
        
        user_request = """
        Create a landing page for a modern tech startup.
        
        Components needed:
        1. Navbar - with logo "TechCo", navigation links (Home, Features, Pricing, Contact), and "Get Started" button
        2. Carousel - image slider showcasing 3 product features with captions
        
        Make it modern, clean, and professional with Tailwind CSS.
        """
        
        allowed_paths = [
            "components/Navbar.tsx",
            "components/Carousel.tsx"
        ]
        
        # Visualize prompt construction
        prompt = self._visualize_prompt_construction(
            iteration=1,
            user_request=user_request,
            allowed_paths=allowed_paths
        )
        
        # Execute and show results
        artifacts = self._execute_and_show_results(
            iteration=1,
            user_request=user_request,
            allowed_paths=allowed_paths,
            prompt=prompt
        )
        
        # Wait before next iteration
        print("\n⏸️  Pausing 5 seconds before next iteration...")
        time.sleep(5)
        
        return artifacts
    
    def iteration_2_modify_navbar(self):
        """
        Iteration 2: Modify the navbar to add dark mode
        
        THIS IS WHERE WE SEE STATE RAG IN ACTION!
        The navbar from iteration 1 will appear in the prompt as AUTHORITATIVE state.
        """
        
        print("\n" + "🌟" * 40)
        print("ITERATION 2: Modify Navbar (Add Dark Mode)")
        print("🌟" * 40 + "\n")
        
        user_request = """
        Update the navbar to add a dark mode toggle button.
        Place it between the navigation links and the "Get Started" button.
        The button should show a moon icon and toggle dark mode on click.
        """
        
        allowed_paths = [
            "components/Navbar.tsx"  # Only modify navbar
        ]
        
        print("🔍 KEY OBSERVATION:")
        print("─" * 80)
        print("   Watch how the EXISTING navbar from Iteration 1 appears in")
        print("   the prompt as AUTHORITATIVE state that the LLM must respect!")
        print()
        
        # Visualize prompt construction
        prompt = self._visualize_prompt_construction(
            iteration=2,
            user_request=user_request,
            allowed_paths=allowed_paths
        )
        
        # Execute and show results
        artifacts = self._execute_and_show_results(
            iteration=2,
            user_request=user_request,
            allowed_paths=allowed_paths,
            prompt=prompt
        )
        
        # Show versioning
        print("🔄 VERSIONING ANALYSIS:")
        print("─" * 80)
        
        all_navbar_versions = [
            a for a in self.orchestrator.state_rag.artifacts
            if a.file_path == "components/Navbar.tsx"
        ]
        
        print(f"   Total navbar versions: {len(all_navbar_versions)}")
        for v in all_navbar_versions:
            status = "✅ ACTIVE" if v.is_active else "❌ INACTIVE"
            print(f"   Version {v.version}: {status}")
        print()
        
        return artifacts
    
    def iteration_3_add_hero(self):
        """
        Iteration 3: Add a hero section
        
        Shows how new components are added alongside existing ones.
        """
        
        print("\n" + "🌟" * 40)
        print("ITERATION 3: Add Hero Section")
        print("🌟" * 40 + "\n")
        
        user_request = """
        Add a hero section for the homepage.
        
        Content:
        - Headline: "Build the Future with TechCo"
        - Subheadline: "Innovative solutions for modern businesses"
        - Primary CTA button: "Start Free Trial"
        - Secondary CTA button: "Watch Demo"
        - Background: gradient or subtle pattern
        
        Make it visually striking and compelling.
        """
        
        allowed_paths = [
            "components/Hero.tsx"
        ]
        
        print("🔍 KEY OBSERVATION:")
        print("─" * 80)
        print("   The navbar and carousel from previous iterations will NOT")
        print("   appear in the prompt because they're not in allowed_paths!")
        print("   This shows SCOPE CONTROL in action.")
        print()
        
        # Visualize prompt construction
        prompt = self._visualize_prompt_construction(
            iteration=3,
            user_request=user_request,
            allowed_paths=allowed_paths
        )
        
        # Execute and show results
        artifacts = self._execute_and_show_results(
            iteration=3,
            user_request=user_request,
            allowed_paths=allowed_paths,
            prompt=prompt
        )
        
        return artifacts
    
    def show_final_state(self):
        """Show the final state of the project"""
        
        print("\n" + "🎯" * 40)
        print("FINAL PROJECT STATE")
        print("🎯" * 40 + "\n")
        
        all_active = [a for a in self.orchestrator.state_rag.artifacts if a.is_active]
        
        print(f"📊 Project Summary:")
        print("─" * 80)
        print(f"   Total artifacts (all versions): {len(self.orchestrator.state_rag.artifacts)}")
        print(f"   Active components: {len(all_active)}")
        print()
        
        print("✅ Active Components:")
        print("─" * 80)
        for artifact in sorted(all_active, key=lambda a: a.file_path):
            print(f"   {artifact.file_path}")
            print(f"      Version: {artifact.version}")
            print(f"      Lines: {len(artifact.content.splitlines())}")
            print(f"      Source: {artifact.source}")
            print()
        
        # Save final state
        state_file = f"{self.demo_dir}/final_state.json"
        with open(state_file, 'w') as f:
            # Use model_dump() for Pydantic v2, dict() for v1
            artifacts_data = []
            for a in self.orchestrator.state_rag.artifacts:
                try:
                    # Try Pydantic v2 method
                    artifacts_data.append(a.model_dump())
                except AttributeError:
                    # Fallback to Pydantic v1 method
                    artifacts_data.append(a.dict())
            
            json.dump(
                artifacts_data,
                f,
                indent=2,
                default=str
            )
        
        print(f"💾 Full state saved to: {state_file}")
        print()
        
        # Show all generated files
        print("📁 Generated Files:")
        print("─" * 80)
        for root, dirs, files in os.walk(self.demo_dir):
            level = root.replace(self.demo_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        print()
    
    def run_complete_demo(self):
        """Run the complete website build demonstration"""
        
        print("\n🚀 Starting Complete Website Build Demo\n")
        
        # Setup
        self.setup()
        
        # Run iterations
        print("\n" + "=" * 80)
        print("BEGINNING ITERATIVE WEBSITE CONSTRUCTION")
        print("=" * 80)
        
        self.iteration_1_initial_website()
        self.iteration_2_modify_navbar()
        self.iteration_3_add_hero()
        
        # Show final state
        self.show_final_state()
        
        # Summary
        print("=" * 80)
        print("DEMO COMPLETE!")
        print("=" * 80)
        print()
        print("📚 What you learned:")
        print("   1. How State RAG provides authoritative context")
        print("   2. How Global RAG provides advisory patterns")
        print("   3. How prompts are constructed with clear boundaries")
        print("   4. How versioning works (navbar v1 → v2)")
        print("   5. How scope control prevents irrelevant context")
        print()
        print(f"📂 All files saved to: {self.demo_dir}/")
        print()


def main():
    """Run the demo"""
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        print("   Set it in your .env file or:")
        print("   $env:GEMINI_API_KEY = 'your-key'")
        return
    
    print(f"✅ API key found: ...{api_key[-8:]}\n")
    
    # Confirm
    print("⚠️  This demo will make 3-6 API calls to Gemini.")
    print("   Estimated time: ~1-2 minutes")
    print("   Cost: FREE (uses free tier)")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run demo
    demo = WebsiteBuildDemo(llm_provider="gemini")
    demo.run_complete_demo()


if __name__ == "__main__":
    main()