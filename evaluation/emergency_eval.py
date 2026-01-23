"""
STATE-RAG EVALUATION - CORRECT ARCHITECTURE COMPARISON
=======================================================

This evaluation tests the core architectural contribution of State-RAG:
separating authoritative project state from advisory design knowledge.

COMPARISON (3 Methods):
1. Full Context + Global RAG - Naive baseline (all files in prompt)
2. Single RAG - Traditional RAG (mixed code + knowledge in one DB)
3. Dual RAG (State-RAG) - Separated authoritative + advisory layers

KEY CLAIM TESTED:
"Dual-layer retrieval with authority enforcement outperforms both
naive full-context and traditional unified RAG approaches"

Test Suite: 20 focused cases
Metrics: 6 core measurements
Runtime: ~2.5 hours
API Calls: ~180 (Gemini free tier)
"""

import os
import sys
import json
import time
import shutil
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import traceback
import re

# Progress tracking
from tqdm import tqdm

# Data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add uploads directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your actual State-RAG implementation
from orchestrator import Orchestrator
from state_rag_manager import StateRAGManager
from global_rag import GlobalRAG
from artifact import Artifact
from state_rag_enums import ArtifactType, ArtifactSource
from llm_adapter import LLMAdapter
from schemas import GlobalRAGEntry

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TestCase:
    """Focused test case"""
    id: str
    category: str
    description: str
    initial_files: Dict[str, str]
    user_request: str
    allowed_paths: List[str]
    
    # Expected behavior
    should_create: List[str] = None
    should_not_create: List[str] = None
    user_modified_files: List[str] = None
    expected_file_count: int = None

@dataclass
class MetricResult:
    """Single metric measurement"""
    method: str
    test_id: str
    metric_name: str
    value: float
    timestamp: str

# ============================================================================
# TEST SUITE - 20 FOCUSED CASES
# ============================================================================

def build_test_suite() -> List[TestCase]:
    """20 carefully chosen test cases covering critical behaviors"""
    
    tests = []
    
    # -------------------------------------------------------------------------
    # CATEGORY 1: AUTHORITY PRESERVATION (5 tests)
    # -------------------------------------------------------------------------
    
    tests.append(TestCase(
        id="auth-01",
        category="Authority",
        description="User modifies navbar, AI should preserve user changes",
        initial_files={
            "components/Navbar.tsx": """import React from 'react';
export default function Navbar() {
  return <nav className="bg-blue-500 px-4 py-2">User Modified Navbar</nav>;
}""",
        },
        user_request="Make the navbar sticky at the top",
        allowed_paths=["components/Navbar.tsx"],
        user_modified_files=["components/Navbar.tsx"],
    ))
    
    tests.append(TestCase(
        id="auth-02",
        category="Authority",
        description="User file exists, create new file (should work)",
        initial_files={
            "components/Navbar.tsx": "// User navbar\nexport default function Navbar() { return <nav>User Version</nav>; }",
        },
        user_request="Create a Hero component with a large heading and subtitle",
        allowed_paths=["components/Hero.tsx"],
        user_modified_files=["components/Navbar.tsx"],
        should_create=["components/Hero.tsx"],
    ))
    
    tests.append(TestCase(
        id="auth-03",
        category="Authority",
        description="Try to modify user file without permission",
        initial_files={
            "components/Button.tsx": "// User button\nexport default function Button() { return <button className='custom'>User</button>; }",
        },
        user_request="Update all button components to use primary color",
        allowed_paths=["components/Hero.tsx"],  # NOT allowed to touch Button
        user_modified_files=["components/Button.tsx"],
    ))
    
    tests.append(TestCase(
        id="auth-04",
        category="Authority",
        description="User file with explicit permission",
        initial_files={
            "components/Card.tsx": "// User card\nexport default function Card() { return <div>Card</div>; }",
        },
        user_request="Add shadow and rounded corners to the card",
        allowed_paths=["components/Card.tsx"],
        user_modified_files=["components/Card.tsx"],
    ))
    
    tests.append(TestCase(
        id="auth-05",
        category="Authority",
        description="Multiple user files, modify one allowed",
        initial_files={
            "components/Navbar.tsx": "// User navbar",
            "components/Footer.tsx": "// User footer",
            "components/Hero.tsx": "// AI hero",
        },
        user_request="Add a subtitle to the hero section",
        allowed_paths=["components/Hero.tsx"],
        user_modified_files=["components/Navbar.tsx", "components/Footer.tsx"],
    ))
    
    # -------------------------------------------------------------------------
    # CATEGORY 2: HALLUCINATION PREVENTION (5 tests)
    # -------------------------------------------------------------------------
    
    tests.append(TestCase(
        id="hall-01",
        category="Hallucination",
        description="Request references non-existent file",
        initial_files={
            "components/Navbar.tsx": "export default function Navbar() { return <nav>Nav</nav>; }",
        },
        user_request="Update the Footer to match the Navbar styling",
        allowed_paths=["components/Footer.tsx"],
        should_not_create=["components/Sidebar.tsx", "components/Header.tsx"],
    ))
    
    tests.append(TestCase(
        id="hall-02",
        category="Hallucination",
        description="Create from scratch without hallucinating dependencies",
        initial_files={},
        user_request="Create a Hero component with title, subtitle, and CTA button",
        allowed_paths=["components/Hero.tsx"],
        should_create=["components/Hero.tsx"],
        expected_file_count=1,
    ))
    
    tests.append(TestCase(
        id="hall-03",
        category="Hallucination",
        description="Modify existing without creating random files",
        initial_files={
            "components/Button.tsx": "export default function Button() { return <button>Click</button>; }",
        },
        user_request="Make the button larger with hover effect",
        allowed_paths=["components/Button.tsx"],
        expected_file_count=1,
    ))
    
    tests.append(TestCase(
        id="hall-04",
        category="Hallucination",
        description="Complex request without inventing structure",
        initial_files={
            "components/Navbar.tsx": "export default function Navbar() { return <nav>Nav</nav>; }",
            "components/Hero.tsx": "export default function Hero() { return <section>Hero</section>; }",
        },
        user_request="Create a landing page that uses navbar and hero",
        allowed_paths=["pages/index.tsx"],
        should_create=["pages/index.tsx"],
        expected_file_count=1,
    ))
    
    tests.append(TestCase(
        id="hall-05",
        category="Hallucination",
        description="Don't assume project structure",
        initial_files={
            "components/Card.tsx": "export default function Card() { return <div>Card</div>; }",
        },
        user_request="Create a products page using the card component",
        allowed_paths=["pages/products.tsx"],
        should_create=["pages/products.tsx"],
        should_not_create=["components/ProductCard.tsx", "lib/utils.ts"],
    ))
    
    # -------------------------------------------------------------------------
    # CATEGORY 3: CONSISTENCY (5 tests)
    # -------------------------------------------------------------------------
    
    tests.append(TestCase(
        id="cons-01",
        category="Consistency",
        description="Same request should produce similar output",
        initial_files={
            "components/Button.tsx": "export default function Button() { return <button>Click</button>; }",
        },
        user_request="Add blue background and white text to button",
        allowed_paths=["components/Button.tsx"],
    ))
    
    tests.append(TestCase(
        id="cons-02",
        category="Consistency",
        description="Create component with specific requirements",
        initial_files={},
        user_request="Create a Navbar with logo on left and Home, About, Contact links on right",
        allowed_paths=["components/Navbar.tsx"],
    ))
    
    tests.append(TestCase(
        id="cons-03",
        category="Consistency",
        description="Modify multiple files consistently",
        initial_files={
            "components/Header.tsx": "export default function Header() { return <header>Header</header>; }",
            "components/Footer.tsx": "export default function Footer() { return <footer>Footer</footer>; }",
        },
        user_request="Add dark theme (bg-gray-900, text-white) to both",
        allowed_paths=["components/Header.tsx", "components/Footer.tsx"],
    ))
    
    tests.append(TestCase(
        id="cons-04",
        category="Consistency",
        description="Sequential modifications maintain structure",
        initial_files={
            "components/Card.tsx": "export default function Card() { return <div className='border'>Card</div>; }",
        },
        user_request="Add padding (p-4) and shadow (shadow-lg)",
        allowed_paths=["components/Card.tsx"],
    ))
    
    tests.append(TestCase(
        id="cons-05",
        category="Consistency",
        description="Create related components",
        initial_files={},
        user_request="Create ProductCard (displays single product) and ProductList (displays grid of cards)",
        allowed_paths=["components/ProductCard.tsx", "components/ProductList.tsx"],
    ))
    
    # -------------------------------------------------------------------------
    # CATEGORY 4: SCOPE ADHERENCE (5 tests)
    # -------------------------------------------------------------------------
    
    tests.append(TestCase(
        id="scope-01",
        category="Scope",
        description="Only modify allowed file",
        initial_files={
            "components/Navbar.tsx": "export default function Navbar() { return <nav>Nav</nav>; }",
            "components/Hero.tsx": "export default function Hero() { return <section>Hero</section>; }",
        },
        user_request="Make navbar sticky at top",
        allowed_paths=["components/Navbar.tsx"],
        expected_file_count=1,
    ))
    
    tests.append(TestCase(
        id="scope-02",
        category="Scope",
        description="Modify both allowed files",
        initial_files={
            "components/Button.tsx": "export default function Button() { return <button>Click</button>; }",
            "components/Card.tsx": "export default function Card() { return <div>Card</div>; }",
        },
        user_request="Add drop shadow (shadow-md) to both button and card",
        allowed_paths=["components/Button.tsx", "components/Card.tsx"],
        expected_file_count=2,
    ))
    
    tests.append(TestCase(
        id="scope-03",
        category="Scope",
        description="Don't modify out-of-scope files",
        initial_files={
            "components/Navbar.tsx": "export default function Navbar() { return <nav>Nav</nav>; }",
            "components/Sidebar.tsx": "export default function Sidebar() { return <aside>Side</aside>; }",
        },
        user_request="Create a main layout component",
        allowed_paths=["pages/layout.tsx"],
        should_not_create=["components/Layout.tsx"],
    ))
    
    tests.append(TestCase(
        id="scope-04",
        category="Scope",
        description="Create new in allowed scope",
        initial_files={},
        user_request="Create a pricing page with three tier cards",
        allowed_paths=["pages/pricing.tsx"],
        should_create=["pages/pricing.tsx"],
        expected_file_count=1,
    ))
    
    tests.append(TestCase(
        id="scope-05",
        category="Scope",
        description="Complex request with clear scope",
        initial_files={
            "components/Navbar.tsx": "export default function Navbar() { return <nav>Nav</nav>; }",
        },
        user_request="Create a landing page with hero, features, and CTA",
        allowed_paths=["pages/index.tsx"],
        expected_file_count=1,
    ))
    
    return tests


# ============================================================================
# GLOBAL RAG SETUP
# ============================================================================

def setup_global_rag() -> GlobalRAG:
    """Initialize Global RAG with design patterns"""
    
    rag = GlobalRAG()
    
    # Add some design knowledge
    patterns = [
        GlobalRAGEntry(
            id="pattern-sticky-nav",
            category="layout",
            title="Sticky Navigation Pattern",
            content="Use 'sticky top-0 z-50' classes for navbar. Add shadow on scroll for depth.",
            tags=["navbar", "layout", "sticky"],
            framework="react",
            styling="tailwind"
        ),
        GlobalRAGEntry(
            id="pattern-hero",
            category="layout",
            title="Hero Section Best Practices",
            content="Hero sections should have large heading (text-4xl or text-5xl), subtitle, and call-to-action button. Use min-h-screen for full viewport height.",
            tags=["hero", "landing", "layout"],
            framework="react",
            styling="tailwind"
        ),
        GlobalRAGEntry(
            id="pattern-cards",
            category="components",
            title="Card Component Guidelines",
            content="Cards should have consistent padding (p-4 or p-6), border or shadow for elevation, and rounded corners (rounded-lg). Group related content.",
            tags=["card", "components"],
            framework="react",
            styling="tailwind"
        ),
    ]
    
    for pattern in patterns:
        try:
            rag.ingest(pattern)
        except:
            pass  # May already exist
    
    return rag


# ============================================================================
# METHOD IMPLEMENTATIONS
# ============================================================================

class FullContextMethod:
    """
    Method 1: Full Context + Global RAG
    - Puts ALL files in the prompt (up to token limit)
    - Uses Global RAG for design patterns
    - No retrieval, no selection, no authority
    """
    
    def __init__(self, llm: LLMAdapter, global_rag: GlobalRAG):
        self.llm = llm
        self.global_rag = global_rag
        self.files = {}
        self.user_modified = set()
        self.max_tokens = 6000  # Conservative limit for context
    
    def reset(self):
        self.files = {}
        self.user_modified = set()
    
    def add_file(self, path: str, content: str, user_modified: bool = False):
        self.files[path] = content
        if user_modified:
            self.user_modified.add(path)
    
    def generate(self, request: str, allowed_paths: List[str]) -> Dict[str, str]:
        """Generate with full context + Global RAG"""
        
        # Get design patterns from Global RAG
        try:
            global_refs = self.global_rag.retrieve(query=request, k=3)
            if not global_refs:
                global_refs = []
        except Exception as e:
            print(f"⚠️  Global RAG warning: {e}")
            global_refs = []
                
        # Build prompt with ALL files (truncated if needed)
        prompt_parts = [
            "You are a React/TypeScript code generator.\n\n",
            "=== DESIGN PATTERNS (advisory) ===\n"
        ]
        
        for i, ref in enumerate(global_refs, 1):
            prompt_parts.append(f"{i}. {ref.title}\n{ref.content}\n\n")
        
        prompt_parts.append("=== PROJECT FILES (all available) ===\n")
        
        # Add all files (truncate if too large)
        total_chars = sum(len(p) for p in prompt_parts)
        for path, content in sorted(self.files.items()):
            marker = "[USER]" if path in self.user_modified else "[AI]"
            file_block = f"\n{marker} FILE: {path}\n{content}\n"
            
            if total_chars + len(file_block) > self.max_tokens * 3:  # Rough estimate
                file_block = f"\n{marker} FILE: {path}\n{content[:500]}...[truncated]\n"
            
            prompt_parts.append(file_block)
            total_chars += len(file_block)
        
        prompt_parts.append(f"\n=== REQUEST ===\n{request}\n\n")
        prompt_parts.append(f"=== ALLOWED TO MODIFY ===\n{', '.join(allowed_paths)}\n\n")
        prompt_parts.append(
            "=== OUTPUT FORMAT ===\n"
            "FILE: <path>\n<complete content>\n\n"
            "Only output files you create or modify.\n"
        )
        
        prompt = "".join(prompt_parts)
        
        try:
            raw_output = self.llm.generate(prompt)
            return self._parse_output(raw_output)
        except Exception as e:
            print(f"Full Context error: {e}")
            return {}
    
    def _parse_output(self, raw: str) -> Dict[str, str]:
        files = {}
        pattern = re.compile(r'^FILE:\s*(.+)$', re.MULTILINE)
        matches = list(pattern.finditer(raw))
        
        for i, match in enumerate(matches):
            path = match.group(1).strip()
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(raw)
            content = raw[start:end].strip()
            if content:
                files[path] = content
        
        return files


class SingleRAGMethod:
    """
    Method 2: Single RAG (Unified)
    - ONE vector DB containing both code files AND design patterns
    - Retrieves top-k most similar regardless of type
    - No authority distinction
    - No versioning
    """
    
    def __init__(self, llm: LLMAdapter, global_rag: GlobalRAG):
        self.llm = llm
        self.global_rag = global_rag
        self.state_rag = StateRAGManager()  # Use same tech but treat as unified
        self.user_modified = set()
    
    def reset(self):
        # Clean up old state
        import gc
        import time
        
        gc.collect()
        time.sleep(0.3)
        
        if os.path.exists("state_rag"):
            try:
                shutil.rmtree("state_rag", ignore_errors=True)
            except:
                # Windows fallback
                try:
                    backup = f"state_rag_old_{int(time.time())}"
                    os.rename("state_rag", backup)
                except:
                    pass
        
        self.state_rag = StateRAGManager()
        self.user_modified = set()
    
    def add_file(self, path: str, content: str, user_modified: bool = False):
        """Add to unified RAG (no authority distinction)"""
        artifact = Artifact(
            type=ArtifactType.component,
            name=path.split("/")[-1],
            file_path=path,
            content=content,
            language=self._infer_language(path),
            source=ArtifactSource.ai_generated,  # Treats everything as AI
        )
        self.state_rag.commit(artifact)
        if user_modified:
            self.user_modified.add(path)
    
    def generate(self, request: str, allowed_paths: List[str]) -> Dict[str, str]:
        """Generate with unified RAG retrieval"""
        
        # Retrieve from unified code DB
        code_artifacts = self.state_rag.retrieve(user_query=request, limit=5)
        
        # Also get design patterns
        try:
            design_refs = self.global_rag.retrieve(query=request, k=3)
            if not design_refs:
                design_refs = []
        except Exception as e:
            print(f"⚠️  Global RAG warning: {e}")
            design_refs = []
        
        # Build prompt - treat everything as suggestions
        prompt_parts = [
            "You are a React/TypeScript code generator.\n\n",
            "=== RETRIEVED CONTEXT (treat as suggestions) ===\n\n"
        ]
        
        # Add design patterns
        for ref in design_refs:
            prompt_parts.append(f"PATTERN: {ref.title}\n{ref.content}\n\n")
        
        # Add code artifacts
        for artifact in code_artifacts:
            prompt_parts.append(f"CODE: {artifact.file_path}\n{artifact.content}\n\n")
        
        prompt_parts.append(f"=== REQUEST ===\n{request}\n\n")
        prompt_parts.append(f"Allowed files: {', '.join(allowed_paths)}\n\n")
        prompt_parts.append("OUTPUT FORMAT:\nFILE: <path>\n<content>\n")
        
        prompt = "".join(prompt_parts)
        
        try:
            raw_output = self.llm.generate(prompt)
            return self._parse_output(raw_output)
        except Exception as e:
            print(f"Single RAG error: {e}")
            return {}
    
    def _parse_output(self, raw: str) -> Dict[str, str]:
        files = {}
        pattern = re.compile(r'^FILE:\s*(.+)$', re.MULTILINE)
        matches = list(pattern.finditer(raw))
        
        for i, match in enumerate(matches):
            path = match.group(1).strip()
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(raw)
            content = raw[start:end].strip()
            if content:
                files[path] = content
        
        return files
    
    def _infer_language(self, path: str) -> str:
        if path.endswith(".tsx"): return "tsx"
        if path.endswith(".ts"): return "ts"
        if path.endswith(".js"): return "js"
        return "tsx"


class DualRAGMethod:
    """
    Method 3: Dual RAG (State-RAG) - Your actual implementation
    - State RAG: Authoritative project state
    - Global RAG: Advisory design knowledge
    - Authority enforcement
    - Versioning
    """
    
    def __init__(self):
        self.orchestrator = Orchestrator(llm_provider="gemini")
    
    def reset(self):
        import gc
        import time
        
        gc.collect()
        time.sleep(0.3)
        
        if os.path.exists("state_rag"):
            try:
                shutil.rmtree("state_rag", ignore_errors=True)
            except:
                try:
                    backup = f"state_rag_old_{int(time.time())}"
                    os.rename("state_rag", backup)
                except:
                    pass
        
        self.orchestrator = Orchestrator(llm_provider="gemini")
    
    def add_file(self, path: str, content: str, user_modified: bool = False):
        """Add with proper authority tracking"""
        artifact = Artifact(
            type=self._infer_type(path),
            name=path.split("/")[-1],
            file_path=path,
            content=content,
            language=self._infer_language(path),
            source=ArtifactSource.user_modified if user_modified else ArtifactSource.ai_generated,
        )
        self.orchestrator.state_rag.commit(artifact)
    
    def generate(self, request: str, allowed_paths: List[str]) -> Dict[str, str]:
        """Generate using State-RAG orchestrator"""
        try:
            artifacts = self.orchestrator.handle_request(
                user_request=request,
                allowed_paths=allowed_paths,
            )
            return {a.file_path: a.content for a in artifacts}
        except Exception as e:
            print(f"Dual RAG error: {e}")
            return {}
    
    def _infer_type(self, path: str) -> ArtifactType:
        if "components/" in path: return ArtifactType.component
        if "pages/" in path or "app/" in path: return ArtifactType.page
        return ArtifactType.config
    
    def _infer_language(self, path: str) -> str:
        if path.endswith(".tsx"): return "tsx"
        if path.endswith(".ts"): return "ts"
        if path.endswith(".js"): return "js"
        return "tsx"


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_hallucination_rate(
    output_files: Dict[str, str],
    initial_files: Dict[str, str],
    allowed_paths: List[str]
) -> float:
    """Files that shouldn't exist"""
    if not output_files:
        return 0.0
    
    unexpected = 0
    for path in output_files.keys():
        if path not in initial_files and path not in allowed_paths:
            unexpected += 1
    
    return unexpected / len(output_files) if output_files else 0.0


def compute_consistency_score(
    run1_output: Dict[str, str],
    run2_output: Dict[str, str]
) -> float:
    """Similarity across repeated runs"""
    if not run1_output or not run2_output:
        return 0.0
    
    files1 = set(run1_output.keys())
    files2 = set(run2_output.keys())
    
    if files1 != files2:
        overlap = len(files1 & files2)
        total = len(files1 | files2)
        file_similarity = overlap / total if total > 0 else 0.0
    else:
        file_similarity = 1.0
    
    content_similarities = []
    for path in files1 & files2:
        content1 = run1_output[path]
        content2 = run2_output[path]
        max_len = max(len(content1), len(content2))
        if max_len == 0:
            content_similarities.append(1.0)
        else:
            matches = sum(c1 == c2 for c1, c2 in zip(content1, content2))
            similarity = matches / max_len
            content_similarities.append(similarity)
    
    content_similarity = np.mean(content_similarities) if content_similarities else 0.0
    return 0.5 * file_similarity + 0.5 * content_similarity


def compute_authority_preservation(
    output_files: Dict[str, str],
    user_modified_files: List[str],
    allowed_paths: List[str]
) -> float:
    """User files preserved when not in allowed_paths"""
    if not user_modified_files:
        return 1.0
    
    preserved = 0
    for user_file in user_modified_files:
        if user_file not in allowed_paths:
            if user_file not in output_files:
                preserved += 1
        else:
            preserved += 1
    
    return preserved / len(user_modified_files)


def compute_scope_adherence(
    output_files: Dict[str, str],
    allowed_paths: List[str]
) -> float:
    """Only allowed files modified"""
    if not output_files:
        return 1.0
    
    in_scope = sum(1 for path in output_files.keys() if path in allowed_paths)
    return in_scope / len(output_files)


def compute_versioning_correctness(method_instance) -> float:
    """Correct version numbers (Dual RAG only)"""
    if not hasattr(method_instance, 'orchestrator'):
        return 0.0
    
    state_rag = method_instance.orchestrator.state_rag
    active_artifacts = [a for a in state_rag.artifacts if a.is_active]
    
    if not active_artifacts:
        return 1.0
    
    file_versions = {}
    for artifact in state_rag.artifacts:
        path = artifact.file_path
        if path not in file_versions:
            file_versions[path] = []
        file_versions[path].append(artifact.version)
    
    correct = 0
    for path, versions in file_versions.items():
        expected = list(range(1, len(versions) + 1))
        if sorted(versions) == expected:
            correct += 1
    
    return correct / len(file_versions) if file_versions else 1.0


def compute_dependency_tracking(
    output_files: Dict[str, str],
    test_case: TestCase
) -> float:
    """Referenced components are included"""
    if not output_files:
        return 0.0
    
    total_deps = 0
    satisfied_deps = 0
    
    for path, content in output_files.items():
        imports = re.findall(r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]', content)
        
        for imp in imports:
            total_deps += 1
            imported_path = imp
            if not imported_path.endswith('.tsx') and not imported_path.endswith('.ts'):
                imported_path += '.tsx'
            
            if imported_path in output_files or imported_path in test_case.initial_files:
                satisfied_deps += 1
    
    return satisfied_deps / total_deps if total_deps > 0 else 1.0


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

class StateRAGEvaluator:
    """Focused 3-method evaluation"""
    
    def __init__(self):
        self.llm = LLMAdapter(provider="gemini")
        self.global_rag = setup_global_rag()
        self.results = []
        
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir.absolute()}")
    
    def run_evaluation(self):
        """Run complete evaluation"""
        
        print("\n" + "="*80)
        print("STATE-RAG ARCHITECTURE EVALUATION")
        print("="*80)
        print()
        
        tests = build_test_suite()
        print(f"📋 Test Suite: {len(tests)} focused cases")
        print(f"🤖 LLM: Gemini 2.0 Flash")
        print(f"⏱️  Estimated time: 2-3 hours")
        print(f"💰 Cost: FREE (Gemini free tier)")
        print()
        
        # Initialize methods
        methods = {
            "Full Context": FullContextMethod(self.llm, self.global_rag),
            "Single RAG": SingleRAGMethod(self.llm, self.global_rag),
            "Dual RAG": DualRAGMethod(),
        }
        
        print("🔧 Methods initialized:")
        for name in methods.keys():
            print(f"  ✓ {name}")
        print()
        
        # Run evaluation
        total_runs = len(tests) * len(methods)
        pbar = tqdm(total=total_runs, desc="Running tests")
        
        for test in tests:
            for method_name, method in methods.items():
                try:
                    method.reset()
                    
                    # Setup initial files
                    for path, content in test.initial_files.items():
                        is_user = test.user_modified_files and path in test.user_modified_files
                        method.add_file(path, content, user_modified=is_user)
                    
                    # Run twice for consistency
                    output1 = method.generate(test.user_request, test.allowed_paths)
                    
                    method.reset()
                    for path, content in test.initial_files.items():
                        is_user = test.user_modified_files and path in test.user_modified_files
                        method.add_file(path, content, user_modified=is_user)
                    output2 = method.generate(test.user_request, test.allowed_paths)
                    
                    # Compute metrics
                    metrics = self._compute_all_metrics(
                        test, method_name, method, output1, output2
                    )
                    self.results.extend(metrics)
                    
                except Exception as e:
                    print(f"\n❌ Error in {method_name} on {test.id}: {e}")
                    traceback.print_exc()
                
                pbar.update(1)
                time.sleep(0.5)
        
        pbar.close()
        
        self._save_results()
        self._generate_visualizations()
        self._print_summary()
    
    def _compute_all_metrics(self, test, method_name, method_instance, output1, output2):
        metrics = []
        timestamp = datetime.now().isoformat()
        
        metrics.append(MetricResult(
            method=method_name, test_id=test.id, metric_name="hallucination_rate",
            value=compute_hallucination_rate(output1, test.initial_files, test.allowed_paths),
            timestamp=timestamp
        ))
        
        metrics.append(MetricResult(
            method=method_name, test_id=test.id, metric_name="consistency_score",
            value=compute_consistency_score(output1, output2),
            timestamp=timestamp
        ))
        
        metrics.append(MetricResult(
            method=method_name, test_id=test.id, metric_name="authority_preservation",
            value=compute_authority_preservation(output1, test.user_modified_files or [], test.allowed_paths),
            timestamp=timestamp
        ))
        
        metrics.append(MetricResult(
            method=method_name, test_id=test.id, metric_name="scope_adherence",
            value=compute_scope_adherence(output1, test.allowed_paths),
            timestamp=timestamp
        ))
        
        metrics.append(MetricResult(
            method=method_name, test_id=test.id, metric_name="versioning_correctness",
            value=compute_versioning_correctness(method_instance),
            timestamp=timestamp
        ))
        
        metrics.append(MetricResult(
            method=method_name, test_id=test.id, metric_name="dependency_tracking",
            value=compute_dependency_tracking(output1, test),
            timestamp=timestamp
        ))
        
        return metrics
    
    def _save_results(self):
        df = pd.DataFrame([asdict(r) for r in self.results])
        csv_path = self.output_dir / "raw_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
    
    def _generate_visualizations(self):
        df = pd.DataFrame([asdict(r) for r in self.results])
        summary = df.groupby(['method', 'metric_name'])['value'].mean().reset_index()
        pivot = summary.pivot(index='method', columns='metric_name', values='value')
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        metrics_to_plot = [
            'hallucination_rate', 'consistency_score', 'authority_preservation',
            'scope_adherence', 'versioning_correctness', 'dependency_tracking'
        ]
        
        pivot[metrics_to_plot].plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('State-RAG vs Baselines: Architectural Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend([m.replace('_', ' ').title() for m in metrics_to_plot], loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_chart.png", dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved")
        
        # Radar chart
        from math import pi
        categories = [m.replace('_', ' ').title() for m in metrics_to_plot]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        for method in pivot.index:
            values = pivot.loc[method, metrics_to_plot].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('State-RAG: Architectural Comparison', size=16, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / "radar_chart.png", dpi=300, bbox_inches='tight')
        print(f"📊 Radar chart saved")
    
    def _print_summary(self):
        df = pd.DataFrame([asdict(r) for r in self.results])
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        summary = df.groupby(['method', 'metric_name'])['value'].agg(['mean', 'std']).reset_index()
        
        for metric in ['hallucination_rate', 'consistency_score', 'authority_preservation', 'scope_adherence']:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 60)
            metric_data = summary[summary['metric_name'] == metric]
            for _, row in metric_data.iterrows():
                print(f"  {row['method']:20s}: {row['mean']:.3f} (±{row['std']:.3f})")
        
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)
        
        dual_data = summary[summary['method'] == 'Dual RAG']
        auth = dual_data[dual_data['metric_name'] == 'authority_preservation']['mean'].values[0]
        scope = dual_data[dual_data['metric_name'] == 'scope_adherence']['mean'].values[0]
        hall = dual_data[dual_data['metric_name'] == 'hallucination_rate']['mean'].values[0]
        
        print(f"\n✓ Authority Preservation: {auth:.2f} (Dual RAG)")
        print(f"✓ Scope Adherence: {scope:.2f} (Dual RAG)")
        print(f"✓ Hallucination Rate: {hall:.2f} (Dual RAG - lower is better)")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        return
    
    print(f"✅ Gemini API key found")
    print("\n" + "="*80)
    print("STATE-RAG EVALUATION - CORRECTED ARCHITECTURE COMPARISON")
    print("="*80)
    print("\nThis compares:")
    print("  1. Full Context + Global RAG (naive baseline)")
    print("  2. Single RAG (traditional unified RAG)")
    print("  3. Dual RAG (State-RAG - your contribution)")
    print()
    print("Testing the claim:")
    print("  'Separating authoritative state from advisory knowledge")
    print("   outperforms both naive and unified approaches'")
    print()
    
    response = input("Start evaluation? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    evaluator = StateRAGEvaluator()
    evaluator.run_evaluation()
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()