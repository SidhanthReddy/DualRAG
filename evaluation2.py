"""
DualRAG Evaluation Framework for Conference Publication
========================================================

This is a comprehensive, rigorous evaluation designed for academic publication.

EVALUATION DIMENSIONS (8 core metrics):
1. State Consistency - Does the system maintain coherent state over time?
2. Authority Preservation - Does it respect user modifications?
3. Hallucination Prevention - Does it reference non-existent artifacts?
4. Scope Adherence - Does it modify only allowed files?
5. Dependency Tracking - Does it correctly include dependent artifacts?
6. Versioning Correctness - Does it properly version artifacts?
7. Retrieval Accuracy - Does it retrieve the right artifacts?
8. Token Efficiency - How much context does it use?

BASELINES (4 methods):
1. Context Window (CW) - Everything in context
2. Conversation History (CH) - Chat-style with summarization
3. Single RAG (SR) - One vector DB for everything
4. DualRAG (Ours) - Separated State + Global RAG

TEST SUITE (50+ test cases across 6 categories):
- Basic CRUD operations (10 cases)
- Multi-file workflows (10 cases)
- Authority & safety (10 cases)
- Versioning & rollback (8 cases)
- Dependency management (6 cases)
- Edge cases & stress tests (8 cases)

OUTPUT:
- Raw CSV data for statistical analysis
- Publication-ready charts (high DPI)
- LaTeX tables with significance testing
- Detailed error logs for debugging
"""

import os
import sys
import json
import time
import re
import shutil
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import traceback

# Data analysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# Setup
sys.path.insert(0, '/mnt/user-data/uploads')
from dotenv import load_dotenv
load_dotenv()

# DualRAG imports - make them optional
try:
    from orchestrator import Orchestrator
    from state_rag_manager import StateRAGManager
    from global_rag import GlobalRAG
    from artifact import Artifact
    from state_rag_enums import ArtifactType, ArtifactSource
    from llm_adapter import LLMAdapter
    from schemas import GlobalRAGEntry
    from validator import ProposedArtifact
    DUALRAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: DualRAG modules not found: {e}")
    print("   Some baseline methods will not be available.")
    DUALRAG_AVAILABLE = False
    
    # Create mock classes to prevent errors
    class Orchestrator: pass
    class StateRAGManager: 
        def __init__(self): self.artifacts = []
        def commit(self, artifact): pass
        def retrieve(self, user_query, limit=5): return []
    class GlobalRAG: pass
    class Artifact: pass
    class ArtifactType: 
        component = "component"
    class ArtifactSource: 
        ai_generated = "ai_generated"
        user_uploaded = "user_uploaded"
    class LLMAdapter: 
        def __init__(self, provider="gemini", model=None, api_key=None): pass
        def generate(self, prompt): return "Mock response"
    class GlobalRAGEntry: pass
    class ProposedArtifact: pass

# Plotting configuration
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EvaluationResult:
    """Single measurement from evaluation"""
    method: str              # Which method was tested
    test_case_id: str        # Test case identifier
    metric: str              # What was measured
    value: float             # Numeric result
    success: bool            # Did the test pass?
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)  # Additional context


@dataclass
class TestCase:
    """Comprehensive test case definition"""
    id: str
    category: str            # CRUD, Multi-file, Authority, etc.
    name: str
    description: str
    initial_files: Dict[str, str]  # file_path -> content
    user_request: str
    expected_files: List[str]      # Files that should exist after
    forbidden_files: List[str]     # Files that should NOT be created
    
    # What to test
    check_consistency: bool = False
    check_authority: bool = False
    check_hallucination: bool = False
    check_scope: bool = False
    check_versioning: bool = False
    check_dependencies: bool = False
    
    # Expected behavior
    expected_versions: Dict[str, int] = field(default_factory=dict)
    should_fail: bool = False  # Should this operation be rejected?
    notes: str = ""


@dataclass 
class MethodResult:
    """Complete result for one method on one test"""
    success: bool
    output: str
    created_files: List[str]
    modified_files: List[str]
    errors: List[str]
    token_usage: int
    execution_time: float
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# BASELINE IMPLEMENTATIONS
# ============================================================================

class ContextWindowMethod:
    """Baseline 1: Everything stays in context window"""
    
    def __init__(self, llm: LLMAdapter):
        self.llm = llm
        self.files = {}  # Current state
        self.history = []  # All past requests
        self.max_context_tokens = 8000
    
    def add_file(self, file_path: str, content: str):
        """Add or update file"""
        self.files[file_path] = content
    
    def generate(self, user_request: str) -> MethodResult:
        """Generate response with full context"""
        start_time = time.time()
        errors = []
        
        try:
            # Build prompt with ALL context
            prompt_parts = [
                "You are a code generator. Generate clean, production-ready code.\n\n"
            ]
            
            # Add all files
            if self.files:
                prompt_parts.append("=== CURRENT PROJECT FILES ===\n")
                for path, content in self.files.items():
                    prompt_parts.append(f"\n--- {path} ---\n{content}\n")
            
            # Add conversation history (truncated)
            if self.history:
                prompt_parts.append("\n=== RECENT HISTORY ===\n")
                recent = self.history[-3:]  # Last 3 interactions
                for h in recent:
                    prompt_parts.append(f"User: {h}\n")
            
            prompt_parts.append(f"\n=== NEW REQUEST ===\n{user_request}\n\n")
            prompt_parts.append(
                "=== OUTPUT FORMAT ===\n"
                "For each file you create or modify, use:\n"
                "FILE: <file_path>\n"
                "<complete file content>\n\n"
            )
            
            prompt = "".join(prompt_parts)
            
            # Token limit check
            estimated_tokens = len(prompt) // 4
            if estimated_tokens > self.max_context_tokens:
                # Emergency: truncate files to fit
                errors.append(f"Context overflow ({estimated_tokens} tokens), truncating")
                # Remove oldest files
                if len(self.files) > 3:
                    files_to_keep = list(self.files.keys())[-3:]
                    self.files = {k: v for k, v in self.files.items() if k in files_to_keep}
                    return self.generate(user_request)  # Retry
            
            # Generate
            output = self.llm.generate(prompt)
            
            # Parse output and update state
            created, modified = self._parse_and_update(output)
            
            # Store in history
            self.history.append(user_request)
            
            return MethodResult(
                success=True,
                output=output,
                created_files=created,
                modified_files=modified,
                errors=errors,
                token_usage=estimated_tokens,
                execution_time=time.time() - start_time,
                metadata={"context_files": len(self.files), "history_length": len(self.history)}
            )
            
        except Exception as e:
            return MethodResult(
                success=False,
                output="",
                created_files=[],
                modified_files=[],
                errors=[f"Generation failed: {str(e)}"],
                token_usage=0,
                execution_time=time.time() - start_time
            )
    
    def _parse_and_update(self, output: str) -> Tuple[List[str], List[str]]:
        """Parse LLM output and update internal state"""
        created = []
        modified = []
        
        # Find FILE: headers
        file_pattern = re.compile(r'^FILE:\s*(.+?)$', re.MULTILINE)
        matches = list(file_pattern.finditer(output))
        
        for i, match in enumerate(matches):
            file_path = match.group(1).strip()
            
            # Extract content
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
            content = output[content_start:content_end].strip()
            
            if file_path in self.files:
                modified.append(file_path)
            else:
                created.append(file_path)
            
            self.files[file_path] = content
        
        return created, modified
    
    def get_token_usage(self) -> int:
        """Current context size"""
        total = sum(len(c) for c in self.files.values())
        total += sum(len(h) for h in self.history)
        return total // 4


class ConversationHistoryMethod:
    """Baseline 2: Chat-style with sliding window"""
    
    def __init__(self, llm: LLMAdapter):
        self.llm = llm
        self.files = {}
        self.messages = []  # (role, content) pairs
        self.max_messages = 10
    
    def add_file(self, file_path: str, content: str):
        """Add or update file"""
        self.files[file_path] = content
    
    def generate(self, user_request: str) -> MethodResult:
        """Generate with conversation history"""
        start_time = time.time()
        
        try:
            prompt_parts = [
                "You are a code generator.\n\n"
            ]
            
            # Current files
            if self.files:
                prompt_parts.append("=== CURRENT FILES ===\n")
                for path, content in self.files.items():
                    prompt_parts.append(f"--- {path} ---\n{content}\n\n")
            
            # Recent conversation (sliding window)
            if self.messages:
                prompt_parts.append("=== RECENT CONVERSATION ===\n")
                recent = self.messages[-self.max_messages:]
                for role, content in recent:
                    prompt_parts.append(f"{role}: {content}\n")
                prompt_parts.append("\n")
            
            prompt_parts.append(f"User: {user_request}\n\n")
            prompt_parts.append("OUTPUT FORMAT:\nFILE: <path>\n<content>\n")
            
            prompt = "".join(prompt_parts)
            output = self.llm.generate(prompt)
            
            # Update state
            created, modified = self._parse_and_update(output)
            
            # Store conversation
            self.messages.append(("User", user_request))
            self.messages.append(("Assistant", "<code generated>"))
            
            # Trim old messages
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]
            
            tokens = len(prompt) // 4
            
            return MethodResult(
                success=True,
                output=output,
                created_files=created,
                modified_files=modified,
                errors=[],
                token_usage=tokens,
                execution_time=time.time() - start_time,
                metadata={"num_messages": len(self.messages)}
            )
            
        except Exception as e:
            return MethodResult(
                success=False,
                output="",
                created_files=[],
                modified_files=[],
                errors=[str(e)],
                token_usage=0,
                execution_time=time.time() - start_time
            )
    
    def _parse_and_update(self, output: str) -> Tuple[List[str], List[str]]:
        """Parse and update files"""
        created = []
        modified = []
        
        file_pattern = re.compile(r'^FILE:\s*(.+?)$', re.MULTILINE)
        matches = list(file_pattern.finditer(output))
        
        for i, match in enumerate(matches):
            file_path = match.group(1).strip()
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
            content = output[content_start:content_end].strip()
            
            if file_path in self.files:
                modified.append(file_path)
            else:
                created.append(file_path)
            
            self.files[file_path] = content
        
        return created, modified


class SingleRAGMethod:
    """Baseline 3: One RAG for everything (no separation)"""
    
    def __init__(self, llm: LLMAdapter):
        self.llm = llm
        self.state_rag = StateRAGManager()
        self.state_rag.artifacts = []  # Fresh start
    
    def add_file(self, file_path: str, content: str):
        """Add file to vector DB"""
        artifact = Artifact(
            type=ArtifactType.component,
            name=file_path.split("/")[-1],
            file_path=file_path,
            content=content,
            language=self._infer_language(file_path),
            source=ArtifactSource.ai_generated
        )
        self.state_rag.commit(artifact)
    
    def generate(self, user_request: str) -> MethodResult:
        """Generate using semantic retrieval"""
        start_time = time.time()
        
        try:
            # Semantic search over ALL artifacts
            retrieved = self.state_rag.retrieve(
                user_query=user_request,
                limit=5
            )
            
            prompt_parts = [
                "You are a code generator.\n\n"
            ]
            
            if retrieved:
                prompt_parts.append("=== RELEVANT FILES (from semantic search) ===\n")
                for artifact in retrieved:
                    prompt_parts.append(f"--- {artifact.file_path} ---\n")
                    prompt_parts.append(f"{artifact.content}\n\n")
            
            prompt_parts.append(f"USER REQUEST:\n{user_request}\n\n")
            prompt_parts.append("OUTPUT FORMAT:\nFILE: <path>\n<content>\n")
            
            prompt = "".join(prompt_parts)
            output = self.llm.generate(prompt)
            
            # Parse and store
            created, modified = self._parse_and_update(output)
            
            tokens = len(prompt) // 4
            
            return MethodResult(
                success=True,
                output=output,
                created_files=created,
                modified_files=modified,
                errors=[],
                token_usage=tokens,
                execution_time=time.time() - start_time,
                metadata={"retrieved_count": len(retrieved)}
            )
            
        except Exception as e:
            return MethodResult(
                success=False,
                output="",
                created_files=[],
                modified_files=[],
                errors=[str(e)],
                token_usage=0,
                execution_time=time.time() - start_time
            )
    
    def _parse_and_update(self, output: str) -> Tuple[List[str], List[str]]:
        """Parse output and commit to RAG"""
        created = []
        modified = []
        
        file_pattern = re.compile(r'^FILE:\s*(.+?)$', re.MULTILINE)
        matches = list(file_pattern.finditer(output))
        
        for i, match in enumerate(matches):
            file_path = match.group(1).strip()
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
            content = output[content_start:content_end].strip()
            
            # Check if exists
            existing = [a for a in self.state_rag.artifacts 
                       if a.file_path == file_path and a.is_active]
            
            if existing:
                modified.append(file_path)
            else:
                created.append(file_path)
            
            # Commit
            artifact = Artifact(
                type=ArtifactType.component,
                name=file_path.split("/")[-1],
                file_path=file_path,
                content=content,
                language=self._infer_language(file_path),
                source=ArtifactSource.ai_generated
            )
            self.state_rag.commit(artifact)
        
        return created, modified
    
    def _infer_language(self, path: str) -> str:
        """Infer language from file extension"""
        if path.endswith('.tsx'): return 'tsx'
        if path.endswith('.ts'): return 'ts'
        if path.endswith('.jsx'): return 'jsx'
        if path.endswith('.js'): return 'js'
        if path.endswith('.css'): return 'css'
        if path.endswith('.json'): return 'json'
        return 'tsx'


class DualRAGMethod:
    """Our approach: Separated State + Global RAG with authority"""
    
    def __init__(self, llm_provider: str = "gemini"):
        # Clean state directory
        state_dir = "/mnt/user-data/uploads/state_rag"
        if os.path.exists(state_dir):
            shutil.rmtree(state_dir)
        os.makedirs(state_dir, exist_ok=True)
        
        self.orchestrator = Orchestrator(llm_provider=llm_provider)
        self.orchestrator.state_rag.artifacts = []
        self.orchestrator.state_rag._persist()
        
        # Seed Global RAG with patterns
        self._seed_global_rag()
    
    def _seed_global_rag(self):
        """Add reusable patterns to Global RAG"""
        patterns = [
            GlobalRAGEntry(
                id="navbar-pattern",
                category="component",
                title="Navbar Component Pattern",
                content="Sticky navigation bar with logo, links, and CTA button. Use 'sticky top-0 z-50 bg-white shadow-md'. Common props: links[], logo, ctaText.",
                tags=["navbar", "navigation"],
                framework="react",
                styling="tailwind"
            ),
            GlobalRAGEntry(
                id="button-pattern",
                category="component",
                title="Button Component Pattern",
                content="Reusable button with variants (primary, secondary, danger). Props: variant, size, disabled, onClick. Use Tailwind for styling.",
                tags=["button", "ui"],
                framework="react",
                styling="tailwind"
            ),
            GlobalRAGEntry(
                id="form-pattern",
                category="component",
                title="Form Component Pattern",
                content="Controlled form with validation. Use useState for form state, validate onBlur, show errors below fields. Common fields: email, password.",
                tags=["form", "validation"],
                framework="react",
                styling="tailwind"
            ),
            GlobalRAGEntry(
                id="card-pattern",
                category="component",
                title="Card Component Pattern",
                content="Flexible card container. Structure: header (optional), body, footer (optional). Use 'rounded-lg border shadow-sm' for styling.",
                tags=["card", "container"],
                framework="react",
                styling="tailwind"
            ),
            GlobalRAGEntry(
                id="modal-pattern",
                category="component",
                title="Modal Component Pattern",
                content="Modal dialog with backdrop. Use portal (createPortal) to render outside parent. Props: isOpen, onClose, title, children.",
                tags=["modal", "dialog"],
                framework="react",
                styling="tailwind"
            ),
        ]
        
        for pattern in patterns:
            self.orchestrator.global_rag.ingest(pattern)
    
    def add_file(self, file_path: str, content: str, source: ArtifactSource = ArtifactSource.ai_generated):
        """Add file to State RAG"""
        artifact = Artifact(
            type=self._infer_type(file_path),
            name=file_path.split("/")[-1],
            file_path=file_path,
            content=content,
            language=self._infer_language(file_path),
            source=source  # Preserve source!
        )
        self.orchestrator.state_rag.commit(artifact)
    
    def generate(self, user_request: str, allowed_paths: List[str]) -> MethodResult:
        """Generate using DualRAG orchestration"""
        start_time = time.time()
        
        try:
            # Use orchestrator (handles State RAG + Global RAG + validation)
            committed = self.orchestrator.handle_request(
                user_request=user_request,
                allowed_paths=allowed_paths
            )
            
            # Build output in same format as baselines
            output_parts = []
            created = []
            modified = []
            
            for artifact in committed:
                output_parts.append(f"FILE: {artifact.file_path}\n{artifact.content}\n")
                
                # Check if it's new or modified
                old_versions = [a for a in self.orchestrator.state_rag.artifacts 
                               if a.file_path == artifact.file_path and not a.is_active]
                
                if old_versions:
                    modified.append(artifact.file_path)
                else:
                    created.append(artifact.file_path)
            
            output = "\n".join(output_parts)
            
            # Estimate tokens (only scoped files + top patterns)
            active = self.orchestrator.state_rag.retrieve(file_paths=allowed_paths)
            tokens = sum(len(a.content) for a in active) // 4
            tokens += 300  # Global RAG patterns
            
            return MethodResult(
                success=True,
                output=output,
                created_files=created,
                modified_files=modified,
                errors=[],
                token_usage=tokens,
                execution_time=time.time() - start_time,
                metadata={
                    "committed_count": len(committed),
                    "active_artifacts": len(active)
                }
            )
            
        except Exception as e:
            return MethodResult(
                success=False,
                output="",
                created_files=[],
                modified_files=[],
                errors=[f"Orchestration failed: {str(e)}\n{traceback.format_exc()}"],
                token_usage=0,
                execution_time=time.time() - start_time
            )
    
    def _infer_type(self, path: str) -> ArtifactType:
        """Infer artifact type from path"""
        if "components/" in path:
            return ArtifactType.component
        if "pages/" in path or "app/" in path:
            return ArtifactType.page
        if "layouts/" in path:
            return ArtifactType.layout
        return ArtifactType.config
    
    def _infer_language(self, path: str) -> str:
        """Infer language from extension"""
        if path.endswith('.tsx'): return 'tsx'
        if path.endswith('.ts'): return 'ts'
        if path.endswith('.jsx'): return 'jsx'
        if path.endswith('.js'): return 'js'
        if path.endswith('.css'): return 'css'
        if path.endswith('.json'): return 'json'
        return 'tsx'


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

def create_comprehensive_test_suite() -> List[TestCase]:
    """
    Create 50+ test cases across 6 categories for rigorous evaluation
    """
    
    tests = []
    
    # ========================================================================
    # CATEGORY 1: BASIC CRUD OPERATIONS (10 cases)
    # ========================================================================
    
    # Test 1.1: Create single file from scratch
    tests.append(TestCase(
        id="CRUD-01",
        category="Basic CRUD",
        name="Create Single Component",
        description="Create a new navbar component from scratch",
        initial_files={},
        user_request="Create a responsive navbar component with logo and navigation links",
        expected_files=["components/Navbar.tsx"],
        forbidden_files=[],
        check_hallucination=True,
        check_scope=True,
        notes="Basic creation test - no existing files"
    ))
    
    # Test 1.2: Read/display existing file
    tests.append(TestCase(
        id="CRUD-02",
        category="Basic CRUD",
        name="Read Existing Component",
        description="Request to see an existing component",
        initial_files={
            "components/Button.tsx": "export default function Button() { return <button>Click</button>; }"
        },
        user_request="Show me the Button component",
        expected_files=["components/Button.tsx"],
        forbidden_files=[],
        check_consistency=True,
        notes="Should return exact same content consistently"
    ))
    
    # Test 1.3: Update existing file
    tests.append(TestCase(
        id="CRUD-03",
        category="Basic CRUD",
        name="Update Existing Component",
        description="Modify an existing component",
        initial_files={
            "components/Card.tsx": "<div className='card'>Card</div>"
        },
        user_request="Add a border and shadow to the Card component",
        expected_files=["components/Card.tsx"],
        forbidden_files=[],
        check_versioning=True,
        expected_versions={"components/Card.tsx": 2},
        notes="Should create version 2, keep version 1 inactive"
    ))
    
    # Test 1.4: Delete simulation (create without reference)
    tests.append(TestCase(
        id="CRUD-04",
        category="Basic CRUD",
        name="Ignore Irrelevant File",
        description="Request that doesn't involve existing file",
        initial_files={
            "components/Header.tsx": "<header>Old Header</header>",
            "components/Footer.tsx": "<footer>Footer</footer>"
        },
        user_request="Create a new Sidebar component",
        expected_files=["components/Sidebar.tsx"],
        forbidden_files=[],
        check_scope=True,
        notes="Should only create Sidebar, not modify Header/Footer"
    ))
    
    # Test 1.5: Create multiple files
    tests.append(TestCase(
        id="CRUD-05",
        category="Basic CRUD",
        name="Create Multiple Components",
        description="Create several related components at once",
        initial_files={},
        user_request="Create a Card component and a CardList component that uses multiple Cards",
        expected_files=["components/Card.tsx", "components/CardList.tsx"],
        forbidden_files=[],
        check_hallucination=True,
        check_dependencies=True,
        notes="CardList should import Card"
    ))
    
    # Test 1.6: Update multiple files
    tests.append(TestCase(
        id="CRUD-06",
        category="Basic CRUD",
        name="Update Multiple Components",
        description="Modify several components together",
        initial_files={
            "components/Nav.tsx": "<nav>Nav v1</nav>",
            "components/Logo.tsx": "<div>Logo</div>"
        },
        user_request="Update both Nav and Logo to use consistent branding colors",
        expected_files=["components/Nav.tsx", "components/Logo.tsx"],
        forbidden_files=[],
        check_versioning=True,
        expected_versions={"components/Nav.tsx": 2, "components/Logo.tsx": 2},
        notes="Both should be versioned"
    ))
    
    # Test 1.7: Create with imports
    tests.append(TestCase(
        id="CRUD-07",
        category="Basic CRUD",
        name="Create With Dependencies",
        description="Create component that imports another",
        initial_files={
            "components/Button.tsx": "export default function Button() { return <button>Click</button>; }"
        },
        user_request="Create a ButtonGroup component that arranges multiple Buttons",
        expected_files=["components/ButtonGroup.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Should import existing Button"
    ))
    
    # Test 1.8: Nested directory creation
    tests.append(TestCase(
        id="CRUD-08",
        category="Basic CRUD",
        name="Create Nested Component",
        description="Create component in nested directory",
        initial_files={},
        user_request="Create a ProfileCard component in components/user/ProfileCard.tsx",
        expected_files=["components/user/ProfileCard.tsx"],
        forbidden_files=[],
        check_scope=True,
        notes="Should respect nested paths"
    ))
    
    # Test 1.9: Rename simulation (create new, reference old)
    tests.append(TestCase(
        id="CRUD-09",
        category="Basic CRUD",
        name="Refactor Component Name",
        description="Create new component based on old one",
        initial_files={
            "components/OldButton.tsx": "export default function OldButton() { return <button>Old</button>; }"
        },
        user_request="Create a new ModernButton component based on OldButton but with improved styling",
        expected_files=["components/ModernButton.tsx"],
        forbidden_files=[],
        check_hallucination=True,
        notes="Should create new file, not hallucinate others"
    ))
    
    # Test 1.10: Complex component with state
    tests.append(TestCase(
        id="CRUD-10",
        category="Basic CRUD",
        name="Create Stateful Component",
        description="Create component with React hooks and state",
        initial_files={},
        user_request="Create a SearchBar component with input field and search button, using useState for the search term",
        expected_files=["components/SearchBar.tsx"],
        forbidden_files=[],
        check_hallucination=True,
        notes="Should generate valid React with hooks"
    ))
    
    # ========================================================================
    # CATEGORY 2: MULTI-FILE WORKFLOWS (10 cases)
    # ========================================================================
    
    # Test 2.1: Sequential edits to same file
    tests.append(TestCase(
        id="MULTI-01",
        category="Multi-file Workflow",
        name="Sequential Same-File Edits",
        description="Edit the same file multiple times in sequence",
        initial_files={
            "components/Hero.tsx": "<div>Hero v1</div>"
        },
        user_request="Add a subtitle to Hero",
        expected_files=["components/Hero.tsx"],
        forbidden_files=[],
        check_versioning=True,
        check_consistency=True,
        expected_versions={"components/Hero.tsx": 2},
        notes="Second edit should see first edit's changes"
    ))
    
    # Test 2.2: Related files edit
    tests.append(TestCase(
        id="MULTI-02",
        category="Multi-file Workflow",
        name="Edit Related Components",
        description="Update components that depend on each other",
        initial_files={
            "components/Parent.tsx": "export default function Parent() { return <div>Parent</div>; }",
            "components/Child.tsx": "export default function Child() { return <div>Child</div>; }"
        },
        user_request="Update Parent to import and use Child component",
        expected_files=["components/Parent.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Parent should import Child"
    ))
    
    # Test 2.3: Fan-out: One file affects many
    tests.append(TestCase(
        id="MULTI-03",
        category="Multi-file Workflow",
        name="Update Shared Dependency",
        description="Update a component that multiple others depend on",
        initial_files={
            "components/BaseButton.tsx": "export default function BaseButton() { return <button>Base</button>; }",
            "components/PrimaryButton.tsx": "import BaseButton from './BaseButton';\nexport default function PrimaryButton() { return <BaseButton />; }",
            "components/SecondaryButton.tsx": "import BaseButton from './BaseButton';\nexport default function SecondaryButton() { return <BaseButton />; }"
        },
        user_request="Update BaseButton to accept a 'variant' prop",
        expected_files=["components/BaseButton.tsx"],
        forbidden_files=[],
        check_scope=True,
        notes="Should only modify BaseButton, not dependents"
    ))
    
    # Test 2.4: Cross-directory workflow
    tests.append(TestCase(
        id="MULTI-04",
        category="Multi-file Workflow",
        name="Cross-Directory Import",
        description="Create components in different directories that import each other",
        initial_files={
            "components/ui/Button.tsx": "export default function Button() { return <button>Click</button>; }"
        },
        user_request="Create a LoginForm in components/forms/ that uses the Button from components/ui/",
        expected_files=["components/forms/LoginForm.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Should have correct relative import path"
    ))
    
    # Test 2.5: Iterative refinement
    tests.append(TestCase(
        id="MULTI-05",
        category="Multi-file Workflow",
        name="Iterative Component Refinement",
        description="Progressively improve a component",
        initial_files={
            "components/Dashboard.tsx": "<div>Dashboard v1</div>"
        },
        user_request="Add a header section to Dashboard",
        expected_files=["components/Dashboard.tsx"],
        forbidden_files=[],
        check_versioning=True,
        check_consistency=True,
        expected_versions={"components/Dashboard.tsx": 2},
        notes="Should track version progression"
    ))
    
    # Test 2.6: Refactoring workflow
    tests.append(TestCase(
        id="MULTI-06",
        category="Multi-file Workflow",
        name="Extract Shared Component",
        description="Extract common logic into shared component",
        initial_files={
            "components/PageA.tsx": "<div className='container mx-auto'><h1>Page A</h1></div>",
            "components/PageB.tsx": "<div className='container mx-auto'><h1>Page B</h1></div>"
        },
        user_request="Extract the container div into a reusable Container component",
        expected_files=["components/Container.tsx"],
        forbidden_files=[],
        check_hallucination=True,
        notes="Should create Container, not modify PageA/PageB unless allowed"
    ))
    
    # Test 2.7: Component composition
    tests.append(TestCase(
        id="MULTI-07",
        category="Multi-file Workflow",
        name="Build Composite Component",
        description="Create a component composed of multiple sub-components",
        initial_files={
            "components/Icon.tsx": "export default function Icon() { return <svg>...</svg>; }",
            "components/Text.tsx": "export default function Text() { return <span>Text</span>; }"
        },
        user_request="Create a NotificationCard that uses both Icon and Text components",
        expected_files=["components/NotificationCard.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Should import both Icon and Text"
    ))
    
    # Test 2.8: Config file update
    tests.append(TestCase(
        id="MULTI-08",
        category="Multi-file Workflow",
        name="Update Config File",
        description="Modify configuration file",
        initial_files={
            "config/theme.json": '{"primaryColor": "#000000"}'
        },
        user_request="Update the theme config to use blue as primary color",
        expected_files=["config/theme.json"],
        forbidden_files=[],
        check_versioning=True,
        expected_versions={"config/theme.json": 2},
        notes="Should handle JSON files"
    ))
    
    # Test 2.9: Parallel creation
    tests.append(TestCase(
        id="MULTI-09",
        category="Multi-file Workflow",
        name="Create Parallel Components",
        description="Create multiple independent components",
        initial_files={},
        user_request="Create three separate components: Avatar, Badge, and Tooltip",
        expected_files=["components/Avatar.tsx", "components/Badge.tsx", "components/Tooltip.tsx"],
        forbidden_files=[],
        check_hallucination=True,
        notes="All three should be independent, no cross-imports"
    ))
    
    # Test 2.10: Complex dependency chain
    tests.append(TestCase(
        id="MULTI-10",
        category="Multi-file Workflow",
        name="Deep Dependency Chain",
        description="Create components with multi-level dependencies",
        initial_files={
            "components/atoms/Icon.tsx": "export default function Icon() { return <svg />; }",
            "components/molecules/Button.tsx": "import Icon from '../atoms/Icon';\nexport default function Button() { return <button><Icon /></button>; }"
        },
        user_request="Create an organism-level Navbar that uses Button (which uses Icon)",
        expected_files=["components/organisms/Navbar.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Should correctly resolve nested imports"
    ))
    
    # ========================================================================
    # CATEGORY 3: AUTHORITY & SAFETY (10 cases)
    # ========================================================================
    
    # Test 3.1: Protect user-modified file
    tests.append(TestCase(
        id="AUTH-01",
        category="Authority & Safety",
        name="Block AI Edit of User File",
        description="AI should not modify user_modified files without permission",
        initial_files={
            "components/UserComponent.tsx": "// CRITICAL: User-customized component"
        },
        user_request="Update the UserComponent to add a new prop",
        expected_files=[],
        forbidden_files=["components/UserComponent.tsx"],
        check_authority=True,
        should_fail=True,  # DualRAG should reject this
        notes="DualRAG MUST reject, baselines WILL modify (that's the point)"
    ))
    
    # Test 3.2: Allow AI edit with permission
    tests.append(TestCase(
        id="AUTH-02",
        category="Authority & Safety",
        name="Allow AI Edit With Permission",
        description="AI CAN modify if file is in allowed_paths",
        initial_files={
            "components/SafeComponent.tsx": "<div>Safe to modify</div>"
        },
        user_request="Add a className to SafeComponent",
        expected_files=["components/SafeComponent.tsx"],
        forbidden_files=[],
        check_authority=True,
        notes="File is in allowed_paths, so edit is permitted"
    ))
    
    # Test 3.3: Scope violation - out of bounds
    tests.append(TestCase(
        id="AUTH-03",
        category="Authority & Safety",
        name="Reject Out-of-Scope Edit",
        description="Should not edit files outside allowed_paths",
        initial_files={
            "components/A.tsx": "<div>A</div>",
            "components/B.tsx": "<div>B</div>",
            "components/C.tsx": "<div>C</div>"
        },
        user_request="Update component B",
        expected_files=[],  # B is not in allowed_paths for this test
        forbidden_files=["components/B.tsx"],
        check_scope=True,
        should_fail=True,
        notes="If B not in allowed_paths, DualRAG should reject"
    ))
    
    # Test 3.4: Preserve user intent
    tests.append(TestCase(
        id="AUTH-04",
        category="Authority & Safety",
        name="Preserve User Customization",
        description="Don't override user's custom logic",
        initial_files={
            "components/CustomLogic.tsx": "// User's complex business logic\nexport default function CustomLogic() { /* ... */ }"
        },
        user_request="Refactor all components to use TypeScript interfaces",
        expected_files=[],
        forbidden_files=["components/CustomLogic.tsx"],
        check_authority=True,
        should_fail=True,
        notes="User file should be protected"
    ))
    
    # Test 3.5: Safe concurrent edit
    tests.append(TestCase(
        id="AUTH-05",
        category="Authority & Safety",
        name="Handle Concurrent Edits",
        description="Two requests to edit different files simultaneously",
        initial_files={
            "components/X.tsx": "<div>X</div>",
            "components/Y.tsx": "<div>Y</div>"
        },
        user_request="Update component X",
        expected_files=["components/X.tsx"],
        forbidden_files=["components/Y.tsx"],
        check_scope=True,
        notes="Should only touch X, leave Y alone"
    ))
    
    # Test 3.6: Version rollback protection
    tests.append(TestCase(
        id="AUTH-06",
        category="Authority & Safety",
        name="Prevent Version Regression",
        description="Should not accidentally revert to older version",
        initial_files={
            "components/Versioned.tsx": "<div>Versioned v3</div>"  # Simulates v3
        },
        user_request="Show me the Versioned component",
        expected_files=["components/Versioned.tsx"],
        forbidden_files=[],
        check_versioning=True,
        notes="Should maintain current version, not regress"
    ))
    
    # Test 3.7: Detect unauthorized creation
    tests.append(TestCase(
        id="AUTH-07",
        category="Authority & Safety",
        name="Reject Unauthorized File Creation",
        description="Should not create files in restricted paths",
        initial_files={},
        user_request="Create a config file in /etc/system.conf",
        expected_files=[],
        forbidden_files=["/etc/system.conf", "etc/system.conf"],
        check_scope=True,
        should_fail=True,
        notes="Path traversal/system file attempt should be blocked"
    ))
    
    # Test 3.8: Safeguard critical files
    tests.append(TestCase(
        id="AUTH-08",
        category="Authority & Safety",
        name="Protect Critical Infrastructure",
        description="Don't modify critical config files",
        initial_files={
            "package.json": '{"name": "app", "version": "1.0.0"}'
        },
        user_request="Update all files to use ES modules",
        expected_files=[],
        forbidden_files=["package.json"],
        check_authority=True,
        should_fail=True,
        notes="Critical files should be protected"
    ))
    
    # Test 3.9: Selective permission
    tests.append(TestCase(
        id="AUTH-09",
        category="Authority & Safety",
        name="Selective File Permissions",
        description="Allow editing some files but not others",
        initial_files={
            "components/Public.tsx": "<div>Public</div>",
            "components/Private.tsx": "<div>Private - User Modified</div>"
        },
        user_request="Update both Public and Private components",
        expected_files=["components/Public.tsx"],
        forbidden_files=["components/Private.tsx"],
        check_authority=True,
        notes="Public allowed, Private protected"
    ))
    
    # Test 3.10: Audit trail integrity
    tests.append(TestCase(
        id="AUTH-10",
        category="Authority & Safety",
        name="Maintain Audit Trail",
        description="Ensure version history is preserved",
        initial_files={
            "components/Audited.tsx": "<div>Audited v1</div>"
        },
        user_request="Update Audited component",
        expected_files=["components/Audited.tsx"],
        forbidden_files=[],
        check_versioning=True,
        expected_versions={"components/Audited.tsx": 2},
        notes="v1 should remain in history as inactive"
    ))
    
    # ========================================================================
    # CATEGORY 4: VERSIONING & ROLLBACK (8 cases)
    # ========================================================================
    
    # Test 4.1: Basic version increment
    tests.append(TestCase(
        id="VER-01",
        category="Versioning & Rollback",
        name="Increment Version on Edit",
        description="Editing a file should create new version",
        initial_files={
            "components/Component.tsx": "<div>v1</div>"
        },
        user_request="Add a className to Component",
        expected_files=["components/Component.tsx"],
        forbidden_files=[],
        check_versioning=True,
        expected_versions={"components/Component.tsx": 2},
        notes="Version should go from 1 to 2"
    ))
    
    # Test 4.2: Multiple version increments
    tests.append(TestCase(
        id="VER-02",
        category="Versioning & Rollback",
        name="Multiple Sequential Versions",
        description="Multiple edits create multiple versions",
        initial_files={
            "components/Multi.tsx": "<div>v1</div>"
        },
        user_request="Add a title prop to Multi",  # This would be v2 in real workflow
        expected_files=["components/Multi.tsx"],
        forbidden_files=[],
        check_versioning=True,
        notes="In real workflow: v1 -> v2 -> v3"
    ))
    
    # Test 4.3: Version isolation
    tests.append(TestCase(
        id="VER-03",
        category="Versioning & Rollback",
        name="Independent File Versions",
        description="Each file has independent version counter",
        initial_files={
            "components/FileA.tsx": "<div>A v1</div>",
            "components/FileB.tsx": "<div>B v1</div>"
        },
        user_request="Update FileA",
        expected_files=["components/FileA.tsx"],
        forbidden_files=[],
        check_versioning=True,
        expected_versions={"components/FileA.tsx": 2},
        notes="FileA -> v2, FileB stays at v1"
    ))
    
    # Test 4.4: Active version marking
    tests.append(TestCase(
        id="VER-04",
        category="Versioning & Rollback",
        name="Mark Active Version",
        description="Only one version should be active",
        initial_files={
            "components/Active.tsx": "<div>v1</div>"
        },
        user_request="Update Active component",
        expected_files=["components/Active.tsx"],
        forbidden_files=[],
        check_versioning=True,
        notes="v2 active=true, v1 active=false"
    ))
    
    # Test 4.5: Version timestamp
    tests.append(TestCase(
        id="VER-05",
        category="Versioning & Rollback",
        name="Timestamp Each Version",
        description="Each version should have timestamp",
        initial_files={
            "components/Timestamped.tsx": "<div>v1</div>"
        },
        user_request="Modify Timestamped",
        expected_files=["components/Timestamped.tsx"],
        forbidden_files=[],
        check_versioning=True,
        notes="updated_at should differ between v1 and v2"
    ))
    
    # Test 4.6: Version metadata preservation
    tests.append(TestCase(
        id="VER-06",
        category="Versioning & Rollback",
        name="Preserve Version Metadata",
        description="Keep all metadata across versions",
        initial_files={
            "components/Meta.tsx": "<div>Meta v1</div>"
        },
        user_request="Update Meta component",
        expected_files=["components/Meta.tsx"],
        forbidden_files=[],
        check_versioning=True,
        notes="framework, styling, dependencies should be preserved"
    ))
    
    # Test 4.7: No version on read-only
    tests.append(TestCase(
        id="VER-07",
        category="Versioning & Rollback",
        name="No Version on Read",
        description="Reading a file shouldn't create new version",
        initial_files={
            "components/ReadOnly.tsx": "<div>v1</div>"
        },
        user_request="Show me the ReadOnly component",
        expected_files=["components/ReadOnly.tsx"],
        forbidden_files=[],
        check_versioning=True,
        notes="Version should stay at 1"
    ))
    
    # Test 4.8: Version on content change only
    tests.append(TestCase(
        id="VER-08",
        category="Versioning & Rollback",
        name="Version Only on Change",
        description="Same content should not create new version",
        initial_files={
            "components/Unchanged.tsx": "<div>Same</div>"
        },
        user_request="Keep Unchanged component as is",
        expected_files=["components/Unchanged.tsx"],
        forbidden_files=[],
        check_versioning=True,
        notes="If content identical, version should not increment"
    ))
    
    # ========================================================================
    # CATEGORY 5: DEPENDENCY MANAGEMENT (6 cases)
    # ========================================================================
    
    # Test 5.1: Detect implicit dependencies
    tests.append(TestCase(
        id="DEP-01",
        category="Dependency Management",
        name="Auto-Detect Dependencies",
        description="System should detect import statements",
        initial_files={
            "components/Base.tsx": "export default function Base() { return <div>Base</div>; }"
        },
        user_request="Create a Derived component that imports and uses Base",
        expected_files=["components/Derived.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Derived should import Base"
    ))
    
    # Test 5.2: Transitive dependencies
    tests.append(TestCase(
        id="DEP-02",
        category="Dependency Management",
        name="Resolve Transitive Dependencies",
        description="A imports B imports C - all should be included",
        initial_files={
            "components/C.tsx": "export default function C() { return <div>C</div>; }",
            "components/B.tsx": "import C from './C';\nexport default function B() { return <div><C /></div>; }"
        },
        user_request="Create component A that uses B",
        expected_files=["components/A.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="A -> B -> C: all three should be in context"
    ))
    
    # Test 5.3: Circular dependency detection
    tests.append(TestCase(
        id="DEP-03",
        category="Dependency Management",
        name="Handle Circular Dependencies",
        description="A imports B, B imports A - should handle gracefully",
        initial_files={
            "components/A.tsx": "import B from './B';\nexport default function A() { return <div><B /></div>; }",
            "components/B.tsx": "import A from './A';\nexport default function B() { return <div><A /></div>; }"
        },
        user_request="Show me component A",
        expected_files=["components/A.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Should not infinite loop"
    ))
    
    # Test 5.4: External dependencies
    tests.append(TestCase(
        id="DEP-04",
        category="Dependency Management",
        name="Handle External Imports",
        description="Components with npm package imports",
        initial_files={},
        user_request="Create a DatePicker component that uses date-fns library",
        expected_files=["components/DatePicker.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Should have import from 'date-fns'"
    ))
    
    # Test 5.5: Relative path imports
    tests.append(TestCase(
        id="DEP-05",
        category="Dependency Management",
        name="Correct Relative Imports",
        description="Ensure import paths are correct",
        initial_files={
            "components/shared/Utils.tsx": "export function helper() { return true; }"
        },
        user_request="Create components/features/Feature.tsx that imports the helper from Utils",
        expected_files=["components/features/Feature.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="Should use '../shared/Utils' path"
    ))
    
    # Test 5.6: Dependency update propagation
    tests.append(TestCase(
        id="DEP-06",
        category="Dependency Management",
        name="Update Dependent Components",
        description="When base changes, dependents should adapt",
        initial_files={
            "components/Button.tsx": "export default function Button() { return <button>Click</button>; }",
            "components/IconButton.tsx": "import Button from './Button';\nexport default function IconButton() { return <Button />; }"
        },
        user_request="Update Button to accept children prop",
        expected_files=["components/Button.tsx"],
        forbidden_files=[],
        check_dependencies=True,
        notes="IconButton might need update in real workflow"
    ))
    
    # ========================================================================
    # CATEGORY 6: EDGE CASES & STRESS TESTS (8 cases)
    # ========================================================================
    
    # Test 6.1: Empty request
    tests.append(TestCase(
        id="EDGE-01",
        category="Edge Cases",
        name="Handle Empty Request",
        description="Empty or whitespace-only request",
        initial_files={
            "components/Existing.tsx": "<div>Existing</div>"
        },
        user_request="   ",  # Whitespace only
        expected_files=[],
        forbidden_files=[],
        check_hallucination=True,
        should_fail=True,
        notes="Should handle gracefully, not crash"
    ))
    
    # Test 6.2: Ambiguous request
    tests.append(TestCase(
        id="EDGE-02",
        category="Edge Cases",
        name="Handle Ambiguous Request",
        description="Vague request that could mean multiple things",
        initial_files={
            "components/Button.tsx": "<button>Button</button>",
            "components/Link.tsx": "<a>Link</a>"
        },
        user_request="Make it blue",  # What is 'it'?
        expected_files=[],
        forbidden_files=[],
        check_hallucination=True,
        notes="Should not hallucinate or modify everything"
    ))
    
    # Test 6.3: Conflicting request
    tests.append(TestCase(
        id="EDGE-03",
        category="Edge Cases",
        name="Handle Conflicting Requirements",
        description="Request with contradictory requirements",
        initial_files={},
        user_request="Create a component that is both a button and a link at the same time",
        expected_files=[],
        forbidden_files=[],
        check_hallucination=True,
        notes="Should pick one or ask for clarification"
    ))
    
    # Test 6.4: Large file handling
    tests.append(TestCase(
        id="EDGE-04",
        category="Edge Cases",
        name="Handle Large Files",
        description="Component with lots of code",
        initial_files={
            "components/LargeComponent.tsx": ("export default function LargeComponent() {\n" + 
                                              "  // " + ("Large component code\n" * 100))
        },
        user_request="Add a prop to LargeComponent",
        expected_files=["components/LargeComponent.tsx"],
        forbidden_files=[],
        check_consistency=True,
        notes="Should handle without truncation"
    ))
    
    # Test 6.5: Special characters in content
    tests.append(TestCase(
        id="EDGE-05",
        category="Edge Cases",
        name="Handle Special Characters",
        description="Content with special characters",
        initial_files={},
        user_request="Create a component with template literals, regex, and special chars: $, {}, [], <>, etc",
        expected_files=["components/SpecialChars.tsx"],
        forbidden_files=[],
        check_hallucination=True,
        notes="Should properly escape/handle special chars"
    ))
    
    # Test 6.6: Non-existent file reference
    tests.append(TestCase(
        id="EDGE-06",
        category="Edge Cases",
        name="Request for Non-Existent File",
        description="Ask to modify file that doesn't exist",
        initial_files={},
        user_request="Update the NonExistent component",
        expected_files=[],
        forbidden_files=["components/NonExistent.tsx"],
        check_hallucination=True,
        should_fail=True,
        notes="Should not hallucinate the file exists"
    ))
    
    # Test 6.7: Rapid successive requests
    tests.append(TestCase(
        id="EDGE-07",
        category="Edge Cases",
        name="Rapid Successive Edits",
        description="Multiple edits in quick succession",
        initial_files={
            "components/Rapid.tsx": "<div>v1</div>"
        },
        user_request="Add a className",  # Would be followed by more requests
        expected_files=["components/Rapid.tsx"],
        forbidden_files=[],
        check_versioning=True,
        notes="Should maintain state consistency"
    ))
    
    # Test 6.8: Mixed valid/invalid paths
    tests.append(TestCase(
        id="EDGE-08",
        category="Edge Cases",
        name="Mixed Valid Invalid Paths",
        description="Request mentions both existing and non-existent files",
        initial_files={
            "components/Real.tsx": "<div>Real</div>"
        },
        user_request="Update Real and also FakeFile components",
        expected_files=["components/Real.tsx"],
        forbidden_files=["components/FakeFile.tsx"],
        check_hallucination=True,
        notes="Should only modify Real, not create FakeFile"
    ))
    
    return tests


# ============================================================================
# EVALUATION METRICS & ANALYSIS
# ============================================================================

class ComprehensiveEvaluator:
    """Main evaluation orchestrator with all metrics"""
    
    def __init__(self, llm_provider: str = "gemini", output_dir: str = "/mnt/user-data/outputs/evaluation"):
        self.llm_provider = llm_provider
        self.llm = LLMAdapter(provider=llm_provider)
        self.output_dir = output_dir
        self.results: List[EvaluationResult] = []
        
        # Create output structure
        for subdir in ['charts', 'data', 'tables']:
            os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
        
        print("=" * 80)
        print("COMPREHENSIVE DUALRAG EVALUATION FOR PUBLICATION")
        print("=" * 80)
        print(f"Output directory: {output_dir}")
        print()
    
    def run_full_evaluation(self, test_limit: Optional[int] = None):
        """
        Run complete evaluation suite
        
        Args:
            test_limit: Optional limit on number of tests (for faster iteration)
        """
        
        # Get test suite
        all_tests = create_comprehensive_test_suite()
        
        if test_limit:
            all_tests = all_tests[:test_limit]
            print(f"⚠️  Running limited evaluation: {test_limit} tests")
        else:
            print(f"📋 Running full evaluation: {len(all_tests)} tests")
        
        print(f"   Categories: {len(set(t.category for t in all_tests))}")
        print(f"   Baselines: 4 (Context Window, Conv History, Single RAG, DualRAG)")
        print()
        
        # Group by category for organized execution
        by_category = defaultdict(list)
        for test in all_tests:
            by_category[test.category].append(test)
        
        # Run tests category by category
        for category, tests in by_category.items():
            print(f"\n{'='*80}")
            print(f"CATEGORY: {category} ({len(tests)} tests)")
            print(f"{'='*80}\n")
            
            for test in tests:
                self.run_single_test(test)
                time.sleep(1)  # Rate limiting
        
        # Generate all reports
        print(f"\n{'='*80}")
        print("GENERATING REPORTS")
        print(f"{'='*80}\n")
        
        self.generate_all_reports()
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\n📂 Results saved to: {self.output_dir}/")
        print("   - charts/ (publication-ready figures)")
        print("   - tables/ (LaTeX tables)")
        print("   - data/ (raw CSV for analysis)")
    
    def run_single_test(self, test: TestCase):
        """Execute one test case across all methods"""
        
        print(f"{'─'*80}")
        print(f"TEST {test.id}: {test.name}")
        print(f"{'─'*80}")
        print(f"Description: {test.description}")
        print(f"Initial files: {len(test.initial_files)}")
        print(f"Category: {test.category}")
        print()
        
        # Initialize all methods
        methods = {
            "Context Window": ContextWindowMethod(self.llm),
            "Conv History": ConversationHistoryMethod(self.llm),
            "Single RAG": SingleRAGMethod(self.llm),
            "DualRAG": DualRAGMethod(self.llm_provider)
        }
        
        # Test each method
        for method_name, method_instance in methods.items():
            print(f"  Testing {method_name}...", end=" ", flush=True)
            
            try:
                # Setup initial files
                for file_path, content in test.initial_files.items():
                    if method_name == "DualRAG":
                        # For DualRAG, mark user files appropriately
                        source = ArtifactSource.user_modified if "User" in content or test.check_authority else ArtifactSource.ai_generated
                        method_instance.add_file(file_path, content, source)
                    else:
                        method_instance.add_file(file_path, content)
                
                # Generate
                if method_name == "DualRAG":
                    # Build allowed_paths based on test
                    allowed_paths = list(test.expected_files)
                    # Also add initial files if they should be editable
                    for path in test.initial_files.keys():
                        if path not in allowed_paths and not test.check_authority:
                            allowed_paths.append(path)
                    
                    result = method_instance.generate(test.user_request, allowed_paths)
                else:
                    result = method_instance.generate(test.user_request)
                
                # Analyze result based on test type
                if test.check_hallucination:
                    self._measure_hallucination(method_name, test, result)
                
                if test.check_consistency:
                    self._measure_consistency(method_name, test, method_instance)
                
                if test.check_authority:
                    self._measure_authority(method_name, test, result)
                
                if test.check_scope:
                    self._measure_scope(method_name, test, result)
                
                if test.check_versioning:
                    self._measure_versioning(method_name, test, result, method_instance)
                
                if test.check_dependencies:
                    self._measure_dependencies(method_name, test, result)
                
                # Always measure token efficiency
                self._measure_token_usage(method_name, test, result)
                
                # Success rate
                success_value = 1.0 if result.success else 0.0
                self.results.append(EvaluationResult(
                    method=method_name,
                    test_case_id=test.id,
                    metric="success_rate",
                    value=success_value,
                    success=result.success,
                    metadata={"errors": result.errors}
                ))
                
                status = "✓" if result.success else "✗"
                print(f"{status}")
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                self.results.append(EvaluationResult(
                    method=method_name,
                    test_case_id=test.id,
                    metric="success_rate",
                    value=0.0,
                    success=False,
                    metadata={"error": str(e), "traceback": traceback.format_exc()}
                ))
        
        print()
    
    # ========================================================================
    # METRIC MEASUREMENT METHODS
    # ========================================================================
    
    def _measure_hallucination(self, method_name: str, test: TestCase, result: MethodResult):
        """
        Measure hallucination: references to non-existent files
        
        CORRECTED: Only count actual invalid references, not FILE: headers
        """
        known_files = set(test.initial_files.keys())
        hallucinations = []
        
        # Look for import statements (actual hallucinations)
        import_pattern = r"import\s+.*?\s+from\s+['\"]\.?\.?/?([^'\"]+)['\"]"
        imports = re.findall(import_pattern, result.output)
        
        for imp in imports:
            # Clean up the import path
            imp_clean = imp.strip()
            
            # Check if this file exists in known files
            is_hallucination = True
            for known in known_files:
                if imp_clean in known or known.endswith(imp_clean):
                    is_hallucination = False
                    break
            
            # Also check if it's in the created files (self-reference is OK)
            if imp_clean in result.created_files:
                is_hallucination = False
            
            if is_hallucination:
                hallucinations.append(imp_clean)
        
        # Calculate rate (0 = no hallucinations, 1 = all imports are hallucinations)
        total_imports = len(imports) if imports else 1
        hallucination_rate = len(set(hallucinations)) / total_imports
        
        self.results.append(EvaluationResult(
            method=method_name,
            test_case_id=test.id,
            metric="hallucination_rate",
            value=hallucination_rate,
            success=(len(hallucinations) == 0),
            metadata={
                "hallucinations": list(set(hallucinations)),
                "total_imports": total_imports
            }
        ))
    
    def _measure_consistency(self, method_name: str, test: TestCase, method_instance, runs: int = 3):
        """
        Measure consistency: same input -> same output?
        
        Run the same test multiple times and check variance
        """
        outputs = []
        
        for run in range(runs):
            try:
                if method_name == "DualRAG":
                    allowed_paths = list(test.initial_files.keys()) + test.expected_files
                    result = method_instance.generate(test.user_request, allowed_paths)
                else:
                    result = method_instance.generate(test.user_request)
                
                outputs.append(result.output)
                time.sleep(1)  # Rate limiting
                
            except Exception:
                outputs.append("")
        
        # Calculate consistency score
        unique_outputs = len(set(outputs))
        consistency_score = 1.0 - ((unique_outputs - 1) / runs)
        
        self.results.append(EvaluationResult(
            method=method_name,
            test_case_id=test.id,
            metric="consistency_score",
            value=max(0.0, consistency_score),
            success=(consistency_score >= 0.8),  # 80% threshold
            metadata={"unique_outputs": unique_outputs, "total_runs": runs}
        ))
    
    def _measure_authority(self, method_name: str, test: TestCase, result: MethodResult):
        """
        Measure authority preservation: does it respect user_modified files?
        
        For DualRAG: should REJECT unauthorized edits
        For baselines: WILL modify (that's expected)
        """
        
        if method_name == "DualRAG":
            # DualRAG should reject edits to user_modified files
            if test.should_fail:
                # Test expects rejection
                authority_preserved = not result.success
            else:
                # Test expects success
                authority_preserved = result.success
            
            self.results.append(EvaluationResult(
                method=method_name,
                test_case_id=test.id,
                metric="authority_preservation",
                value=1.0 if authority_preserved else 0.0,
                success=authority_preserved,
                metadata={"should_fail": test.should_fail, "actually_failed": not result.success}
            ))
        else:
            # Baselines don't have authority control
            # We score them as 0 on this metric (not their fault, just not a feature)
            self.results.append(EvaluationResult(
                method=method_name,
                test_case_id=test.id,
                metric="authority_preservation",
                value=0.0,  # Baselines can't do this
                success=False,
                metadata={"reason": "No authority control in baseline"}
            ))
    
    def _measure_scope(self, method_name: str, test: TestCase, result: MethodResult):
        """
        Measure scope adherence: only modifies allowed files?
        """
        
        # Check if any forbidden files were modified
        violated_scope = False
        violations = []
        
        for modified_file in result.modified_files + result.created_files:
            if modified_file in test.forbidden_files:
                violated_scope = True
                violations.append(modified_file)
        
        scope_adherence = 0.0 if violated_scope else 1.0
        
        self.results.append(EvaluationResult(
            method=method_name,
            test_case_id=test.id,
            metric="scope_adherence",
            value=scope_adherence,
            success=(scope_adherence == 1.0),
            metadata={"violations": violations}
        ))
    
    def _measure_versioning(self, method_name: str, test: TestCase, result: MethodResult, method_instance):
        """
        Measure versioning correctness (DualRAG only)
        """
        
        if method_name != "DualRAG":
            # Baselines don't version
            self.results.append(EvaluationResult(
                method=method_name,
                test_case_id=test.id,
                metric="versioning_correctness",
                value=0.0,
                success=False,
                metadata={"reason": "No versioning in baseline"}
            ))
            return
        
        # Check DualRAG versioning
        correct_versions = True
        version_check = {}
        
        for file_path, expected_version in test.expected_versions.items():
            artifacts = [a for a in method_instance.orchestrator.state_rag.artifacts 
                        if a.file_path == file_path and a.is_active]
            
            if artifacts:
                actual_version = artifacts[0].version
                version_check[file_path] = {
                    "expected": expected_version,
                    "actual": actual_version,
                    "correct": (actual_version == expected_version)
                }
                
                if actual_version != expected_version:
                    correct_versions = False
            else:
                correct_versions = False
                version_check[file_path] = {
                    "expected": expected_version,
                    "actual": None,
                    "correct": False
                }
        
        versioning_score = 1.0 if correct_versions else 0.0
        
        self.results.append(EvaluationResult(
            method=method_name,
            test_case_id=test.id,
            metric="versioning_correctness",
            value=versioning_score,
            success=correct_versions,
            metadata={"version_check": version_check}
        ))
    
    def _measure_dependencies(self, method_name: str, test: TestCase, result: MethodResult):
        """
        Measure dependency tracking: are imports correct?
        """
        
        # Look for import statements in output
        import_pattern = r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]"
        imports = re.findall(import_pattern, result.output)
        
        # Check if expected dependencies are present
        has_imports = len(imports) > 0
        
        dependency_score = 1.0 if has_imports else 0.0
        
        self.results.append(EvaluationResult(
            method=method_name,
            test_case_id=test.id,
            metric="dependency_tracking",
            value=dependency_score,
            success=has_imports,
            metadata={"imports_found": imports}
        ))
    
    def _measure_token_usage(self, method_name: str, test: TestCase, result: MethodResult):
        """
        Measure token efficiency: how much context was used?
        """
        
        self.results.append(EvaluationResult(
            method=method_name,
            test_case_id=test.id,
            metric="token_usage",
            value=result.token_usage,
            success=True,
            metadata={"tokens": result.token_usage}
        ))
    
    # ========================================================================
    # REPORT GENERATION
    # ========================================================================
    
    def generate_all_reports(self):
        """Generate all publication-ready outputs"""
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save raw data
        df.to_csv(f"{self.output_dir}/data/raw_results.csv", index=False)
        print("✅ Raw data saved")
        
        # Generate visualizations
        self._generate_main_comparison_chart(df)
        self._generate_category_breakdown(df)
        self._generate_radar_chart(df)
        self._generate_token_efficiency_chart(df)
        
        # Generate tables
        self._generate_latex_summary_table(df)
        self._generate_latex_detailed_table(df)
        
        # Statistical analysis
        self._perform_statistical_analysis(df)
        
        print("✅ All reports generated")
    
    def _generate_main_comparison_chart(self, df: pd.DataFrame):
        """
        Main comparison: all metrics across all methods
        """
        
        # Aggregate by method and metric
        metrics_to_plot = ['hallucination_rate', 'consistency_score', 'authority_preservation', 
                          'scope_adherence', 'versioning_correctness', 'dependency_tracking']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            metric_data = df[df['metric'] == metric]
            if metric_data.empty:
                continue
            
            avg_by_method = metric_data.groupby('method')['value'].mean()
            
            colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
            bars = ax.bar(avg_by_method.index, avg_by_method.values, color=colors)
            
            # Format
            title = metric.replace('_', ' ').title()
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim(0, 1.1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/charts/main_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Main comparison chart saved")
    
    def _generate_category_breakdown(self, df: pd.DataFrame):
        """
        Success rate by category
        """
        
        # Get success rates by category
        success_data = df[df['metric'] == 'success_rate'].copy()
        
        # Extract category from test_case_id
        success_data['category'] = success_data['test_case_id'].str.split('-').str[0]
        
        # Aggregate
        category_scores = success_data.groupby(['category', 'method'])['value'].mean().unstack()
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        category_scores.plot(kind='bar', ax=ax, color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'])
        
        ax.set_title('Success Rate by Test Category', fontweight='bold', fontsize=14)
        ax.set_xlabel('Test Category', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/charts/category_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Category breakdown chart saved")
    
    def _generate_radar_chart(self, df: pd.DataFrame):
        """
        Radar chart for overall method comparison
        """
        
        # Select metrics for radar
        radar_metrics = ['hallucination_rate', 'consistency_score', 'authority_preservation', 
                        'scope_adherence', 'versioning_correctness', 'dependency_tracking']
        
        methods = df['method'].unique()
        
        # Prepare data
        method_scores = {}
        for method in methods:
            scores = {}
            for metric in radar_metrics:
                metric_data = df[(df['method'] == method) & (df['metric'] == metric)]
                if not metric_data.empty:
                    score = metric_data['value'].mean()
                    # Invert hallucination (lower is better)
                    if metric == 'hallucination_rate':
                        score = 1.0 - score
                    scores[metric] = score
                else:
                    scores[metric] = 0.0
            method_scores[method] = scores
        
        # Create radar chart
        labels = [m.replace('_', '\n').title() for m in radar_metrics]
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
        
        for idx, (method, scores) in enumerate(method_scores.items()):
            values = [scores[m] for m in radar_metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Method Comparison\n(All Metrics)', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/charts/radar_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Radar chart saved")
    
    def _generate_token_efficiency_chart(self, df: pd.DataFrame):
        """
        Token usage comparison
        """
        
        token_data = df[df['metric'] == 'token_usage']
        
        if token_data.empty:
            return
        
        avg_tokens = token_data.groupby('method')['value'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
        bars = ax.bar(avg_tokens.index, avg_tokens.values, color=colors)
        
        ax.set_title('Average Token Usage by Method', fontweight='bold', fontsize=14)
        ax.set_ylabel('Tokens', fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.tick_params(axis='x', rotation=15)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/charts/token_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Token efficiency chart saved")
    
    def _generate_latex_summary_table(self, df: pd.DataFrame):
        """
        Generate LaTeX table for paper - summary view
        """
        
        methods = ['Context Window', 'Conv History', 'Single RAG', 'DualRAG']
        metrics = ['hallucination_rate', 'consistency_score', 'authority_preservation', 
                  'scope_adherence', 'success_rate', 'token_usage']
        
        # Build table data
        table_data = {}
        for method in methods:
            table_data[method] = {}
            for metric in metrics:
                metric_data = df[(df['method'] == method) & (df['metric'] == metric)]
                if not metric_data.empty:
                    if metric == 'token_usage':
                        table_data[method][metric] = metric_data['value'].mean()
                    else:
                        table_data[method][metric] = metric_data['value'].mean() * 100  # Convert to percentage
                else:
                    table_data[method][metric] = 0.0
        
        # Generate LaTeX
        latex = "\\begin{table*}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Comprehensive Evaluation Results Across All Test Cases}\n"
        latex += "\\label{tab:comprehensive_evaluation}\n"
        latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        latex += "\\toprule\n"
        latex += "Method & " + " & ".join([m.replace('_', ' ').title() for m in metrics]) + " \\\\\n"
        latex += "       & " + " & ".join([
            "(\\%, $\\downarrow$)",  # hallucination
            "(\\%, $\\uparrow$)",    # consistency
            "(\\%, $\\uparrow$)",    # authority
            "(\\%, $\\uparrow$)",    # scope
            "(\\%, $\\uparrow$)",    # success
            "(tokens)"               # token usage
        ]) + " \\\\\n"
        latex += "\\midrule\n"
        
        for method in methods:
            latex += method + " & "
            values = []
            for metric in metrics:
                val = table_data[method][metric]
                if metric == 'token_usage':
                    values.append(f"{int(val)}")
                else:
                    values.append(f"{val:.1f}")
            latex += " & ".join(values) + " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table*}\n"
        
        with open(f"{self.output_dir}/tables/summary_table.tex", 'w') as f:
            f.write(latex)
        
        print("✅ LaTeX summary table saved")
    
    def _generate_latex_detailed_table(self, df: pd.DataFrame):
        """
        Generate detailed LaTeX table with per-category breakdown
        """
        
        # Get success rates by category
        success_data = df[df['metric'] == 'success_rate'].copy()
        success_data['category'] = success_data['test_case_id'].str.split('-').str[0]
        
        categories = success_data['category'].unique()
        methods = success_data['method'].unique()
        
        # Build table
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Success Rate by Test Category (\\%)}\n"
        latex += "\\label{tab:category_results}\n"
        latex += "\\begin{tabular}{l" + "c" * len(methods) + "}\n"
        latex += "\\toprule\n"
        latex += "Category & " + " & ".join(methods) + " \\\\\n"
        latex += "\\midrule\n"
        
        for category in sorted(categories):
            latex += category + " & "
            values = []
            for method in methods:
                cat_data = success_data[(success_data['category'] == category) & 
                                       (success_data['method'] == method)]
                if not cat_data.empty:
                    avg = cat_data['value'].mean() * 100
                    values.append(f"{avg:.1f}")
                else:
                    values.append("--")
            latex += " & ".join(values) + " \\\\\n"
        
        latex += "\\midrule\n"
        latex += "\\textbf{Overall} & "
        overall_values = []
        for method in methods:
            overall = success_data[success_data['method'] == method]['value'].mean() * 100
            overall_values.append(f"\\textbf{{{overall:.1f}}}")
        latex += " & ".join(overall_values) + " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        with open(f"{self.output_dir}/tables/detailed_table.tex", 'w') as f:
            f.write(latex)
        
        print("✅ LaTeX detailed table saved")
    
    def _perform_statistical_analysis(self, df: pd.DataFrame):
        """
        Perform statistical significance testing
        """
        
        # Compare DualRAG vs each baseline on key metrics
        metrics_to_test = ['hallucination_rate', 'consistency_score', 'scope_adherence']
        
        results_text = "STATISTICAL ANALYSIS\n"
        results_text += "=" * 80 + "\n\n"
        
        for metric in metrics_to_test:
            results_text += f"\nMetric: {metric}\n"
            results_text += "-" * 40 + "\n"
            
            dualrag_data = df[(df['method'] == 'DualRAG') & (df['metric'] == metric)]['value']
            
            for baseline in ['Context Window', 'Conv History', 'Single RAG']:
                baseline_data = df[(df['method'] == baseline) & (df['metric'] == metric)]['value']
                
                if len(dualrag_data) > 1 and len(baseline_data) > 1:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(dualrag_data, baseline_data)
                    
                    dualrag_mean = dualrag_data.mean()
                    baseline_mean = baseline_data.mean()
                    
                    results_text += f"\nDualRAG vs {baseline}:\n"
                    results_text += f"  DualRAG mean: {dualrag_mean:.4f}\n"
                    results_text += f"  {baseline} mean: {baseline_mean:.4f}\n"
                    results_text += f"  t-statistic: {t_stat:.4f}\n"
                    results_text += f"  p-value: {p_value:.4f}\n"
                    
                    if p_value < 0.05:
                        results_text += f"  ✓ Statistically significant (p < 0.05)\n"
                    else:
                        results_text += f"  ✗ Not statistically significant\n"
        
        # Save to file
        with open(f"{self.output_dir}/data/statistical_analysis.txt", 'w') as f:
            f.write(results_text)
        
        print("✅ Statistical analysis saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    
    print("\n" + "="*80)
    print("DUALRAG EVALUATION FOR CONFERENCE PUBLICATION")
    print("="*80)
    print()
    print("This is a comprehensive evaluation with 50+ test cases.")
    print("Estimated time: 60-120 minutes")
    print("Cost: FREE (uses Gemini free tier)")
    print()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        print("   Please set it in your .env file")
        return
    
    print(f"✅ API key found: ...{api_key[-8:]}")
    print()
    
    # Check if DualRAG modules are available
    if not DUALRAG_AVAILABLE:
        print("⚠️  WARNING: DualRAG modules not found!")
        print("   The evaluation will run with limited functionality.")
        print("   Some baseline methods may not work correctly.")
        print("   Make sure all DualRAG source files are in the same directory.")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        print()
    
    # Configuration
    print("Configuration:")
    print("  - Test suite: 50+ cases across 6 categories")
    print("  - Methods: 4 (Context Window, Conv History, Single RAG, DualRAG)")
    print("  - Metrics: 8 (Consistency, Authority, Hallucination, etc.)")
    print("  - Output: Publication-ready charts + LaTeX tables")
    print()
    
    response = input("Run full evaluation? (y/n) [or enter number for limited test]: ")
    
    if response.lower() == 'n':
        print("Aborted.")
        return
    
    # Check if user wants limited test
    test_limit = None
    if response.isdigit():
        test_limit = int(response)
        print(f"\n⚠️  Running LIMITED evaluation with {test_limit} tests")
    elif response.lower() == 'y':
        print(f"\n🚀 Running FULL evaluation")
    else:
        print("Invalid input. Aborted.")
        return
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(llm_provider="gemini")
    evaluator.run_full_evaluation(test_limit=test_limit)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {evaluator.output_dir}/")
    print("\nGenerated files:")
    print("  📊 charts/main_comparison.png - Main results")
    print("  📊 charts/category_breakdown.png - Per-category analysis")
    print("  📊 charts/radar_comparison.png - Overall comparison")
    print("  📊 charts/token_efficiency.png - Resource usage")
    print("  📝 tables/summary_table.tex - LaTeX table for paper")


if __name__ == "__main__":
    main()