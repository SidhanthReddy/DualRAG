import os
import shutil
import traceback

from artifact import Artifact
from state_rag_enums import ArtifactType, ArtifactSource
from state_rag_manager import StateRAGManager
from orchestrator import Orchestrator
from global_rag import GlobalRAG
from global_rag_formatter import format_global_rag_for_prompt

# -----------------------------
# Helpers
# -----------------------------

def reset_state():
    if os.path.exists("state_rag"):
        shutil.rmtree("state_rag")
    print("✓ State reset")


def case(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}")
        traceback.print_exc()


# -----------------------------
# Test Cases
# -----------------------------

def test_commit_and_versioning():
    mgr = StateRAGManager()

    a1 = Artifact(
        type=ArtifactType.component,
        name="Navbar",
        file_path="components/Navbar.tsx",
        content="<nav>v1</nav>",
        language="tsx",
        source=ArtifactSource.ai_generated
    )
    mgr.commit(a1)

    a2 = Artifact(
        type=ArtifactType.component,
        name="Navbar",
        file_path="components/Navbar.tsx",
        content="<nav>v2 user</nav>",
        language="tsx",
        source=ArtifactSource.user_modified
    )
    mgr.commit(a2)

    active = [a for a in mgr.artifacts if a.is_active]
    assert len(active) == 1
    assert active[0].version == 2


def test_authority_enforcement():
    mgr = StateRAGManager()

    # First, commit user-modified Navbar
    user_nav = Artifact(
        type=ArtifactType.component,
        name="Navbar",
        file_path="components/Navbar.tsx",
        content="<nav>user</nav>",
        language="tsx",
        source=ArtifactSource.user_modified
    )
    mgr.commit(user_nav)

    # Now try AI override (should fail)
    bad = Artifact(
        type=ArtifactType.component,
        name="Navbar",
        file_path="components/Navbar.tsx",
        content="<nav>evil ai</nav>",
        language="tsx",
        source=ArtifactSource.ai_modified
    )

    try:
        mgr.commit(bad)
        raise AssertionError("Authority violation allowed")
    except ValueError:
        pass


def test_multi_file_isolation():
    mgr = StateRAGManager()

    hero = Artifact(
        type=ArtifactType.component,
        name="Hero",
        file_path="components/Hero.tsx",
        content="<section>Hero</section>",
        language="tsx",
        source=ArtifactSource.ai_generated
    )
    mgr.commit(hero)

    active = mgr.retrieve(scope=[ArtifactType.component])
    paths = {a.file_path for a in active}

    assert "components/Navbar.tsx" in paths
    assert "components/Hero.tsx" in paths


def test_dependency_expansion():
    mgr = StateRAGManager()

    layout = Artifact(
        type=ArtifactType.layout,
        name="MainLayout",
        file_path="layouts/Main.tsx",
        content="<Layout />",
        language="tsx",
        source=ArtifactSource.ai_generated
    )
    mgr.commit(layout)

    hero = Artifact(
        type=ArtifactType.component,
        name="Hero",
        file_path="components/Hero.tsx",
        content="<Hero />",
        language="tsx",
        source=ArtifactSource.ai_generated,
        dependencies=[layout.artifact_id]
    )
    mgr.commit(hero)

    results = mgr.retrieve(scope=[ArtifactType.component])
    paths = {a.file_path for a in results}

    assert "layouts/Main.tsx" in paths


def test_faiss_ranking():
    mgr = StateRAGManager()

    results = mgr.retrieve(
        scope=[ArtifactType.component],
        user_query="sticky navbar"
    )

    paths = [a.file_path for a in results]

    assert "components/Navbar.tsx" in paths
    assert "components/Hero.tsx" in paths

def test_global_rag_formatter():
    rag = GlobalRAG()
    entries = rag.retrieve("sticky navbar", k=2)
    formatted = format_global_rag_for_prompt(entries)

    assert "GLOBAL REFERENCES" in formatted


def test_orchestrator_prompt():
    orch = Orchestrator()

    prompt = orch.handle_request(
        user_request="Make navbar sticky",
        scope=[ArtifactType.component]
    )

    assert "PROJECT STATE" in prompt
    assert "GLOBAL REFERENCES" in prompt
    assert "USER REQUEST" in prompt


# -----------------------------
# Run All Tests
# -----------------------------

if __name__ == "__main__":
    print("\n=== SYSTEM STRESS TEST START ===\n")

    reset_state()

    case("Commit & Versioning", test_commit_and_versioning)
    case("Authority Enforcement", test_authority_enforcement)
    case("Multi-file Isolation", test_multi_file_isolation)
    case("Dependency Expansion", test_dependency_expansion)
    case("FAISS Ranking (State RAG)", test_faiss_ranking)
    case("Global RAG Formatter", test_global_rag_formatter)
    case("Orchestrator Prompt Assembly", test_orchestrator_prompt)

    print("\n=== SYSTEM STRESS TEST COMPLETE ===")
