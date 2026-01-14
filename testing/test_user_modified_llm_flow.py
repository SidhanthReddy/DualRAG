"""
This test demonstrates the full lifecycle:

1. User manually edits Navbar -> user_modified
2. User asks LLM to change Navbar colors
3. LLM proposes change
4. Validator allows change because user explicitly targeted Navbar
5. State RAG commits change while preserving user authority
"""

from datetime import datetime

from state_rag_manager import StateRAGManager
from state_rag_enums import ArtifactType, ArtifactSource
from artifact import Artifact
from validator import Validator, ProposedArtifact
from orchestrator import Orchestrator


def print_artifacts(title, artifacts):
    print(f"\n--- {title} ---")
    for a in artifacts:
        print(
            f"{a.file_path} | "
            f"v{a.version} | "
            f"{a.source.value} | "
            f"active={a.is_active}"
        )


def main():
    print("\n=== TEST: USER_MODIFIED FILE + LLM EDIT FLOW ===")

    # --------------------------------------------------
    # STEP 0: Fresh State
    # --------------------------------------------------
    print("\n[Step 0] Initializing State RAG")
    manager = StateRAGManager()

    # --------------------------------------------------
    # STEP 1: Initial AI-generated Navbar
    # --------------------------------------------------
    print("\n[Step 1] Initial AI-generated Navbar")

    navbar_v1 = Artifact(
        type=ArtifactType.component,
        name="Navbar",
        file_path="components/Navbar.tsx",
        content="<nav class='h-16 bg-white' />",
        language="tsx",
        source=ArtifactSource.ai_generated,
        version=1,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    manager.commit(navbar_v1)
    print_artifacts("After initial AI generation", manager.artifacts)

    # --------------------------------------------------
    # STEP 2: User manually edits Navbar height
    # --------------------------------------------------
    print("\n[Step 2] User manually edits Navbar height")

    navbar_v2 = Artifact(
        type=ArtifactType.component,
        name="Navbar",
        file_path="components/Navbar.tsx",
        content="<nav class='h-20 bg-white' />",
        language="tsx",
        source=ArtifactSource.user_modified,
    )

    manager.commit(navbar_v2)
    print_artifacts("After user manual edit", manager.artifacts)

    # --------------------------------------------------
    # STEP 3: User asks LLM to update Navbar colors
    # --------------------------------------------------
    print("\n[Step 3] User asks LLM to update Navbar colors")
    user_request = "Choose a color scheme for the navbar based on the home page"
    allowed_paths = ["components/Navbar.tsx"]

    print(f"User request: {user_request}")
    print(f"Allowed paths: {allowed_paths}")

    # --------------------------------------------------
    # STEP 4: LLM proposes change (simulated)
    # --------------------------------------------------
    print("\n[Step 4] LLM proposes updated Navbar")

    proposed = [
        ProposedArtifact(
            file_path="components/Navbar.tsx",
            content="<nav class='h-20 bg-blue-600 text-white' />",
            language="tsx",
        )
    ]

    print("Proposed artifact:")
    print(proposed[0].content)

    # --------------------------------------------------
    # STEP 5: Validator checks proposal
    # --------------------------------------------------
    print("\n[Step 5] Validator checks authority and scope")

    validator = Validator()
    active = manager.retrieve(file_paths=allowed_paths)

    result = validator.validate(
        proposed=proposed,
        active_artifacts=active,
        allowed_paths=allowed_paths,
    )

    if not result.ok:
        print("❌ Validation failed:", result.reason)
        return

    print("✅ Validation passed")

    # --------------------------------------------------
    # STEP 6: Commit validated artifact (user-authoritative)
    # --------------------------------------------------
    print("\n[Step 6] Committing validated artifact")

    old = active[0]
    source = (
        ArtifactSource.user_modified
        if old.source == ArtifactSource.user_modified
        else ArtifactSource.ai_modified
    )

    navbar_v3 = Artifact(
        type=ArtifactType.component,
        name="Navbar",
        file_path="components/Navbar.tsx",
        content=proposed[0].content,
        language="tsx",
        source=source,
    )

    manager.commit(navbar_v3)
    print_artifacts("After LLM-assisted user-approved edit", manager.artifacts)

    # --------------------------------------------------
    # STEP 7: Final Assertion (Human-readable)
    # --------------------------------------------------
    print("\n[Step 7] Final check")

    active_navbar = [
        a for a in manager.artifacts
        if a.file_path == "components/Navbar.tsx" and a.is_active
    ][0]

    print("\nActive Navbar:")
    print(f"Version: {active_navbar.version}")
    print(f"Source: {active_navbar.source.value}")
    print(f"Content: {active_navbar.content}")

    print("\n=== TEST COMPLETE: FLOW VERIFIED ===\n")


if __name__ == "__main__":
    main()
