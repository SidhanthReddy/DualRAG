"""
End-to-end test using a REAL LLM.

This test verifies:
1. Prompt construction
2. Real LLM invocation
3. Strict output parsing
4. Validator enforcement
5. State RAG commit

Expected behavior:
- Either a successful commit
- OR a clean, explainable failure (parser / validator)
"""

from orchestrator import Orchestrator
from state_rag_manager import StateRAGManager
from state_rag_enums import ArtifactSource
from artifact import Artifact
from state_rag_enums import ArtifactType, ArtifactSource

def reset_state():
    """
    Hard reset State RAG for clean testing.
    """
    mgr = StateRAGManager()
    mgr.artifacts = []
    mgr._persist()
    print("✓ State reset")


def main():
    print("\n=== REAL LLM ORCHESTRATOR TEST START ===\n")

    # --------------------------------------------------
    # Step 0: Reset state
    # --------------------------------------------------
    reset_state()

    # --------------------------------------------------
    # Step 1: Initialize orchestrator (REAL LLM)
    # Change provider to "openai" if needed
    # --------------------------------------------------
    orch = Orchestrator(llm_provider="gemini")
    print("✓ Orchestrator initialized with Gemini\n")

    # --------------------------------------------------
    # Step 2: Seed initial Navbar (AI-generated)
    # --------------------------------------------------
    mgr = orch.state_rag

    mgr.commit(
        Artifact(
            type=ArtifactType.component,
            name="Navbar",
            file_path="components/Navbar.tsx",
            content="<nav class='h-16 bg-white'>Navbar</nav>",
            language="tsx",
            source=ArtifactSource.ai_generated,
        )
    )

    print("✓ Seeded initial Navbar\n")

    # --------------------------------------------------
    # Step 3: User request
    # --------------------------------------------------
    user_request = "Make the navbar sticky and reduce its height"
    allowed_paths = ["components/Navbar.tsx"]

    print("USER REQUEST:")
    print(user_request)
    print("ALLOWED FILES:", allowed_paths, "\n")

    # --------------------------------------------------
    # Step 4: Run full pipeline
    # --------------------------------------------------
    try:
        committed = orch.handle_request(
            user_request=user_request,
            allowed_paths=allowed_paths,
        )

        print("\n✓ LLM execution completed")
        print("\nCommitted artifacts:")

        for a in committed:
            print(
                f"- {a.file_path} | "
                f"v{a.version} | "
                f"{a.source.value}"
            )

    except Exception as e:
        print("\n❌ PIPELINE FAILED (EXPECTED POSSIBLY)")
        print("Reason:")
        print(e)

    # --------------------------------------------------
    # Step 5: Inspect final state
    # --------------------------------------------------
    print("\nFINAL STATE SNAPSHOT:")
    for a in mgr.artifacts:
        print(
            f"{a.file_path} | "
            f"v{a.version} | "
            f"{a.source.value} | "
            f"active={a.is_active}"
        )

    print("\n=== REAL LLM ORCHESTRATOR TEST END ===\n")


if __name__ == "__main__":
    main()
