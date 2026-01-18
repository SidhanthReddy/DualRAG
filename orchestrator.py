from typing import List

from state_rag_manager import StateRAGManager
from global_rag import GlobalRAG
from validator import Validator
from artifact import Artifact
from state_rag_enums import ArtifactSource

from llm_adapter import LLMAdapter
from llm_output_parser import parse_llm_output


class Orchestrator:
    """
    Central execution controller.

    Responsibilities:
    - Retrieve authoritative state (State RAG)
    - Retrieve advisory knowledge (Global RAG)
    - Build strict prompt
    - Invoke LLM (stateless)
    - Parse LLM output
    - Validate proposed changes
    - Commit validated artifacts
    """

    def __init__(self, llm_provider: str = "mock"):
        self.state_rag = StateRAGManager()
        self.global_rag = GlobalRAG()
        self.validator = Validator()
        self.llm = LLMAdapter(provider=llm_provider)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def handle_request(
        self,
        user_request: str,
        allowed_paths: List[str],
    ):
        """
        Executes one full user interaction.
        """

        # 1. Retrieve authoritative project state
        active_artifacts = self.state_rag.retrieve(
            file_paths=allowed_paths
        )

        # 2. Retrieve advisory global knowledge
        global_refs = self.global_rag.retrieve(
            query=user_request,
            k=3
        )

        # 3. Build strict prompt
        prompt = self._build_prompt(
            user_request=user_request,
            active_artifacts=active_artifacts,
            global_refs=global_refs,
            allowed_paths=allowed_paths,
        )

        # 4. Invoke LLM (stateless)
        raw_output = self.llm.generate(prompt)

        # 5. Parse LLM output (strict contract)
        proposed = parse_llm_output(raw_output)

        # 6. Validate proposed changes
        result = self.validator.validate(
            proposed=proposed,
            active_artifacts=active_artifacts,
            allowed_paths=allowed_paths,
        )

        if not result.ok:
            raise RuntimeError(
                f"Validation failed: {result.reason}"
            )

        # 7. Commit validated artifacts
        committed = []

        for p in result.artifacts:
            old = next(
                (a for a in active_artifacts if a.file_path == p.file_path),
                None
            )

            # Preserve user authority if user explicitly allowed the edit
            if old and old.source == ArtifactSource.user_modified:
                source = ArtifactSource.user_modified
            else:
                source = ArtifactSource.ai_modified

            artifact = Artifact(
                type=self._infer_type(p.file_path),
                name=p.file_path.split("/")[-1],
                file_path=p.file_path,
                content=p.content,
                language=p.language,
                source=source,
            )

            committed.append(
                self.state_rag.commit(artifact)
            )

        return committed

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _build_prompt(
        self,
        user_request: str,
        active_artifacts,
        global_refs,
        allowed_paths,
    ) -> str:
        """
        Constructs a strict, authority-aware prompt for the LLM.
        """

        parts = []

        parts.append(
            "SYSTEM:\n"
            "You are an AI website builder.\n"
            "You are stateless.\n"
            "PROJECT STATE is authoritative.\n"
            "GLOBAL REFERENCES are advisory.\n"
            "Modify only explicitly allowed files.\n"
            "Output full updated files only.\n"
        )

        parts.append("\nPROJECT STATE (AUTHORITATIVE):\n")
        for a in active_artifacts:
            parts.append(f"--- {a.file_path} ---\n{a.content}\n")

        parts.append("\nGLOBAL REFERENCES (ADVISORY):\n")
        for i, ref in enumerate(global_refs, 1):
            parts.append(f"{i}. {ref.title}\n{ref.content}\n")

        parts.append("\nALLOWED FILES:\n")
        for p in allowed_paths:
            parts.append(f"- {p}")

        parts.append("\nUSER REQUEST:\n")
        parts.append(user_request)

        parts.append(
            "\nOUTPUT FORMAT:\n"
            "FILE: <file_path>\n"
            "<full file content>\n"
        )

        return "\n".join(parts)

    def _infer_type(self, file_path: str):
        if "components/" in file_path:
            return "component"
        if "app/" in file_path:
            return "page"
        return "config"
